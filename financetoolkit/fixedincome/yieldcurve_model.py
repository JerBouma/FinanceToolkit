"""Yield Curve Model Module"""

import numpy as np
import pandas as pd

from financetoolkit.fixedincome import bond_model

# pylint: disable=unsubscriptable-object


def get_forward_rate(
    near_rate: float | pd.Series,
    far_rate: float | pd.Series,
    near_maturity: float | pd.Series,
    far_maturity: float | pd.Series,
) -> float | pd.Series:
    """
    Calculate the implied forward rate between two points on a yield curve.

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
    `get_par_yield` and `bond_model._get_bond_price_from_curve`, which read the same kind
    of spot curve as nominal annual rates compounded `frequency` times per year. The two
    coincide at frequency=1; at any other frequency a curve should not be passed to both
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
        ValueError: If the far maturity is not greater than the near maturity.
    """
    for value in (near_rate, far_rate, near_maturity, far_maturity):
        if not isinstance(value, int | float | pd.Series):
            raise TypeError(
                "Expected near_rate, far_rate, near_maturity and far_maturity to be "
                f"a float, int or pandas Series, received {type(value)} instead."
            )

    if (
        isinstance(near_maturity, int | float)
        and isinstance(far_maturity, int | float)
        and far_maturity <= near_maturity
    ):
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
    Calculate the par yield for a given maturity given a zero-coupon (spot) yield curve.

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
    compounded at `frequency`, consistent with `bond_model._get_bond_price_from_curve`.

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
    if not isinstance(frequency, int):
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


def get_yield_curve_spread(
    long_yield: float | pd.Series, short_yield: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the spread between two points on the yield curve.

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


def get_breakeven_inflation_rate(
    nominal_yield: float | pd.Series, real_yield: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the breakeven inflation rate implied by a nominal and a real (inflation-protected) yield.

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
