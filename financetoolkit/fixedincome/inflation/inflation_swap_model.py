"""Inflation Swap Model

Zero-coupon and year-on-year inflation swaps and the forward inflation index they imply
(Kerkhof, 2005; Brigo and Mercurio, 2006, ch. 15). An inflation swap exchanges a fixed rate
for the growth of a price index, such as the US CPI-U or the euro area HICP excluding tobacco.

Every function takes its curves as the Toolkit's curves are returned: annually compounded
rates by maturity in years, interpolated linearly in between and held at the first and last
rate outside them (see `financetoolkit.fixedincome.helpers.discount_factors`). The nominal
zero-coupon curve discounts; the zero-coupon inflation curve holds the zero-coupon inflation
swap rates z(T), so that the forward index grows as I(T) / I(0) = (1 + z(T))^T. Times are
year fractions from the base of the index I(0) and the index observed at time T is I(T): the
observation lag of the index (three months for the CPI-U of TIPS and US inflation swaps) is
left to the dates the caller measures the curves on.
"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.utilities.error_model import check_option

MONTHS_PER_YEAR = 12

SWAP_TYPES = ("zero-coupon", "year-on-year")


def _index_growth(
    zero_inflation_curve: pd.Series, years: float | np.ndarray
) -> np.ndarray:
    """
    The growth of the forward index from today, I(T) / I(0) = (1 + z(T))^T.

    The zero-coupon inflation rates compound annually like the zero-coupon rates of a
    nominal curve, so the growth is the inverse of the "discount factor" of the curve.

    Args:
        zero_inflation_curve (pd.Series): The zero-coupon inflation swap rates by maturity.
        years (float | np.ndarray): The times in years.

    Returns:
        np.ndarray: The growth of the index to each time.
    """
    return 1 / discount_factors(zero_inflation_curve, years)


def get_zero_coupon_inflation_swap_rate(
    zero_curve: pd.Series,
    real_zero_curve: pd.Series,
    maturities: float | list | np.ndarray,
) -> pd.Series:
    """
    Calculate the fair rate of zero-coupon inflation swaps from a nominal and a real
    zero-coupon curve: the fixed rate, compounded to maturity, that the market swaps against
    the realized inflation, its expected inflation.

    A zero-coupon inflation swap pays, at its maturity T, the growth of the price index
    I(T) / I(0) - 1 against the fixed amount (1 + K)^T - 1, both on the notional. Its fair
    rate K is the rate at which the two legs are worth the same today. With the real
    zero-coupon curve of inflation-linked bonds, the forward index is
    I(0) * P_real(T) / P_nominal(T) (Jarrow and Yildirim, 2003), so the fair rate is the
    zero-coupon breakeven inflation rate, the Fisher relation of the two zero rates.

    The formula is as follows:

    - (1 + K(T))^T = E^T[I(T)] / I(0) = P_real(T) / P_nominal(T)
    - K(T) = (1 + n(T)) / (1 + r(T)) - 1

    where n(T) and r(T) are the annually compounded nominal and real zero rates of maturity T
    and K(T) is annually compounded as well. The rate is the zero-coupon breakeven of the
    bonds: swaps trade above it by the inflation swap basis (the liquidity premium of
    inflation-linked bonds and the funding of the bond that hedges the swap), so a quoted
    swap curve takes precedence where one is available.

    For more information, see the following sources:

    - Jarrow, R. and Yildirim, Y. (2003). "Pricing Treasury Inflation Protected Securities
      and Related Derivatives using an HJM Model." Journal of Financial and Quantitative
      Analysis, 38(2), 337-358.
    - Kerkhof, J. (2005). "Inflation Derivatives Explained." Lehman Brothers Fixed Income
      Quantitative Research.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, ch. 15.

    QuantLib's fair rate is taken on a zero inflation term structure of the forward index
    ratios P_real(T) / P_nominal(T).

    Referenced against:
    - QuantLib, ZeroCouponInflationSwap.fairRate: https://www.quantlib.org/reference/class_quant_lib_1_1_zero_coupon_inflation_swap.html

    Args:
        zero_curve (pd.Series): The annually compounded nominal zero-coupon rates (decimals),
            indexed by maturity in years.
        real_zero_curve (pd.Series): The annually compounded real zero-coupon rates
            (decimals) of the inflation-linked bonds, indexed by maturity in years.
        maturities (float | list | np.ndarray): The maturities of the swaps in years.

    Returns:
        pd.Series: The zero-coupon inflation swap rates (annually compounded decimals),
        indexed by maturity.

    Raises:
        ValueError: If a maturity is not positive.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.inflation import inflation_swap_model

    nominal = pd.Series([0.040, 0.042, 0.045], index=[1, 5, 10])
    real = pd.Series([0.015, 0.018, 0.021], index=[1, 5, 10])

    inflation_swap_model.get_zero_coupon_inflation_swap_rate(nominal, real, [1, 5, 10])
    ```

    Which returns:

    ```
    1     0.024631
    5     0.023576
    10    0.023506
    Name: Zero-Coupon Inflation Swap Rate, dtype: float64
    ```
    """
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    if np.any(maturities <= 0):
        raise ValueError("The maturities of inflation swaps must be positive.")

    nominal_discount = discount_factors(zero_curve, maturities)
    real_discount = discount_factors(real_zero_curve, maturities)

    return pd.Series(
        (real_discount / nominal_discount) ** (1 / maturities) - 1,
        index=[int(m) if float(m).is_integer() else m for m in maturities],
        name="Zero-Coupon Inflation Swap Rate",
    )


def get_forward_year_on_year_inflation_rates(
    zero_inflation_curve: pd.Series, years: int
) -> pd.Series:
    """
    Calculate the forward year-on-year inflation rates of the coming years from the
    zero-coupon inflation curve, without a convexity adjustment: the inflation the market
    expects in each coming year.

    The year-on-year rate of year i is the growth of the index over that year,
    I(i) / I(i - 1) - 1. It is fixed only at the end of the year and paid then, so its
    expectation under the payment date's forward measure differs from the ratio of the
    forward indices by a convexity adjustment that depends on the volatilities of the
    real and nominal rates and their correlation (Jarrow and Yildirim, 2003; Mercurio,
    2005). This function leaves that adjustment out and returns the ratio of the forward
    indices, as a market without year-on-year option prices to calibrate it has to: the
    adjustment is of the order of a few basis points for maturities up to ten years.

    The formula is as follows:

    - f_i = (1 + z(i))^i / (1 + z(i - 1))^(i - 1) - 1, for i = 1, ..., years

    where z(t) is the annually compounded zero-coupon inflation swap rate of maturity t, so
    f_1 = z(1) and the f_i compound back to the zero-coupon rates:
    PRODUCT(1 + f_i) = (1 + z(n))^n.

    For more information, see the following sources:

    - Mercurio, F. (2005). "Pricing Inflation-Indexed Derivatives." Quantitative Finance,
      5(3), 289-302.
    - Kerkhof, J. (2005). "Inflation Derivatives Explained." Lehman Brothers Fixed Income
      Quantitative Research.

    Args:
        zero_inflation_curve (pd.Series): The annually compounded zero-coupon inflation swap
            rates (decimals), indexed by maturity in years.
        years (int): The number of years.

    Returns:
        pd.Series: The forward year-on-year inflation rate (decimal) of each year, indexed
        by the year's end (1, 2, ..., years).

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.inflation import inflation_swap_model

    zero_inflation = pd.Series([0.030, 0.026, 0.024], index=[1, 3, 5])

    inflation_swap_model.get_forward_year_on_year_inflation_rates(zero_inflation, 5)
    ```

    Which returns:

    ```
    1    0.030000
    2    0.026004
    3    0.022012
    4    0.022006
    5    0.020010
    Name: Forward Year-on-Year Inflation Rate, dtype: float64
    ```
    """
    ends = np.arange(1, int(years) + 1, dtype=float)
    growth = _index_growth(zero_inflation_curve, np.concatenate([[0.0], ends]))

    return pd.Series(
        growth[1:] / growth[:-1] - 1,
        index=ends.astype(int),
        name="Forward Year-on-Year Inflation Rate",
    )


def get_year_on_year_inflation_swap_rate(
    zero_curve: pd.Series,
    zero_inflation_curve: pd.Series,
    maturities: int | list | np.ndarray,
) -> pd.Series:
    """
    Calculate the fair rate of year-on-year inflation swaps: the fixed rate paid every year
    against that year's inflation, I(i) / I(i - 1) - 1, at which the swap is worth nothing
    today.

    The fair rate is the average of the forward year-on-year rates, weighted by the nominal
    discount factor of each payment (Kerkhof, 2005; Brigo and Mercurio, 2006, ch. 15). The
    forward rates are the ratios of the forward indices, without the convexity adjustment of
    a rate that is fixed and paid at the end of its year (see
    `get_forward_year_on_year_inflation_rates`); the swap rate is therefore the one of a
    market with deterministic real rates.

    The formula is as follows:

    - f_i = (1 + z(i))^i / (1 + z(i - 1))^(i - 1) - 1
    - YoY Swap Rate(n) = SUM(P(i) * f_i) / SUM(P(i)), over i = 1, ..., n

    where P(i) = (1 + r(i))^(-i) is the discount factor of the annually compounded nominal
    zero rate r(i) and z(i) the annually compounded zero-coupon inflation swap rate. Both
    legs pay once a year, so each accrues a full year.

    For more information, see the following sources:

    - Kerkhof, J. (2005). "Inflation Derivatives Explained." Lehman Brothers Fixed Income
      Quantitative Research.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, ch. 15.
    - Mercurio, F. (2005). "Pricing Inflation-Indexed Derivatives." Quantitative Finance,
      5(3), 289-302.

    QuantLib's fair rate is taken on a year-on-year inflation term structure of the same
    forward rates.

    Referenced against:
    - QuantLib, YearOnYearInflationSwap.fairRate: https://www.quantlib.org/reference/class_quant_lib_1_1_year_on_year_inflation_swap.html

    Args:
        zero_curve (pd.Series): The annually compounded nominal zero-coupon rates
            (decimals), indexed by maturity in years.
        zero_inflation_curve (pd.Series): The annually compounded zero-coupon inflation swap
            rates (decimals), indexed by maturity in years.
        maturities (int | list | np.ndarray): The maturities of the swaps, in whole years.

    Returns:
        pd.Series: The year-on-year inflation swap rates (decimals), indexed by maturity.

    Raises:
        ValueError: If a maturity is not a positive whole number of years.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.inflation import inflation_swap_model

    nominal = pd.Series([0.040, 0.042, 0.045], index=[1, 5, 10])
    zero_inflation = pd.Series([0.030, 0.026, 0.024], index=[1, 3, 5])

    inflation_swap_model.get_year_on_year_inflation_swap_rate(nominal, zero_inflation, [1, 5, 10])
    ```

    Which returns:

    ```
    1     0.030000
    5     0.024206
    10    0.024114
    Name: Year-on-Year Inflation Swap Rate, dtype: float64
    ```
    """
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    if np.any(maturities < 1) or not np.all(np.mod(maturities, 1) == 0):
        raise ValueError(
            "A year-on-year inflation swap pays once a year: its maturities must be "
            "positive whole numbers of years."
        )

    forward_rates = get_forward_year_on_year_inflation_rates(
        zero_inflation_curve, int(maturities.max())
    ).to_numpy()
    discount = discount_factors(
        zero_curve, np.arange(1, len(forward_rates) + 1, dtype=float)
    )
    cumulative_annuity = np.cumsum(discount)
    cumulative_leg = np.cumsum(discount * forward_rates)
    positions = maturities.astype(int) - 1

    return pd.Series(
        cumulative_leg[positions] / cumulative_annuity[positions],
        index=maturities.astype(int),
        name="Year-on-Year Inflation Swap Rate",
    )


def get_inflation_seasonality(consumer_price_index: pd.Series) -> pd.Series:
    """
    Estimate the multiplicative seasonal factors of a monthly price index, normalised so
    that the twelve factors multiply to 1: how much faster or slower prices rise in each
    month than over the year.

    Prices rise faster in some months than in others (in the US CPI-U, the first half of
    the year more than the last months), while a zero-coupon inflation swap only fixes the
    growth between its whole-year maturities. A forward index between them follows the trend
    of the swap curve times the seasonal factor of its month relative to that of the base
    month (Belgrade, Benhamou and Koehler, 2004). The factors are estimated from the
    month-over-month changes of the index: the average log change of each calendar month
    above the average log change of all months, accumulated over the year.

    The formula is as follows:

    - s_m = MEAN(ln(I_t / I_(t-1)) for t in calendar month m) - MEAN(ln(I_t / I_(t-1)))
    - S_m = exp(s_1 + ... + s_m) / GEOMETRIC MEAN(exp(s_1 + ... + s_k), k = 1, ..., 12)

    where the twelve s_m sum to zero, so a whole year of seasonal factors cancels out and
    the index keeps the growth of the swap curve at whole years. The level of S_m is the
    index of month m above (S_m > 1) or below its trend.

    For more information, see the following sources:

    - Belgrade, N., Benhamou, E. and Koehler, E. (2004). "A Market Model for Inflation."
      SSRN working paper 576081.
    - Kerkhof, J. (2005). "Inflation Derivatives Explained." Lehman Brothers Fixed Income
      Quantitative Research, section on seasonality.

    Shares similar logic:
    - QuantLib, MultiplicativePriceSeasonality: https://www.quantlib.org/reference/class_quant_lib_1_1_multiplicative_price_seasonality.html

    Args:
        consumer_price_index (pd.Series): The price index, not seasonally adjusted, indexed
            by monthly period, covering at least a year of month-over-month changes.

    Returns:
        pd.Series: The seasonal factor of each calendar month (1 to 12).

    Raises:
        ValueError: If a calendar month has no month-over-month change.

    As an example:

    ```python
    import numpy as np
    import pandas as pd

    from financetoolkit.fixedincome.inflation import inflation_swap_model

    months = pd.period_range("2022-01", "2025-12", freq="M")
    cpi = pd.Series(
        100 * np.exp(0.0025 * np.arange(48) + 0.003 * np.sin(2 * np.pi * months.month / 12)),
        index=months,
    )

    inflation_swap_model.get_inflation_seasonality(cpi).round(4)
    ```

    Which returns:

    ```
    1     1.0015
    2     1.0026
    3     1.0030
    4     1.0026
    5     1.0015
    6     1.0000
    7     0.9985
    8     0.9974
    9     0.9970
    10    0.9974
    11    0.9985
    12    1.0000
    Name: Seasonal Factor, dtype: float64
    ```
    """
    index = consumer_price_index.dropna().sort_index()
    index = index[index > 0]
    months = pd.PeriodIndex(index.index, freq="M")
    changes = pd.Series(np.log(index.to_numpy(dtype=float)), index=months).diff()
    # Only changes between consecutive months, so a gap in the history is not a month's change.
    consecutive = np.concatenate([[False], np.diff(months.asi8) == 1])
    changes = changes[consecutive]

    by_month = changes.groupby(changes.index.month).mean()
    if len(by_month) < MONTHS_PER_YEAR:
        raise ValueError(
            "The seasonality needs a month-over-month change of the index in every calendar "
            "month: give at least thirteen consecutive months."
        )

    seasonal = by_month.reindex(range(1, MONTHS_PER_YEAR + 1)) - by_month.mean()
    levels = np.cumsum(seasonal.to_numpy())

    return pd.Series(
        np.exp(levels - levels.mean()),
        index=range(1, MONTHS_PER_YEAR + 1),
        name="Seasonal Factor",
    )


def get_forward_inflation_index(
    base_index: float,
    zero_inflation_curve: pd.Series,
    months: int = 120,
    base_month: int = 1,
    seasonal_factors: pd.Series | None = None,
) -> pd.Series:
    """
    Calculate the forward price index of each coming month from the zero-coupon inflation
    curve, with or without monthly seasonality: the price level the inflation market expects
    in each month.

    The forward index of a month is the index the zero-coupon inflation swaps imply for it:
    the base index grown at the zero-coupon inflation rate of that horizon. With seasonal
    factors, each month moves by the ratio of its seasonal factor to that of the base month,
    which leaves whole-year horizons on the swap curve, as the twelve factors multiply to 1
    (Belgrade, Benhamou and Koehler, 2004; Kerkhof, 2005).

    The formula is as follows:

    - I(m) = I(0) * (1 + z(m / 12))^(m / 12) * S_(month of m) / S_(base month)

    where I(0) is the base index, m the months after the base month, z(t) the annually
    compounded zero-coupon inflation rate of t years, interpolated linearly between the
    maturities of the curve, and S the seasonal factors (all 1 without seasonality).

    Referenced against:
    - QuantLib, ZeroInflationIndex: https://www.quantlib.org/reference/class_quant_lib_1_1_zero_inflation_index.html
    - QuantLib, MultiplicativePriceSeasonality: https://www.quantlib.org/reference/class_quant_lib_1_1_multiplicative_price_seasonality.html

    Args:
        base_index (float): The index of the base month, such as the latest published CPI or
            the reference CPI of the swaps.
        zero_inflation_curve (pd.Series): The annually compounded zero-coupon inflation swap
            rates (decimals), indexed by maturity in years from the base month.
        months (int, optional): The number of months after the base month. Defaults to 120.
        base_month (int, optional): The calendar month (1 to 12) of the base index, which
            sets the seasonal factor of every month after it. Defaults to 1 (January).
        seasonal_factors (pd.Series | None, optional): The seasonal factor of each calendar
            month (1 to 12), such as those of `get_inflation_seasonality`. Defaults to None,
            which leaves seasonality out.

    Returns:
        pd.Series: The forward index of each month after the base month (1, 2, ..., months).

    Raises:
        ValueError: If the base month is not between 1 and 12, or the seasonal factors do not
            cover every calendar month.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.inflation import inflation_swap_model

    zero_inflation = pd.Series([0.030, 0.026, 0.024], index=[1, 3, 5])
    seasonality = pd.Series(
        [0.998, 1.001, 1.003, 1.004, 1.004, 1.003, 1.001, 1.0, 0.999, 0.998, 0.996, 0.993],
        index=range(1, 13),
    )

    inflation_swap_model.get_forward_inflation_index(
        320.0, zero_inflation, months=6, base_month=10, seasonal_factors=seasonality
    ).round(3)
    ```

    Which returns:

    ```
    1    320.146
    2    319.969
    3    322.373
    4    324.140
    5    325.589
    6    326.717
    Name: Forward Index, dtype: float64
    ```
    """
    if not 1 <= base_month <= MONTHS_PER_YEAR:
        raise ValueError(f"The base month must be between 1 and 12, not {base_month}.")

    steps = np.arange(1, int(months) + 1)
    forward_index = base_index * _index_growth(
        zero_inflation_curve, steps / MONTHS_PER_YEAR
    )

    if seasonal_factors is not None:
        factors = seasonal_factors.reindex(range(1, MONTHS_PER_YEAR + 1)).to_numpy(
            dtype=float
        )
        if not np.all(np.isfinite(factors) & (factors > 0)):
            raise ValueError(
                "The seasonal factors must hold a positive factor for every calendar month "
                "(1 to 12)."
            )
        calendar_months = (base_month - 1 + steps) % MONTHS_PER_YEAR
        forward_index = (
            forward_index * factors[calendar_months] / factors[base_month - 1]
        )

    return pd.Series(forward_index, index=steps, name="Forward Index")


def get_inflation_swap_value(
    zero_curve: pd.Series,
    zero_inflation_curve: pd.Series,
    fixed_rate: float,
    years_to_maturity: float,
    notional: float = 10_000_000,
    method: str = "zero-coupon",
    index_ratio: float = 1.0,
    years_elapsed: float = 0.0,
    is_receiver: bool = False,
) -> dict[str, float]:
    """
    Calculate the value of a zero-coupon or year-on-year inflation swap, a new one or one
    that is already running: what the swap is worth to the inflation receiver given the
    inflation the market now expects.

    The inflation leg pays the growth of the price index and the fixed leg the fixed rate.
    The swap is valued from the side of the inflation receiver (the fixed rate payer), as
    the counterparty that hedges inflation-linked liabilities usually is; `is_receiver`
    turns it around (Kerkhof, 2005; Brigo and Mercurio, 2006, ch. 15).

    The formula is as follows:

    - Zero-coupon: Inflation Leg = N * (R * (1 + z(T))^T - 1) * P(T), Fixed Leg =
      N * ((1 + K)^(T + E) - 1) * P(T)
    - Year-on-year: Inflation Leg = N * SUM(f_i * P(t_i)), Fixed Leg = N * K * SUM(P(t_i)),
      with f_1 = R * (1 + z(t_1))^(t_1) - 1 and f_i = (1 + z(t_i))^(t_i) /
      (1 + z(t_(i-1)))^(t_(i-1)) - 1
    - Value = Inflation Leg - Fixed Leg (the negative for the fixed rate receiver)

    where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded nominal
    zero rate, z(t) the annually compounded zero-coupon inflation swap rate from today, T the
    years to maturity, E the years since the start of the swap and R the index ratio: the
    index today over the base index of the swap (zero-coupon) or of the current year
    (year-on-year). The year-on-year payments t_i are counted back from maturity in whole
    years, so a running swap's first payment comes within a year and pays the full year's
    inflation. The year-on-year rates carry no convexity adjustment (see
    `get_forward_year_on_year_inflation_rates`).

    Referenced against:
    - QuantLib, ZeroCouponInflationSwap (zero-coupon): https://www.quantlib.org/reference/class_quant_lib_1_1_zero_coupon_inflation_swap.html
    - QuantLib, YearOnYearInflationSwap (year-on-year): https://www.quantlib.org/reference/class_quant_lib_1_1_year_on_year_inflation_swap.html
    - QuantLib, DiscountingSwapEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_discounting_swap_engine.html

    Args:
        zero_curve (pd.Series): The annually compounded nominal zero-coupon rates
            (decimals), indexed by maturity in years.
        zero_inflation_curve (pd.Series): The annually compounded zero-coupon inflation swap
            rates (decimals) from today, indexed by maturity in years.
        fixed_rate (float): The fixed rate K of the swap (decimal, annually compounded for a
            zero-coupon swap).
        years_to_maturity (float): The years until the swap's last payment.
        notional (float, optional): The notional of the swap. Defaults to 10,000,000.
        method (str, optional): The kind of swap. Options are: "zero-coupon" (one payment of
            the index growth at maturity), "year-on-year" (a payment of each year's inflation).
            Defaults to "zero-coupon".
        index_ratio (float, optional): The index today over the base index of the swap
            (zero-coupon) or of the current year (year-on-year). Defaults to 1, a new swap.
        years_elapsed (float, optional): The years since the start of a zero-coupon swap,
            over which its fixed leg compounds as well. Defaults to 0, a new swap.
        is_receiver (bool, optional): Whether the swap receives the fixed rate (and pays
            inflation). Defaults to False, the inflation receiver.

    Returns:
        dict[str, float]: The Value, the Inflation Leg and the Fixed Leg present values, in
        the currency of the notional.

    Raises:
        ValueError: If the method is unknown or the years to maturity are not positive.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.inflation import inflation_swap_model

    nominal = pd.Series([0.040, 0.042, 0.045], index=[1, 5, 10])
    zero_inflation = pd.Series([0.030, 0.026, 0.024], index=[1, 3, 5])

    inflation_swap_model.get_inflation_swap_value(
        nominal, zero_inflation, fixed_rate=0.023, years_to_maturity=5
    )
    ```

    Which returns:

    ```
    {'Value': 44666.61124740451, 'Inflation Leg': 1024912.5567614173, 'Fixed Leg': 980245.9455140128}
    ```
    """
    check_option("method", method, SWAP_TYPES)
    if years_to_maturity <= 0:
        raise ValueError(
            f"The years to maturity of a swap must be positive, not {years_to_maturity}."
        )

    if method == "zero-coupon":
        discount = float(discount_factors(zero_curve, years_to_maturity))
        growth = float(_index_growth(zero_inflation_curve, years_to_maturity))
        inflation_leg = (index_ratio * growth - 1) * discount
        fixed_leg = (
            (1 + fixed_rate) ** (years_to_maturity + years_elapsed) - 1
        ) * discount
    else:
        times = bond_model.get_cash_flow_periods(years_to_maturity, 1)
        growth = _index_growth(zero_inflation_curve, np.concatenate([[0.0], times]))
        forward_rates = growth[1:] / growth[:-1] - 1
        forward_rates[0] = index_ratio * growth[1] - 1
        discount = discount_factors(zero_curve, times)
        inflation_leg = float(np.sum(forward_rates * discount))
        fixed_leg = float(fixed_rate * np.sum(discount))

    sign = -1 if is_receiver else 1

    return {
        "Value": sign * notional * (inflation_leg - fixed_leg),
        "Inflation Leg": notional * inflation_leg,
        "Fixed Leg": notional * fixed_leg,
    }
