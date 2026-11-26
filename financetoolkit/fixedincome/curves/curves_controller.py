"""Yield Curves Module"""

__docformat__ = "google"

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from financetoolkit import helpers
from financetoolkit.economics.helpers import (
    buffered_start_date,
    resample_to_period,
    validate_period,
)
from financetoolkit.economics.sources import (
    boe_model,
    ecb_model as economics_ecb_model,
    nyfed_model,
    oecd_model,
    treasury_model,
)
from financetoolkit.fixedincome import helpers as fixedincome_helpers
from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.curves import (
    curve_model,
    interpolation_model,
    smith_wilson_model,
    yieldcurve_model,
)
from financetoolkit.fixedincome.helpers import maturity_key, parse_maturities
from financetoolkit.fixedincome.sources import eiopa_model, fmp_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
    validate_window,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding, finalize_dataset

# ruff: noqa: E501

logger = logger_model.get_logger()


def _maturity_in_months(label: str) -> float:
    """
    Converts a maturity label such as "3M" or "10Y" into months, for sorting a curve.

    Args:
        label (str): The maturity label.

    Returns:
        float: The maturity in months.
    """
    return float(label[:-1]) * (12 if label.endswith("Y") else 1)


# The Euro Area and German curves are estimated as zero-coupon curves; the others are par
# yields of coupon bonds, paying twice a year except in Sweden and Norway.
ZERO_COUPON_CURVES = ("Euro Area", "Germany")

COUPON_FREQUENCIES = {"Sweden": 1, "Norway": 1}

# How the countries quote their bill yields (up to a year), see curve_model.bill_yields_to_annual:
# the US Treasury's bond-equivalent yields (simple interest up to half a year), simple interest
# on ACT/365 for Government of Canada bills and on ACT/360 for Swedish statsskuldväxlar (the
# Swedish money market calculation principles of April 2001). The other countries' yields are
# compounded with the coupon frequency.
BILL_CONVENTIONS = {
    "United States": "bond-equivalent",
    "Canada": "act/365",
    "Sweden": "act/360",
}

# The share of the dates a maturity must cover to be part of a principal component analysis.
CURVE_COVERAGE = 0.9


@filter_shown_columns
@track_sources
class CurvesFixedIncome:
    """
    Yield curves: government bond yields and curves, Treasury rates, forward rates, par yields, spreads, breakeven inflation, zero-coupon curves, Nelson-Siegel-Svensson fits, carry and roll-down, principal components, the term premium, the Bank of England curves and the EIOPA risk-free rates.
    """

    def _latest_zero_curve(
        self, country: str, date: str | None = None, coupon_grid: bool = True
    ) -> pd.Series:
        """
        The zero-coupon curve of a country on a day, see `_zero_curve`.

        Args:
            country (str): The country, see get_zero_coupon_curve.
            date (str | None): The day (YYYY-MM-DD), or the last day before it with a curve.
                Defaults to None, the end date.
            coupon_grid (bool): Whether to return the zero rate of every coupon date, or only
                those of the quoted maturities (the points key rate durations shock).
                Defaults to True.

        Returns:
            pd.Series: The annually compounded zero-coupon rates by maturity in years.
        """
        return self._zero_curve(
            self._latest_yield_curve(country, date), country, coupon_grid
        )

    @staticmethod
    def _zero_curve(
        par_curve: pd.Series, country: str, coupon_grid: bool = True
    ) -> pd.Series:
        """
        The zero-coupon curve of one day of a country's yield curve, bootstrapped from its
        par yields as `get_zero_coupon_curve` does, with the zero rate of every coupon date
        rather than only those of the quoted maturities. Interpolating between the quoted
        maturities alone would misprice par bonds at long maturities (a 20-year par bond at
        about 99.95), as the bootstrap interpolates the par yields, not the zero rates.

        Args:
            par_curve (pd.Series): The yields as decimals by maturity label.
            country (str): The country, which sets the coupon frequency and the quote
                convention of the bills.
            coupon_grid (bool): Whether to return the zero rate of every coupon date, or only
                those of the quoted maturities (the points key rate durations shock).
                Defaults to True.

        Returns:
            pd.Series: The annually compounded zero-coupon rates by maturity in years.
        """
        par_curve = par_curve.dropna()

        if country in ZERO_COUPON_CURVES:
            zero_curve = CurvesFixedIncome._to_zero_coupon(
                par_curve.to_frame().T, country
            ).iloc[0]
            return pd.Series(
                zero_curve.to_numpy(dtype=float),
                index=[maturity_key(label) for label in zero_curve.index],
            )

        maturities = np.array(
            [curve_model.maturity_in_years(label) for label in par_curve.index]
        )
        if par_curve.empty or maturities.min() > 1:
            raise ValueError(
                f"The {country} yield curve has no maturity of a year or less to bootstrap "
                "zero-coupon rates from."
            )
        bootstrap = curve_model.bootstrap_zero_rates(
            maturities,
            par_curve.to_numpy(dtype=float),
            COUPON_FREQUENCIES.get(country, 2),
            BILL_CONVENTIONS.get(country),
            return_grid=coupon_grid,
        )
        if not coupon_grid:
            return pd.Series(
                bootstrap, index=[maturity_key(label) for label in par_curve.index]
            )

        years, zero_rates = bootstrap
        return pd.Series(zero_rates, index=years)

    @staticmethod
    def _curve_nodes(
        par_curve: pd.Series, country: str, interpolation: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The maturities and continuously compounded zero rates of one day of a country's
        yield curve, bootstrapped with an interpolation, see
        interpolation_model.bootstrap_zero_curve. The Euro Area and German curves are
        zero-coupon curves already, published continuously compounded.

        Args:
            par_curve (pd.Series): The yields as decimals by maturity label.
            country (str): The country, which sets the coupon frequency and the quote
                convention of the bills.
            interpolation (str): One of interpolation_model.INTERPOLATIONS.

        Returns:
            tuple[np.ndarray, np.ndarray]: The maturities in years and their zero rates.
        """
        par_curve = par_curve.dropna()
        maturities = np.array(
            [curve_model.maturity_in_years(label) for label in par_curve.index]
        )
        if country in ZERO_COUPON_CURVES:
            return maturities, par_curve.to_numpy(dtype=float)
        if par_curve.empty or maturities.min() > 1:
            raise ValueError(
                f"The {country} yield curve has no maturity of a year or less to bootstrap "
                "zero-coupon rates from."
            )
        return interpolation_model.bootstrap_zero_curve(
            maturities,
            par_curve.to_numpy(dtype=float),
            COUPON_FREQUENCIES.get(country, 2),
            BILL_CONVENTIONS.get(country),
            interpolation,
        )

    @handle_errors
    def get_forward_rate(
        self,
        near_maturity: float | str | list | None = None,
        far_maturity: float | str | list | None = None,
        country: str = "United States",
        date: str | None = None,
        interpolation: str = "linear_par",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the implied forward rates between pairs of maturities of a country's
        zero-coupon (spot) yield curve on a day: the interest rate, implied by that day's curve,
        for a loan that starts at a future date.

        It follows from no-arbitrage pricing rather than being a forecast of future rates, and
        contains a term premium.

        The formula is as follows:

        - Forward Rate = ((1 + r2)^t2 / (1 + r1)^t1)^(1 / (t2 - t1)) - 1

        where:

        - r1 = Zero-coupon rate at the near maturity, read from the curve with the interpolation
        - t1 = Near maturity, in years
        - r2 = Zero-coupon rate at the far maturity, read from the curve with the interpolation
        - t2 = Far maturity, in years

        The curve is the zero-coupon curve of `get_zero_coupon_curve`, bootstrapped from the
        country's government bond par yields and annually compounded, so the forward rate is
        annually compounded as well. By default (linear par interpolation) it holds the zero
        rate of every coupon date the bootstrap solves for, not only those of the quoted
        maturities, and is linear between them; another interpolation bootstraps the curve
        with that interpolation, see `get_zero_coupon_curve`. For a curve of your own, use
        `yieldcurve_model.get_forward_rate`.

        Also known as: implied forward rate, forward-forward rate.

        Args:
            near_maturity (float | str | list | None, optional): The nearer maturity (or
                maturities), in years or as labels such as '6M'. Defaults to None, 1 to 5 years.
            far_maturity (float | str | list | None, optional): The further maturity (or
                maturities), in years or as labels such as '10Y'. Defaults to None, 5 to 10 years.
            country (str, optional): The country, see `get_zero_coupon_curve`. Defaults to
                "United States".
            date (str | None, optional): The day of the curve (YYYY-MM-DD), or the last day
                before it with a curve. Defaults to None, the end date.
            interpolation (str, optional): How the zero-coupon curve is read between its maturities, see
                `get_zero_coupon_curve`. Options are: "linear_par" (linear annually compounded zero rates on every
                coupon date), "linear_zero" (linear zero rates), "log_linear" (linear log discount factors),
                "natural_cubic" (a natural cubic spline), "monotone_cubic" (a monotone cubic spline), "monotone_convex"
                (Hagan and West's monotone convex method). Defaults to "linear_par".
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The forward rate (decimal, annually compounded) for each combination
            of near (rows) and far maturity (columns). Combinations where the far maturity is
            not greater than the near maturity are NaN.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        fixedincome.get_forward_rate(
            near_maturity=[1, 2, 5], far_maturity=[5, 10, 30], date="2026-11-05"
        )
        ```

        Which returns:

        |   Near Maturity |        5 |     10 |     30 |
        |----------------:|---------:|-------:|-------:|
        |               1 |   0.0473 | 0.0501 | 0.0548 |
        |               2 |   0.0474 | 0.0505 | 0.0551 |
        |               5 | nan      | 0.0524 | 0.056  |
        """
        maturities, zero_rates = self._curve_nodes(
            self._latest_yield_curve(country, date), country, interpolation
        )
        near = parse_maturities(near_maturity, range(1, 6))
        far = parse_maturities(far_maturity, range(5, 11))
        if not any(far_years > near_years for near_years in near for far_years in far):
            raise InvalidArgumentError(
                f"A forward rate runs from the near to a later far maturity, while far_maturity "
                f"({', '.join(f'{years:g}' for years in far)}) is nowhere above near_maturity "
                f"({', '.join(f'{years:g}' for years in near)}). Please pass a far_maturity above "
                "the near_maturity."
            )

        def zero_rate(maturity: float) -> float:
            discount = interpolation_model.get_discount_factors(
                maturities, zero_rates, maturity, interpolation
            )[0]
            return float(discount ** (-1 / maturity) - 1) if maturity > 0 else 0.0

        forward_rates = pd.DataFrame(
            {
                far_years: {
                    near_years: (
                        yieldcurve_model.get_forward_rate(
                            near_rate=zero_rate(near_years),
                            far_rate=zero_rate(far_years),
                            near_maturity=near_years,
                            far_maturity=far_years,
                        )
                        if far_years > near_years
                        else np.nan
                    )
                    for near_years in near
                }
                for far_years in far
            }
        )
        forward_rates.index.name = "Near Maturity"
        forward_rates.columns.name = "Far Maturity"

        return apply_rounding(
            forward_rates, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_par_yield(
        self,
        years_to_maturity: float | str | list | None = None,
        country: str = "United States",
        date: str | None = None,
        interpolation: str = "linear_par",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the par yields of any maturity from a country's zero-coupon (spot) yield
        curve on a day: the coupon rate a newly issued bond of each maturity needs so that,
        discounted with the spot curve, its price equals par.

        The government bond yield curve quotes par yields at its own maturities; this gives the
        par yield of any maturity in between (such as 4 or 15 years) from the zero-coupon rates
        of every coupon date the bootstrap of `get_zero_coupon_curve` solves for, so a maturity
        on a coupon date reprices the par yield the bootstrap assumed there and a maturity
        between coupon dates is interpolated on the zero-coupon rates.

        The formula is as follows:

        - DF(t) = 1 / (1 + Spot Rate(t / Frequency) / Frequency)^t, the discount factor of a cash flow after t
          coupon periods
        - Par Yield = Frequency * (1 - DF(t_n)) / (SUM(DF(t_i)) - a), over the remaining coupons i = 1, ..., n

        where Spot Rate(τ) is the zero-coupon rate of `get_zero_coupon_curve` at τ years, restated from annual
        compounding to the country's coupon frequency (twice a year, once for Sweden and Norway) and interpolated
        linearly between the bills and the coupon dates of the bootstrap (or read with another interpolation, see
        below), a is the elapsed fraction of the current coupon period (0 on a coupon date) and t_i = i - a is
        the time of the i-th coupon in periods. With an interpolation other than linear par the zero-coupon
        curve is bootstrapped with that interpolation (see `get_zero_coupon_curve`), so every quoted par yield
        reprices exactly and the maturities in between follow the interpolation. The par yield is a nominal annual coupon rate paid at the coupon
        frequency, like the government bond yield curve. For a curve of your own, use
        `yieldcurve_model.get_par_yield`.

        Also known as: par rate, par coupon rate.

        Args:
            years_to_maturity (float | str | list | None, optional): The maturities, in years
                or as labels such as '15Y'. Defaults to None, 1 to 30 years.
            country (str, optional): The country, see `get_zero_coupon_curve`. Defaults to
                "United States".
            date (str | None, optional): The day of the curve (YYYY-MM-DD), or the last day
                before it with a curve. Defaults to None, the end date.
            interpolation (str, optional): How the zero-coupon curve is read between its maturities, see
                `get_zero_coupon_curve`. Options are: "linear_par" (linear annually compounded zero rates on every
                coupon date), "linear_zero" (linear zero rates), "log_linear" (linear log discount factors),
                "natural_cubic" (a natural cubic spline), "monotone_cubic" (a monotone cubic spline), "monotone_convex"
                (Hagan and West's monotone convex method). Defaults to "linear_par".
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The par yield (decimal) by maturity in years.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        fixedincome.get_par_yield(years_to_maturity=[1, 4, 8, 15, 25], date="2026-11-05")
        ```

        Which returns:

        |   Years to Maturity |   Par Yield |
        |--------------------:|------------:|
        |                   1 |      0.0428 |
        |                   4 |      0.0456 |
        |                   8 |      0.0475 |
        |                  15 |      0.0505 |
        |                  25 |      0.0526 |
        """
        frequency = COUPON_FREQUENCIES.get(country, 2)
        maturities, zero_rates = self._curve_nodes(
            self._latest_yield_curve(country, date), country, interpolation
        )

        par_yields = pd.Series(
            {
                years: yieldcurve_model.get_par_yield(
                    spot_rates=interpolation_model.get_coupon_spot_rates(
                        maturities, zero_rates, years, frequency, interpolation
                    ),
                    years_to_maturity=years,
                    frequency=frequency,
                )
                for years in parse_maturities(years_to_maturity, range(1, 31))
            },
            name="Par Yield",
            dtype=float,
        )
        par_yields.index.name = "Years to Maturity"

        return apply_rounding(
            par_yields, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_instantaneous_forward_curve(
        self,
        maturities: float | str | list | None = None,
        country: str = "United States",
        date: str | None = None,
        interpolation: str = "monotone_convex",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the instantaneous forward curve of a country's zero-coupon curve on a day:
        at every maturity t, the rate the curve implies for borrowing over an infinitesimal
        period starting at t. Where a zero rate averages the forward rates up to its maturity,
        the instantaneous forward shows each part of the curve on its own, which makes it the
        sharpest view of the curve's shape and the curve the Heath-Jarrow-Morton and
        Hull-White models start from.

        The formula is as follows:

        - f(t) = -d ln P(t) / dt = r(t) + t × dr(t) / dt
        - r(t) = (1 / t) × ∫ f(s) ds, from 0 to t

        where P(t) is the discount factor and r(t) the continuously compounded zero rate of
        maturity t, read from the zero-coupon curve of `get_zero_coupon_curve` (bootstrapped
        from the par yields) with the interpolation. As the forward rate is the slope of
        r(t) t, it shows every kink of the interpolation: it jumps at every maturity of
        the linear zero and linear par interpolations and is flat between those of the
        log-linear one. The
        default, Hagan and West's (2006) monotone convex method, gives a continuous forward
        curve that keeps the discrete forwards between the maturities and does not
        oscillate, which is why it is the default here (the other curve methods default to
        linear par, the curve of `get_zero_coupon_curve`). At a maturity of the curve
        itself the forward is that of the interval that starts there; beyond the last
        maturity the zero rate, and so the forward, is held at the last zero rate.

        The instantaneous forward rate is continuously compounded, as it is defined.

        For more information, see the following papers:

        - Hagan, P.S. and West, G. (2006). "Interpolation Methods for Curve Construction."
          Applied Mathematical Finance, 13(2), 89-129.
        - Hagan, P.S. and West, G. (2008). "Methods for Constructing a Yield Curve." Wilmott
          Magazine, May, 70-81.

        Referenced against:
        - QuantLib, ZeroCurve (linear_zero): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
        - QuantLib, DiscountCurve (log_linear): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_discount_curve.html
        - QuantLib, NaturalCubicZeroCurve (natural_cubic): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
        - QuantLib, MonotonicCubicZeroCurve (monotone_cubic): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
        - QuantLib, PiecewiseYieldCurve with ConvexMonotone (monotone_convex): https://www.quantlib.org/reference/class_quant_lib_1_1_convex_monotone.html

        Also known as: instantaneous forward rate, forward curve, short forward curve.

        Args:
            maturities (float | str | list | None, optional): The maturities, in years or as
                labels such as '6M' (a list or a comma-separated string). Defaults to None,
                3 and 6 months and 1 to 30 years.
            country (str, optional): The country, see `get_zero_coupon_curve`. Defaults to
                "United States".
            date (str | None, optional): The day of the curve (YYYY-MM-DD), or the last day
                before it with a curve. Defaults to None, the end date.
            interpolation (str, optional): How the zero-coupon curve is read between its maturities, see
                `get_zero_coupon_curve`. Options are: "linear_par" (linear annually compounded zero rates on every
                coupon date), "linear_zero" (linear zero rates), "log_linear" (linear log discount factors),
                "natural_cubic" (a natural cubic spline), "monotone_cubic" (a monotone cubic spline), "monotone_convex"
                (Hagan and West's monotone convex method). Defaults to "monotone_convex".
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Instantaneous Forward Rate (continuously compounded) and the Zero
            Rate (annually compounded, as `get_zero_coupon_curve`) by maturity in years.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        fixedincome.get_instantaneous_forward_curve(maturities=[0.5, 1, 2, 5, 10, 20, 30], date="2026-11-05")
        ```

        Which returns:

        |   Maturity |   Instantaneous Forward Rate |   Zero Rate |
        |-----------:|-----------------------------:|------------:|
        |        0.5 |                       0.0421 |      0.0423 |
        |        1   |                       0.0441 |      0.0433 |
        |        2   |                       0.0461 |      0.0451 |
        |        5   |                       0.048  |      0.0465 |
        |       10   |                       0.0536 |      0.0494 |
        |       20   |                       0.0549 |      0.0552 |
        |       30   |                       0.0485 |      0.0541 |
        """
        nodes = self._curve_nodes(
            self._latest_yield_curve(country, date), country, interpolation
        )
        years = np.array(
            parse_maturities(maturities, [0.25, 0.5, *range(1, 31)]), dtype=float
        )

        forwards = interpolation_model.get_instantaneous_forward_rates(
            *nodes, years, interpolation
        )
        discount = interpolation_model.get_discount_factors(
            *nodes, years, interpolation
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            annual = np.where(years > 0, discount ** (-1 / years) - 1, np.nan)

        curve = pd.DataFrame(
            {"Instantaneous Forward Rate": forwards, "Zero Rate": annual},
            index=pd.Index([maturity_key(value) for value in years], name="Maturity"),
        )

        return apply_rounding(
            curve, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_smith_wilson_curve(
        self,
        maturities: float | str | list | None = None,
        country: str = "United States",
        date: str | None = None,
        ultimate_forward_rate: float = 0.033,
        convergence_speed: float | None = None,
        last_liquid_point: float | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Extends a country's government bond yield curve to any maturity with the
        Smith-Wilson method (Smith and Wilson, 2001): the risk-free curve that prices every
        bill and par bond of the curve exactly and, beyond its last liquid point, converges
        to an ultimate forward rate (UFR), as EIOPA builds the Solvency II risk-free rate
        term structures.

        Insurers and pension funds discount liabilities of 50 to 100 years with such a
        curve, longer than any bond that trades.

        The formula is as follows:

        - P(t) = e^(-ω t) + Σ_j ζ_j W(t, u_j)
        - W(t, u) = e^(-ω (t + u)) [α min(t, u) - ½ e^(-α max(t, u)) (e^(α min(t, u)) - e^(-α min(t, u)))]
        - ζ = Cᵀ (C W Cᵀ)⁻¹ (m - C μ), with μ_j = e^(-ω u_j)
        - ω = ln(1 + UFR)

        where u_j are the cash flow times of the instruments, C their cash flows and m their
        prices. The instruments are the curve's quotes of the day up to the last liquid point:
        the bills (a year or less) as zero-coupon bonds whose price follows from their yield
        in the country's quote convention, and the longer maturities as par bonds paying their
        par yield in coupons twice a year (once for Sweden and Norway) at a price of par. The
        Euro Area and German curves are zero-coupon curves, so every maturity is a
        zero-coupon bond.

        The convergence speed α, EIOPA's alpha, sets how fast the forward rate converges to
        the UFR. By default it is set as EIOPA does: the smallest alpha of at least 0.05 for which the instantaneous forward
        rate at the convergence point lies within a basis point of ω, with the convergence
        point at the later of 60 years and the last liquid point plus 40 years (see
        `smith_wilson_model.get_smith_wilson_alpha`).

        The zero rates are annually compounded, as EIOPA publishes them and as
        `get_zero_coupon_curve` returns them, and the instantaneous forward rate is
        continuously compounded, so it converges to ln(1 + UFR).

        For more information, see the following sources:

        - Smith, A. and Wilson, T. (2001). "Fitting Yield Curves with Long Term Constraints."
          Research Notes, Bacon and Woodrow.
        - EIOPA (2022). "Technical Documentation of the Methodology to Derive EIOPA's Risk-Free
          Interest Rate Term Structures." EIOPA-BoS-22/409.

        EIOPA's October 2026 euro curve, its Smith-Wilson coefficients, alpha (0.068369) and
        rates are reproduced from the same par swaps.

        Referenced against:
        - rateslib, SmithWilsonCurve: https://rateslib.com/py/en/3.0.x/api/rateslib.curves.SmithWilsonCurve.html

        Also known as: Smith-Wilson extrapolation, UFR curve, Solvency II curve extrapolation,
        ultimate forward rate.

        Args:
            maturities (float | str | list | None, optional): The maturities, in years or as
                labels such as '50Y' (a list or a comma-separated string). Defaults to None,
                1, 2, 3, 5, 7, 10, 20, 30, 40, 50, 60, 80, 100 and 150 years.
            country (str, optional): The country, see `get_zero_coupon_curve`. Defaults to
                "United States".
            date (str | None, optional): The day of the curve (YYYY-MM-DD), or the last day
                before it with a curve. Defaults to None, the end date.
            ultimate_forward_rate (float, optional): The UFR, annually compounded (decimal);
                EIOPA publishes it every year per currency. Defaults to 0.033.
            convergence_speed (float | None, optional): The speed α of convergence to the UFR,
                EIOPA's alpha, above 0. Defaults to None, EIOPA's convergence criterion.
            last_liquid_point (float | None, optional): The longest maturity, in years, whose
                quote the curve keeps. Defaults to None, the longest maturity of the curve.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Zero Rate (annually compounded), the Discount Factor and the
            Instantaneous Forward Rate (continuously compounded) by maturity in years, and the
            Alpha of the curve.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        fixedincome.get_smith_wilson_curve(date="2026-11-05")
        ```

        Which returns:

        |   Maturity |   Zero Rate |   Discount Factor |   Instantaneous Forward Rate |   Alpha |
        |-----------:|------------:|------------------:|-----------------------------:|--------:|
        |          1 |      0.0433 |            0.9585 |                       0.0444 |  0.1202 |
        |          2 |      0.0451 |            0.9155 |                       0.0466 |  0.1202 |
        |          3 |      0.0459 |            0.8741 |                       0.0459 |  0.1202 |
        |          5 |      0.0465 |            0.7966 |                       0.0479 |  0.1202 |
        |          7 |      0.0478 |            0.7213 |                       0.0506 |  0.1202 |
        |         10 |      0.0494 |            0.6172 |                       0.0543 |  0.1202 |
        |         20 |      0.0553 |            0.3407 |                       0.0579 |  0.1202 |
        |         30 |      0.0541 |            0.2061 |                       0.0436 |  0.1202 |
        |         40 |      0.0505 |            0.1393 |                       0.036  |  0.1202 |
        |         50 |      0.0474 |            0.0986 |                       0.0336 |  0.1202 |
        |         60 |      0.0451 |            0.0708 |                       0.0328 |  0.1202 |
        |         80 |      0.0421 |            0.0369 |                       0.0325 |  0.1202 |
        |        100 |      0.0403 |            0.0193 |                       0.0325 |  0.1202 |
        |        150 |      0.0379 |            0.0038 |                       0.0325 |  0.1202 |
        """
        par_curve = self._latest_yield_curve(country, date).dropna()
        quoted = np.array(
            [curve_model.maturity_in_years(label) for label in par_curve.index]
        )
        yields = par_curve.to_numpy(dtype=float)
        liquid = quoted <= (
            quoted.max()
            if last_liquid_point is None
            else float(last_liquid_point) + 1e-9
        )
        if not liquid.any():
            raise ValueError(
                f"The {country} yield curve quotes no maturity up to the last liquid point."
            )
        quoted, yields = quoted[liquid], yields[liquid]
        frequency = COUPON_FREQUENCIES.get(country, 2)

        if country in ZERO_COUPON_CURVES:
            instruments = [
                (np.array([years]), np.array([1.0]), np.exp(-rate * years))
                for years, rate in zip(quoted, yields, strict=True)
            ]
        else:
            bills = curve_model.bill_yields_to_annual(
                quoted, yields, frequency, BILL_CONVENTIONS.get(country)
            )
            instruments = []
            for years, par_yield, bill in zip(quoted, yields, bills, strict=True):
                if years <= 1 + 1e-9:
                    instruments.append(
                        (np.array([years]), np.array([1.0]), (1 + bill) ** -years)
                    )
                    continue
                times = bond_model.get_cash_flow_periods(years, frequency) / frequency
                flows = np.full(len(times), par_yield / frequency)
                flows[-1] += 1
                accrued = bond_model.get_accrued_fraction(years, frequency)
                instruments.append((times, flows, 1 + par_yield / frequency * accrued))

        cash_flow_times = np.unique(
            np.round(np.concatenate([times for times, _, _ in instruments]), 10)
        )
        cash_flows = np.zeros((len(instruments), len(cash_flow_times)))
        for row, (times, flows, _) in enumerate(instruments):
            cash_flows[row, np.searchsorted(cash_flow_times, np.round(times, 10))] = (
                flows
            )
        prices = np.array([price for _, _, price in instruments])

        if convergence_speed is None:
            convergence_speed = smith_wilson_model.get_smith_wilson_alpha(
                cash_flow_times,
                cash_flows,
                prices,
                ultimate_forward_rate,
                max(float(quoted.max()) + 40, 60.0),
            )
        years = np.array(
            parse_maturities(
                maturities, [1, 2, 3, 5, 7, 10, 20, 30, 40, 50, 60, 80, 100, 150]
            ),
            dtype=float,
        )
        curve = smith_wilson_model.get_smith_wilson_curve(
            cash_flow_times,
            cash_flows,
            prices,
            years,
            ultimate_forward_rate,
            convergence_speed,
        )
        curve.index = pd.Index(
            [maturity_key(value) for value in years], name="Maturity"
        )
        curve["Alpha"] = convergence_speed

        return apply_rounding(
            curve, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_yield_curve_spread(
        self,
        long_maturity: float | str | list | None = None,
        short_maturity: float | str | list | None = None,
        country: str = "United States",
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        middle_maturity: float | str | list | None = None,
    ):
        """
        Calculates the spreads between pairs of maturities of a country's government bond yield
        curve on a day: the slope of the curve, e.g. the widely followed 10-year minus 2-year
        and 10-year minus 3-month Treasury spreads.

        A positive spread indicates a "normal" upward-sloping curve, while a negative spread
        ("inversion") has historically been used as a leading indicator of an economic slowdown.

        The formula is as follows:

        - Yield Curve Spread = Long-Term Yield - Short-Term Yield

        Where both yields are the par yields of `get_government_bond_yield_curve`, as the
        Treasury and FRED (T10Y2Y, T10Y3M) calculate the spreads, interpolated linearly for a
        maturity the curve does not quote. They keep the source's quote convention: for the
        United States bond-equivalent yields, compounded twice a year beyond a year and simple
        interest for the bill maturities up to half a year, so a spread against a bill yield
        mixes the two conventions, as the published spreads do. For the history of these spreads, take the
        difference of two columns of `get_government_bond_yield_curve`; for yields of your own,
        use `yieldcurve_model.get_yield_curve_spread`.

        With `middle_maturity` it calculates butterflies instead, a measure of the curvature of
        the curve: twice how far the yield of a middle maturity (the body) lies above the average
        yield of a short and a long maturity (the wings), with the wings weighted equally as
        butterflies are quoted (2s5s10s). That average is the straight line through the wings
        only when the body lies midway between them, which 5 years does not in 2s5s10s:

        - Butterfly = 2 * Middle Yield - Short Yield - Long Yield

        A positive butterfly means the body yields more than the wings on average.

        Also known as: term spread, yield curve slope, 10Y-2Y spread, 2s10s, 2y-10y spread,
        inverted yield curve, curve inversion, fly, 2s5s10s.

        Args:
            long_maturity (float | str | list | None, optional): The longer maturity (or
                maturities), in years or as labels such as '10Y'. Defaults to None, 5, 10 and
                30 years, or 10 years for a butterfly.
            short_maturity (float | str | list | None, optional): The shorter maturity (or
                maturities), in years or as labels such as '3M'. Defaults to None, 3 months
                and 2 years, or 2 years for a butterfly.
            country (str, optional): The country, see `get_government_bond_yield_curve`.
                Defaults to "United States".
            date (str | None, optional): The day of the curve (YYYY-MM-DD), or the last day
                before it with a curve. Defaults to None, the end date.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.
            middle_maturity (float | str | list | None, optional): The middle maturity (or
                maturities) of a butterfly, in years or as labels such as '5Y'. Defaults to
                None, which calculates spreads.

        Returns:
            pd.DataFrame: The spread (decimal) for each combination of long (rows) and short
            maturity (columns), or with `middle_maturity` the butterfly (decimal) for each
            middle maturity (rows) and pair of wings (columns, short/long), NaN where the middle
            maturity does not lie between the wings.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        fixedincome.get_yield_curve_spread(date="2026-11-05")
        ```

        Which returns:

        | Long Maturity   |     3M |     2Y |
        |:----------------|-------:|-------:|
        | 5Y              | 0.0042 | 0.0013 |
        | 10Y             | 0.0067 | 0.0038 |
        | 30Y             | 0.0108 | 0.0079 |
        """
        par_curve = self._latest_yield_curve(country, date)
        maturities = [maturity_key(label) for label in par_curve.index]

        def par_yield(maturity: float) -> float:
            return float(
                np.interp(maturity, maturities, par_curve.to_numpy(dtype=float))
            )

        if middle_maturity is not None:
            middles = parse_maturities(middle_maturity, [])
            wings = [
                (short_years, long_years)
                for short_years in parse_maturities(short_maturity, [2])
                for long_years in parse_maturities(long_maturity, [10])
            ]
            label = fixedincome_helpers.maturity_label
            butterflies = pd.DataFrame(
                {
                    f"{label(short_years)}/{label(long_years)}": {
                        label(middle_years): (
                            yieldcurve_model.get_yield_curve_butterfly(
                                short_yield=par_yield(short_years),
                                middle_yield=par_yield(middle_years),
                                long_yield=par_yield(long_years),
                            )
                            if short_years < middle_years < long_years
                            else np.nan
                        )
                        for middle_years in middles
                    }
                    for short_years, long_years in wings
                }
            )
            butterflies.index.name = "Middle Maturity"
            butterflies.columns.name = "Wings"

            return apply_rounding(
                butterflies, rounding if rounding is not None else self._rounding
            )

        long = parse_maturities(long_maturity, [5, 10, 30])
        short = parse_maturities(short_maturity, [0.25, 2])
        spreads = pd.DataFrame(
            {
                fixedincome_helpers.maturity_label(short_years): {
                    fixedincome_helpers.maturity_label(
                        long_years
                    ): yieldcurve_model.get_yield_curve_spread(
                        long_yield=par_yield(long_years),
                        short_yield=par_yield(short_years),
                    )
                    for long_years in long
                }
                for short_years in short
            }
        )
        spreads.index.name = "Long Maturity"
        spreads.columns.name = "Short Maturity"

        return apply_rounding(
            spreads, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_breakeven_inflation_rate(
        self,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the breakeven inflation rates of the US Treasury curve on a day: the nominal
        par yield minus the real (TIPS) par yield of the 5, 7, 10, 20 and 30-year maturities. It
        is the rate of inflation that would make an investor indifferent between holding a
        nominal Treasury and a TIPS of the same maturity, and is widely used as a market-implied
        measure of expected inflation.

        The formula is as follows:

        - Breakeven Inflation Rate = Nominal Yield - Real Yield

        Where both yields are the Treasury's par yield curve rates (nominal and real), quoted as
        bond-equivalent yields compounded twice a year, and the breakeven rate is their simple
        difference, as FRED publishes it as T5YIE and T10YIE, not (1 + Nominal) / (1 + Real) - 1.

        The breakeven rate is often read as the market's expected inflation, but it also
        contains an inflation risk premium, which raises it, and a liquidity premium on
        inflation-linked bonds, which lowers it, so it rises and falls in market stress for
        reasons other than inflation expectations. The comparison is with CPI-U, the index TIPS
        pay on. The daily history, with the 5-year, 5-year forward rate, is
        `get_inflation_expectations` in the Economics module; for yields of your own, use
        `yieldcurve_model.get_breakeven_inflation_rate`.

        Also known as: TIPS breakeven spread, inflation breakeven.

        Args:
            date (str | None, optional): The day (YYYY-MM-DD), or the last business day
                before it. Defaults to None, the end date.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Nominal Yield, Real Yield and Breakeven Inflation Rate (decimals)
            by maturity, with the date of the yields.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        fixedincome.get_breakeven_inflation_rate(date="2026-11-05")
        ```

        Which returns:

        | Maturity   |   Nominal Yield |   Real Yield |   Breakeven Inflation Rate | Date       |
        |:-----------|----------------:|-------------:|---------------------------:|:-----------|
        | 5 Year     |          0.0459 |       0.0233 |                     0.0226 | 2026-11-05 |
        | 7 Year     |          0.047  |       0.0246 |                     0.0224 | 2026-11-05 |
        | 10 Year    |          0.0484 |       0.026  |                     0.0224 | 2026-11-05 |
        | 20 Year    |          0.0527 |       0.029  |                     0.0237 | 2026-11-05 |
        | 30 Year    |          0.0525 |       0.0305 |                     0.022  | 2026-11-05 |
        """
        end_date = date or self._end_date
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=21)
        ).strftime("%Y-%m-%d")
        # The Treasury serves whole months, so the days after the date are cut off here.
        nominal = treasury_model.get_yield_curve("nominal", start_date, end_date).loc[
            : pd.Period(end_date, "D")
        ]
        real = treasury_model.get_yield_curve("real", start_date, end_date).loc[
            : pd.Period(end_date, "D")
        ]
        if nominal.empty or real.empty:
            raise ValueError(
                f"The Treasury has no nominal and real yield curve in the weeks before {end_date}."
            )
        day = nominal.index.intersection(real.index).max()

        rates = pd.DataFrame(
            {
                "Nominal Yield": {
                    maturity: nominal.loc[day].get(columns["nominal"], np.nan)
                    for maturity, columns in treasury_model.MATURITIES.items()
                },
                "Real Yield": {
                    maturity: real.loc[day].get(columns["real"], np.nan)
                    for maturity, columns in treasury_model.MATURITIES.items()
                },
            }
        )
        rates["Breakeven Inflation Rate"] = (
            yieldcurve_model.get_breakeven_inflation_rate(
                nominal_yield=rates["Nominal Yield"], real_yield=rates["Real Yield"]
            )
        )
        rates["Date"] = str(pd.Period(day, "D"))
        rates.index.name = "Maturity"

        return apply_rounding(
            rates, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_government_bond_yield(
        self,
        short_term: bool = False,
        period: str | None = None,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the government bond yield for a variety of countries over time from the OECD: what governments pay to
        borrow, by default the long-term (10-year) government bond yield or else the short-term (3-month) rate.

        The two maturities are described below.

        Long-term: long-term interest rates refer to government bonds maturing
        in ten years. Rates are mainly determined by the price charged by the lender, the risk
        from the borrower and the fall in the capital value. Long-term interest rates
        are generally averages of daily rates, measured as a percentage. These interest
        rates are implied by the prices at which the government bonds are traded on
        financial markets, not the interest rates at which the loans were issued.

        In all cases, they refer to bonds whose capital repayment is guaranteed by governments.
        Long-term interest rates are one of the determinants of business investment. Low long
        term interest rates encourage investment in new equipment and high interest rates
        discourage it. Investment is, in turn, a major source of economic growth.

        See definition: https://data.oecd.org/interest/long-term-interest-rates.htm

        Short-term: short-term interest rates are the rates at which short-term
        borrowings are effected between financial institutions or the rate at which short-term
        government paper is issued or traded in the market. Short-term interest rates are
        generally averages of daily rates, measured as a percentage. They are based on
        three-month money market rates where available; the OECD source specifically returns
        the 3-month interbank offered rate rather than a government bill yield, so for most
        countries it tracks the central bank's policy rate closely.

        See definition: https://data.oecd.org/interest/short-term-interest-rates.htm

        Also known as: treasury yield, bond yield by maturity.

        Args:
            short_term (bool, optional): Whether to return the short-term interest rate. Defaults to False.
                This means that the long-term interest rate will be returned.
            period (str | None, optional): The frequency of the data. Options are: "monthly", "quarterly",
                "yearly", None (the monthly data, or the quarterly data when the class was created
                with quarterly data). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the long-term (10-year) government bond yield, or the
                short-term (3-month) interest rate.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2023-05-01', end_date='2023-12-31')

        long_term_interest_rate = fixedincome.get_government_bond_yield(short_term=False, period='monthly')

        long_term_interest_rate.loc[:, ['Japan', 'United States', 'Germany']]
        ```

        Which returns:

        |         |   Japan |   United States |   Germany |
        |:--------|--------:|----------------:|----------:|
        | 2023-05 |  0.0043 |          0.0357 |    0.0234 |
        | 2023-06 |  0.004  |          0.0375 |    0.0238 |
        | 2023-07 |  0.0059 |          0.039  |    0.0246 |
        | 2023-08 |  0.0064 |          0.0417 |    0.0255 |
        | 2023-09 |  0.0076 |          0.0438 |    0.0266 |
        | 2023-10 |  0.0095 |          0.048  |    0.0282 |
        | 2023-11 |  0.0066 |          0.045  |    0.026  |
        | 2023-12 |  0.0062 |          0.0402 |    0.021  |
        """
        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "monthly"
        )

        if short_term:
            government_bond_yield = oecd_model.get_short_term_interest_rate(
                period=period,
            )
        else:
            government_bond_yield = oecd_model.get_long_term_interest_rate(
                period=period,
            )

        return finalize_dataset(
            dataset=government_bond_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_bond_yield_curve(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the government bond yield curve of a variety of countries: the yield of
        every maturity a country's government borrows at, from treasury bills to bonds of 30
        years and more.

        The yield curve is the basis for discounting cash flows and pricing bonds, and its
        shape is one of the most watched signals in markets: an inverted curve, with short
        rates above long rates, has preceded most recessions.

        Every curve comes from the official source:

        - United States: the U.S. Department of the Treasury's par yield curve, 1 month to
          30 years, from 1990.
        - Euro Area: the yield curve the European Central Bank estimates from the bonds of
          all euro area central governments, 3 months to 30 years, from 2004, as zero-coupon
          (spot) rates of a Svensson model, continuously compounded.
        - Germany: the term structure the Deutsche Bundesbank estimates from listed federal
          securities, 1 to 30 years, as zero-coupon (spot) rates of a Svensson model,
          continuously compounded.
        - United Kingdom: the Bank of England's nominal par yields of gilts, 5, 10 and 20
          years.
        - Japan: the Ministry of Finance's Japanese government bond yields, 1 to 40 years,
          from 1974.
        - Canada: the Bank of Canada's treasury bill yields (1 month to 1 year, weekly) and
          benchmark bond yields (2 to 30 years).
        - Sweden: the Riksbank's treasury bill and government bond yields, 1 month to 10
          years.
        - Norway: Norges Bank's generic government bond yields, 3 to 10 years.
        - Australia: the Reserve Bank of Australia's government bond yields interpolated to
          2, 3, 5 and 10 years, from 2013.

        With monthly data, every other European Union member, such as France, Italy,
        Spain, the Netherlands and Poland, is included with its 10-year yield, the long-term
        interest rate for convergence purposes the ECB publishes monthly. For other
        countries, see `get_government_bond_yield` for the monthly 3-month and 10-year
        rates of around forty countries from the OECD. Only the requested countries are
        retrieved, and only the days that are not cached yet.

        The yields are returned as decimal fractions per annum (0.0419 for 4.19%), with one
        column per country and maturity, sorted from the shortest to the longest maturity.
        Each curve is returned as its source publishes it: the Euro Area and Germany curves
        are continuously compounded zero-coupon rates, while the other countries publish
        par or benchmark bond yields, which compound with the coupon frequency of their
        bonds. The two are not interchangeable: a continuously compounded zero rate r
        equals an annually compounded zero rate of exp(r) - 1, and a par yield is the
        coupon rate at which a bond prices at par, which differs from the zero rate of the
        same maturity whenever the curve is not flat.
        Weekly and monthly periods take the yields on the last trading day of each period
        (weeks end on Friday).

        Also known as: yield curve, term structure of interest rates, sovereign curve,
        treasury curve, bund curve, gilt curve, JGB curve.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "United States", "Euro Area", "Germany", "United Kingdom", "Japan", "Canada",
                "Sweden", "Norway" and "Australia", and with monthly data any European Union
                member.
                Defaults to None, which retrieves every country.
            period (str, optional): The frequency of the data. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame with the yields as decimals, indexed by date with a
            column per country and maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-09-28', end_date='2026-10-01')

        yield_curve = fixedincome.get_government_bond_yield_curve(
            countries=['United States', 'Germany']
        )

        yield_curve['Germany'][['1Y', '2Y', '5Y', '10Y', '20Y', '30Y']]
        ```

        Which returns:

        |            |     1Y |     2Y |     5Y |    10Y |    20Y |    30Y |
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2026-09-28 | 0.031  | 0.0332 | 0.0344 | 0.0369 | 0.0393 | 0.0401 |
        | 2026-09-29 | 0.0307 | 0.0329 | 0.0341 | 0.0368 | 0.0394 | 0.0403 |
        | 2026-09-30 | 0.0304 | 0.0324 | 0.0336 | 0.0364 | 0.039  | 0.0399 |
        | 2026-10-01 | 0.0302 | 0.0321 | 0.0335 | 0.0366 | 0.0394 | 0.0403 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "government bond yield curve"
        )
        sources = self._get_country_yield_curves(
            buffered_start_date(self._start_date, period)
        )

        if countries is not None and not isinstance(countries, str | list | tuple):
            raise TypeError(
                "The countries must be a country name or a list of country names, such as "
                f"'Germany' or ['Germany', 'Japan'], not a {type(countries).__name__} ({countries!r})."
            )

        requested = (
            list(sources)
            if countries is None
            else [countries] if isinstance(countries, str) else list(countries)
        )

        # The 10-year yield of every other European Union member is published monthly, so a
        # monthly request includes those as a curve of one maturity.
        convergence_yields = (
            economics_ecb_model.get_long_term_convergence_yields(
                buffered_start_date(self._start_date, "monthly"), self._end_date
            )
            if period == "monthly"
            else pd.DataFrame()
        )
        if countries is None and not convergence_yields.empty:
            requested += [
                country
                for country in convergence_yields.columns
                if country not in sources
            ]

        if unavailable := [
            country
            for country in requested
            if country not in sources and country not in convergence_yields.columns
        ]:
            logger.warning(
                "No official daily yield curve is available for %s. The government bond yield "
                "curve covers %s daily, and the 10-year yield of every other European Union "
                "member with period='monthly'; get_government_bond_yield returns the monthly "
                "3-month and 10-year rates of around forty countries from the OECD.",
                ", ".join(unavailable),
                ", ".join(sources),
            )

        curves = {}
        for country in requested:
            if country not in sources:
                if country in convergence_yields.columns:
                    curves[country] = (
                        convergence_yields[[country]]
                        .rename(columns={country: "10Y"})
                        .dropna()
                    )
                continue

            curve = sources[country]()

            if curve.empty:
                continue

            curve = curve[sorted(curve.columns, key=_maturity_in_months)]
            curves[country] = resample_to_period(curve.dropna(how="all"), period)

        if not curves:
            return pd.DataFrame()

        yield_curve = pd.concat(curves, axis=1).sort_index()
        yield_curve.index.name = None
        yield_curve.columns.names = ["Country", "Maturity"]

        return finalize_dataset(
            dataset=yield_curve,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    def _get_keyless_treasury_rates(self) -> pd.DataFrame:
        """
        Retrieves the Treasury par yield curve from the U.S. Department of the Treasury, in
        the maturities and column names FinancialModelingPrep uses.

        Returns:
            pd.DataFrame: The rates as decimals, indexed by date ("Date").
        """
        curve = treasury_model.get_yield_curve(
            "nominal", self._start_date, self._end_date
        )

        if curve.empty:
            return curve

        # The Treasury's names for the maturities FinancialModelingPrep publishes.
        maturities = {
            "1 Mo": "1 Month",
            "2 Mo": "2 Month",
            "3 Mo": "3 Month",
            "6 Mo": "6 Month",
            "1 Yr": "1 Year",
            "2 Yr": "2 Year",
            "3 Yr": "3 Year",
            "5 Yr": "5 Year",
            "7 Yr": "7 Year",
            "10 Yr": "10 Year",
            "20 Yr": "20 Year",
            "30 Yr": "30 Year",
        }
        treasury_rates = curve.rename(columns=maturities)[
            [name for column, name in maturities.items() if column in curve.columns]
        ]
        treasury_rates.index.name = "Date"

        return treasury_rates

    @handle_errors
    def get_treasury_rates(
        self,
        maturities: float | str | list | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the daily U.S. Treasury par yield curve rates as officially published by the
        U.S. Department of the Treasury: the yields at which the US government borrows, for
        every maturity from 1 Month through 30 Year in a single dataset.

        This is the official, risk-free curve widely used as the discount curve for bond
        valuation and as the benchmark for credit spreads.

        With a FinancialModelingPrep API key the rates come from FinancialModelingPrep, and
        without one, or when the key's plan does not include them, from the U.S. Department
        of the Treasury directly, which gives the same figures.

        These are par yields: the coupons at which a new Treasury of each maturity would price at
        par, interpolated by the Treasury from the most recently auctioned securities. Discounting
        individual cash flows needs zero-coupon (spot) rates bootstrapped from them; on a steep
        curve the two differ by several basis points at long maturities. The 10-year minus 2-year
        and 10-year minus 3-month spreads from this curve are the most watched recession indicators.

        Also known as: the Treasury yield curve, the risk-free curve.

        Args:
            maturities (float | str | list | None, optional): The maturities to return, in years
                or as labels such as "3M" and "10Y" (a list or a comma-separated string, e.g.
                "2Y,10Y,30Y"), from 1M, 2M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y and 30Y.
                Defaults to None, which returns every maturity.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Notes:
            The underlying endpoint caps each request at 90 calendar days of data, so this method
            paginates in 90-day windows to cover the full start_date to end_date range the class
            was initialized with. A long range therefore issues many requests -- be mindful of
            this on a Free plan's daily request limit and consider a narrower start_date where
            possible.

            The U.S. Department of the Treasury publishes these rates in percentage points (a
            ten-year yield of 3.95%), but they are converted to decimals here (0.0395) so that
            they match every other rate method in this module -- `get_euribor_rates`,
            `get_european_central_bank_rates`, `get_federal_reserve_rates`,
            `get_government_bond_yield` and the ICE BofA yield methods -- as well as the risk-free
            rate returned by `Toolkit.get_treasury_data`. They can therefore be passed directly
            into `get_present_value`, `get_z_spread`, `get_par_yield` and `get_key_rate_duration`,
            each of which is documented as taking a rate in decimal form.

            The rates are decimal fractions, so they feed straight into the bond-pricing methods
            above. Multiply by 100 to recover the published Treasury figures.

        Returns:
            pd.DataFrame: A DataFrame containing the Treasury par yield curve rates, as decimals,
            with one column per maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2024-01-01', end_date='2024-01-15')

        fixedincome.get_treasury_rates(maturities='3M,2Y,10Y,30Y')
        ```

        Which returns:

        | Date       |   3 Month |   2 Year |   10 Year |   30 Year |
        |:-----------|----------:|---------:|----------:|----------:|
        | 2024-01-02 |    0.0546 |   0.0433 |    0.0395 |    0.0408 |
        | 2024-01-03 |    0.0548 |   0.0433 |    0.0391 |    0.0405 |
        | 2024-01-04 |    0.0548 |   0.0438 |    0.0399 |    0.0413 |
        | 2024-01-05 |    0.0547 |   0.044  |    0.0405 |    0.0421 |
        | 2024-01-08 |    0.0549 |   0.0436 |    0.0401 |    0.0417 |
        | 2024-01-09 |    0.0547 |   0.0436 |    0.0402 |    0.0418 |
        | 2024-01-10 |    0.0546 |   0.0437 |    0.0404 |    0.042  |
        | 2024-01-11 |    0.0546 |   0.0426 |    0.0398 |    0.0418 |
        | 2024-01-12 |    0.0545 |   0.0414 |    0.0396 |    0.042  |
        """
        treasury_rates = pd.DataFrame()

        if self._api_key:
            try:
                treasury_rates = fmp_model.get_treasury_rates(
                    api_key=self._api_key,
                    start_date=self._start_date,
                    end_date=self._end_date,
                )
            except (ValueError, requests.exceptions.RequestException) as error:
                logger.warning(
                    "Could not retrieve the Treasury rates from FinancialModelingPrep (%s), "
                    "retrieving them from the U.S. Department of the Treasury instead.",
                    error,
                )
                treasury_rates = pd.DataFrame()

        # The Treasury publishes these in percentage points while every other rate method in this module returns decimals, so convert here rather than in the model -- that keeps the cached payload a faithful mirror of the endpoint. The error path, such as a plan that does not include the endpoint, returns a frame with no numeric columns, hence the guard.  # noqa: E501
        numeric_columns = treasury_rates.select_dtypes(include="number").columns

        if not numeric_columns.empty:
            treasury_rates[numeric_columns] = treasury_rates[numeric_columns] / 100
        else:
            # Without a key, or when FinancialModelingPrep does not serve them, the same
            # rates come from the U.S. Department of the Treasury, already as decimals.
            treasury_rates = self._get_keyless_treasury_rates()

        if maturities is not None and not treasury_rates.empty:
            # The columns are named "1 Month" to "30 Year"; in years they compare with the
            # maturities asked for, whether written as years or as labels such as "10Y".
            labels = {
                column: f"{column.split()[0]}{column.split()[1][0]}"
                for column in treasury_rates.columns
            }
            columns = {maturity_key(label): column for column, label in labels.items()}
            requested = parse_maturities(maturities, [])
            if unknown := [
                maturity for maturity in requested if maturity not in columns
            ]:
                raise ValueError(
                    f"The Treasury par yield curve has no maturity of {', '.join(map(str, unknown))} "
                    f"years. Choose from {', '.join(labels.values())}."
                )
            treasury_rates = treasury_rates[
                [columns[maturity] for maturity in requested]
            ]

        return finalize_dataset(
            dataset=treasury_rates,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    def _latest_yield_curve(self, country: str, date: str | None = None) -> pd.Series:
        """
        The most recent yield curve of a country up to the end date, retrieving only the last
        weeks rather than the full history.

        Args:
            country (str): The country, see get_government_bond_yield_curve.
            date (str | None): The day to take the curve of (YYYY-MM-DD), or the last day
                before it with a curve. Defaults to None, the end date.

        Returns:
            pd.Series: The yields as decimals by maturity label, sorted by maturity.
        """
        end_date = date or self._end_date
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=21)
        ).strftime("%Y-%m-%d")
        sources = self._get_country_yield_curves(start_date)
        check_option("country", country, sources)
        curve = sources[country]()
        curve = curve.loc[: pd.Period(end_date, "D")].dropna(how="all")
        if curve.empty:
            raise ValueError(
                f"The {country} yield curve has no observations in the weeks before {end_date}."
            )
        latest = curve.iloc[-1].dropna()
        return latest[sorted(latest.index, key=_maturity_in_months)]

    def _yield_curves(
        self, countries: list[str] | str, period: str
    ) -> dict[str, pd.DataFrame]:
        """
        The government bond yield curve of every requested country, by date (rows) and
        maturity label (columns), between the start and end date.

        Args:
            countries (list[str] | str): The countries, see get_government_bond_yield_curve.
            period (str): "daily", "weekly" or "monthly".

        Returns:
            dict[str, pd.DataFrame]: The curve per country that has one.

        Raises:
            InvalidArgumentError: If the period is not daily, weekly or monthly.
        """
        # Checked here rather than reported by get_government_bond_yield_curve, whose empty
        # answer would read as a country without a curve.
        check_period(str(period).lower(), ("daily", "weekly", "monthly"))
        requested = [countries] if isinstance(countries, str) else list(countries)
        # The yields are published to a basis point or finer, so the curve is not rounded
        # before the models use it.
        curves = self.get_government_bond_yield_curve(
            requested, period=period, rounding=10
        )
        available = (
            set(curves.columns.get_level_values(0))
            if isinstance(curves, pd.DataFrame) and not curves.empty
            else set()
        )
        result = {
            country: curves[country].dropna(how="all")
            for country in requested
            if country in available
        }
        if not result:
            raise ValueError(
                f"No government bond yield curve is available for {', '.join(requested)} "
                f"between {self._start_date} and {self._end_date}."
            )
        return result

    @staticmethod
    def _to_zero_coupon(curve: pd.DataFrame, country: str) -> pd.DataFrame:
        """
        Converts a country's par yield curve into zero-coupon rates by bootstrapping every
        date, see curve_model.bootstrap_zero_rates. The curves of the Euro Area and Germany
        are zero-coupon already, published continuously compounded by the ECB and the
        Bundesbank, and are only converted to annual compounding. A date is left out when
        the curve has no maturity of a year or less to start the bootstrap from.

        Args:
            curve (pd.DataFrame): The par yields by date (rows) and maturity label (columns).
            country (str): The country, which sets the coupon frequency.

        Returns:
            pd.DataFrame: The zero-coupon rates (annually compounded), in the same layout.
        """
        if country in ZERO_COUPON_CURVES:
            return np.expm1(curve)

        maturities = np.array(
            [curve_model.maturity_in_years(label) for label in curve.columns]
        )
        values = curve.to_numpy(dtype=float)
        zero_rates = np.full_like(values, np.nan)

        for row, yields in enumerate(values):
            available = np.isfinite(yields)
            if not available.any() or maturities[available].min() > 1:
                continue
            zero_rates[row, available] = curve_model.bootstrap_zero_rates(
                maturities[available],
                yields[available],
                COUPON_FREQUENCIES.get(country, 2),
                BILL_CONVENTIONS.get(country),
            )

        return pd.DataFrame(
            zero_rates, index=curve.index, columns=curve.columns
        ).dropna(how="all")

    @classmethod
    def _interpolated_zero_curve(
        cls,
        curve: pd.DataFrame,
        country: str,
        interpolation: str,
        maturities: float | str | list | None,
    ) -> pd.DataFrame:
        """
        The zero-coupon rates of every date of a country's par curve, bootstrapped with an
        interpolation and read at the maturities, see get_zero_coupon_curve. Each date uses
        its own yields only.

        Args:
            curve (pd.DataFrame): The par yields by date (rows) and maturity label (columns).
            country (str): The country, which sets the coupon frequency.
            interpolation (str): One of interpolation_model.INTERPOLATIONS.
            maturities (float | str | list | None): The maturities, or None for those the
                curve quotes (left NaN on a date without a yield for it).

        Returns:
            pd.DataFrame: The zero-coupon rates (annually compounded) by date and maturity.
        """
        if maturities is None:
            years = np.array(
                [curve_model.maturity_in_years(label) for label in curve.columns]
            )
            labels = list(curve.columns)
        else:
            years = np.array(parse_maturities(maturities, []), dtype=float)
            labels = [fixedincome_helpers.maturity_label(value) for value in years]
        if np.any(years <= 0):
            raise ValueError("The maturities must be longer than zero.")

        rows = {}
        for date, par_curve in curve.iterrows():
            try:
                nodes = cls._curve_nodes(par_curve, country, interpolation)
            except ValueError as error:
                if "maturity of a year or less" in str(error):
                    continue
                raise
            discount = interpolation_model.get_discount_factors(
                *nodes, years, interpolation
            )
            zero_rates = discount ** (-1 / years) - 1
            if maturities is None:
                zero_rates = np.where(par_curve.notna(), zero_rates, np.nan)
            rows[date] = zero_rates

        return pd.DataFrame.from_dict(rows, orient="index", columns=labels).dropna(
            how="all"
        )

    @handle_errors
    def get_zero_coupon_curve(
        self,
        countries: list[str] | str = "United States",
        period: str = "daily",
        interpolation: str = "linear_par",
        maturities: float | str | list | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the zero-coupon (spot) yield curve of a country by bootstrapping its
        government bond par yields: the single rate that discounts a payment at each
        maturity.

        A par yield is the coupon at which a bond trades at par, so it mixes the discount
        rates of every coupon date. Zero-coupon rates are what bonds and other cash flows
        are discounted with, and what forward rates and the curve models (Nelson-Siegel,
        Vasicek) are built on.

        Maturities up to one year are treasury bills, which pay no coupon, so their yield
        only needs converting to annual compounding: US bill yields are bond-equivalent
        yields (simple interest up to half a year), Canadian bill yields simple interest on a
        365-day year and Swedish bill yields simple interest on a 360-day year. Beyond that the
        par yields are interpolated linearly to every coupon date and each discount factor
        follows from the earlier ones:

            P(t_n) = (1 - c/f Σ_{i<n} P(t_i)) / (1 + c/f)

        with c the par yield of maturity t_n and f the coupons per year (two, and one for
        Sweden and Norway). The curves of the Euro Area and Germany are estimated as
        zero-coupon curves by the ECB and the Bundesbank, which publish them continuously
        compounded; they are converted to annual compounding like the others. A
        curve needs a maturity of a year or less to start from, so the United Kingdom,
        Norway and Australia (and Canada before its bill yields start) are not covered.

        How the curve is read between the quoted maturities is a choice (Hagan and West,
        2006). By default (linear par interpolation) the par yields are interpolated
        linearly to every coupon date, as above, and the zero rates linearly between those
        dates. The other interpolations bootstrap the zero curve itself: the coupon dates of
        a bond lie between quoted maturities, so their discount factors are read from the
        curve with the interpolation, and the discount factor of each maturity is solved so
        that its par bond prices at par on that curve. For the interpolations whose shape
        depends on the maturities further out (the cubic splines and monotone convex) the
        bootstrap is repeated until the curve no longer changes (to 1e-13). They work on
        continuously compounded zero rates r(t), with P(t) = e^(-r(t) t):

        - Linear zero: r(t) linear between the maturities.
        - Log-linear: ln P(t) linear between the maturities, so the instantaneous forward
          rate is flat between them (piecewise flat forward, Hagan and West's raw
          interpolation).
        - Natural cubic: a natural cubic spline on r(t).
        - Monotone cubic: a cubic on r(t) with the slopes of the natural spline held by
          Hyman's (1983) filter, so it does not overshoot where the rates rise or fall.
        - Monotone convex: Hagan and West's (2006) monotone convex method on the forward
          rates, which keeps the discrete forwards between the maturities, gives a continuous
          forward curve and does not oscillate.

        The zero rate before the first maturity is that of the first maturity (monotone
        convex extrapolates the forward instead) and beyond the last maturity it is held at
        the last one. The curves of the Euro Area and Germany are zero-coupon curves already,
        so the interpolation only matters between their maturities. Whatever the
        interpolation, the rates returned are annually compounded.

        For more information, see the following papers:

        - Hagan, P.S. and West, G. (2006). "Interpolation Methods for Curve Construction."
          Applied Mathematical Finance, 13(2), 89-129.
        - Hyman, J.M. (1983). "Accurate Monotonicity Preserving Cubic Interpolation." SIAM
          Journal on Scientific and Statistical Computing, 4(4), 645-654.

        QuantLib's convex monotone forward curve is taken without forced positivity,
        quadraticity 0 and monotonicity 1. The monotone cubic sets the slope to zero at a local
        extremum of the zero rates, where QuantLib applies Hyman's extended filter.

        Also known as: spot curve, zero curve, bootstrapped curve, discount curve, curve
        interpolation, monotone convex, flat forward.

        Referenced against:
        - QuantLib, PiecewiseYieldCurve: https://www.quantlib.org/reference/class_quant_lib_1_1_piecewise_yield_curve.html
        - QuantLib, FixedRateBondHelper: https://www.quantlib.org/reference/class_quant_lib_1_1_fixed_rate_bond_helper.html
        - QuantLib, DepositRateHelper: https://www.quantlib.org/reference/class_quant_lib_1_1_deposit_rate_helper.html
        - QuantLib, PiecewiseYieldCurve (linear_par): https://www.quantlib.org/reference/class_quant_lib_1_1_piecewise_yield_curve.html

        Args:
            countries (list[str] | str, optional): The countries, see
                `get_government_bond_yield_curve`. Defaults to "United States".
            period (str, optional): The frequency of the curves. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            interpolation (str, optional): How the curve is read between the quoted maturities. Options are:
                "linear_par" (linear annually compounded zero rates on every coupon date), "linear_zero" (linear zero
                rates), "log_linear" (linear log discount factors), "natural_cubic" (a natural cubic spline),
                "monotone_cubic" (a monotone cubic spline), "monotone_convex" (Hagan and West's monotone convex method).
                Defaults to "linear_par".
            maturities (float | str | list | None, optional): The maturities to return the
                zero rates of, in years or as labels such as '4Y' (a list or a
                comma-separated string). Defaults to None, the maturities the curve quotes.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The zero-coupon rates as decimals (annually compounded), indexed by
            date with a column per country and maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-10-19', end_date='2026-10-23')

        zero_curve = fixedincome.get_zero_coupon_curve()

        zero_curve['United States'][['1Y', '2Y', '5Y', '10Y', '20Y', '30Y']]
        ```

        Which returns:

        |            |     1Y |     2Y |     5Y |    10Y |    20Y |    30Y |
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2026-10-19 | 0.0455 | 0.0484 | 0.0501 | 0.0526 | 0.0581 | 0.0564 |
        | 2026-10-20 | 0.0456 | 0.0486 | 0.0501 | 0.0522 | 0.0573 | 0.0556 |
        | 2026-10-21 | 0.0457 | 0.0488 | 0.0502 | 0.0523 | 0.0572 | 0.0556 |
        | 2026-10-22 | 0.0446 | 0.0471 | 0.0485 | 0.0506 | 0.0557 | 0.0545 |
        | 2026-10-23 | 0.0449 | 0.0475 | 0.0489 | 0.051  | 0.0563 | 0.0548 |
        """
        zero_curves = {}
        for country, curve in self._yield_curves(countries, period).items():
            if interpolation == "linear_par" and maturities is None:
                zero_curve = self._to_zero_coupon(curve, country)
            else:
                zero_curve = self._interpolated_zero_curve(
                    curve, country, interpolation, maturities
                )
            if zero_curve.empty:
                logger.warning(
                    "The %s yield curve has no maturity of a year or less to bootstrap "
                    "zero-coupon rates from.",
                    country,
                )
                continue
            zero_curves[country] = zero_curve

        if not zero_curves:
            return pd.DataFrame()

        zero_curve = pd.concat(zero_curves, axis=1).sort_index()
        zero_curve.columns.names = ["Country", "Maturity"]

        return apply_rounding(
            zero_curve, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_nelson_siegel_svensson(
        self,
        countries: list[str] | str = "United States",
        period: str = "daily",
        method: str = "svensson",
        maturities: list[float | str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Fits the Nelson-Siegel-Svensson model to the zero-coupon yield curve of a country on
        every date: the level, slope and curvature factors that describe each curve with a
        few parameters.

        The model smooths out the noise of individual maturities, gives a yield for any
        maturity in between and turns the history of curves into a history of a handful of
        factors. It is the model central banks such as the ECB, the Bundesbank and the
        Federal Reserve (Gürkaynak, Sack and Wright) publish their curves with.

        The Nelson-Siegel model has a level β0 (the yield at an infinite maturity), a slope
        β1 (the short end minus the long end) and a curvature β2 (a hump that peaks at about
        1.8 τ1 years):

            y(m) = β0 + β1 (1 - e^(-m/τ1)) / (m/τ1) + β2 [(1 - e^(-m/τ1)) / (m/τ1) - e^(-m/τ1)]

        Svensson adds a second curvature β3 with its own decay τ2, which fits the long end
        better. With the dozen maturities a government curve has, the six Svensson
        parameters are weakly identified, as the level and the second curvature can offset
        each other; the Nelson-Siegel model has three factors that read as level, slope and
        curvature (Diebold and Li, 2006).

        Given the decays the betas follow exactly from least squares, so the decays are
        searched on a grid and refined for every date. The second decay is kept at least 1.5
        times the first and the first at least a third of the shortest maturity, and the
        level at or above the lower of zero and the longest yield (as Gürkaynak, Sack and
        Wright keep it at or above zero), so that the curve extrapolates sensibly beyond the
        maturities it is fitted to. The curve is the zero-coupon curve of
        `get_zero_coupon_curve`, so the countries are those it covers that have at least as
        many maturities as the model has parameters. The model is fitted to the
        continuously compounded rates, as the ECB, the Bundesbank and the Federal Reserve
        fit it, so the parameters compare with theirs; the yields at `maturities` are
        annually compounded, like those of `get_zero_coupon_curve`.

        For more information, see the following papers:

        - Nelson, C.R. and Siegel, A.F. (1987). "Parsimonious Modeling of Yield Curves."
          The Journal of Business, 60(4), 473-489.
        - Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest Rates:
          Sweden 1992-1994." NBER Working Paper 4871.
        - Diebold, F.X. and Li, C. (2006). "Forecasting the Term Structure of Government
          Bond Yields." Journal of Econometrics, 130(2), 337-364.
        - Gürkaynak, R.S., Sack, B. and Wright, J.H. (2007). "The U.S. Treasury Yield Curve:
          1961 to the Present." Journal of Monetary Economics, 54(8), 2291-2304.

        Also known as: NSS, Svensson curve, Nelson-Siegel curve, yield curve fitting,
        level slope curvature factors.

        Args:
            countries (list[str] | str, optional): The countries, see
                `get_zero_coupon_curve`. Defaults to "United States".
            period (str, optional): The frequency of the curves. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            method (str, optional): The model to fit. Options are: "svensson" (the six
                parameters of Svensson, 1994) or "nelson_siegel" (the three factors of Nelson and
                Siegel, 1987). Defaults to "svensson".
            maturities (list[float | str] | str | None, optional): The maturities, in years or
                as labels such as "3M" and "10Y" (a list or a comma-separated string), to
                return the fitted zero-coupon yields at instead of the parameters. Defaults
                to None, which returns the parameters.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per date, the parameters (Beta 0 to Beta 3 as continuously
            compounded rates, Tau 1 and Tau 2 in years) and the Root Mean Squared Error of the
            fit, or the fitted (annually compounded) yields, with a column per country and
            parameter (or maturity).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-10-19', end_date='2026-10-23')

        fixedincome.get_nelson_siegel_svensson(countries='Germany')
        ```

        Which returns:

        |            |   Beta 0 |   Beta 1 |   Beta 2 |   Beta 3 |   Tau 1 |   Tau 2 |   Root Mean Squared Error |
        |:-----------|---------:|---------:|---------:|---------:|--------:|--------:|--------------------------:|
        | 2026-10-19 |   0.0409 |  -0.0173 |   0.0418 |  -0.051  |  1.1799 |  1.7805 |                         0 |
        | 2026-10-20 |   0.0408 |  -0.0172 |   0.0411 |  -0.0505 |  1.1997 |  1.7996 |                         0 |
        | 2026-10-21 |   0.0406 |  -0.0168 |   0.0308 |  -0.041  |  1.1497 |  1.8913 |                         0 |
        | 2026-10-22 |   0.0397 |  -0.0161 |   0.0313 |  -0.0385 |  1.1471 |  1.8929 |                         0 |
        | 2026-10-23 |   0.0395 |  -0.0154 |   0.0365 |  -0.0462 |  1.2162 |  1.8245 |                         0 |
        """
        check_option("method", method, ("svensson", "nelson_siegel"))
        svensson = method == "svensson"
        # A number of years (also written as a string, "0.5") is labeled "0.5Y".
        labels = [
            (
                f"{curve_model.maturity_in_years(label):g}Y"
                if not isinstance(label, str) or label.replace(".", "", 1).isdigit()
                else label
            )
            for label in helpers.split_values(maturities)
        ] or None

        fits = {}
        for country, curve in self._yield_curves(countries, period).items():
            zero_curve = self._to_zero_coupon(curve, country)
            if zero_curve.empty:
                logger.warning(
                    "The %s yield curve has no maturity of a year or less to bootstrap "
                    "zero-coupon rates from.",
                    country,
                )
                continue
            zero_curve.columns = [
                curve_model.maturity_in_years(label) for label in zero_curve.columns
            ]
            parameters = curve_model.fit_nelson_siegel_curves(
                np.log1p(zero_curve), svensson
            )
            parameters = parameters.dropna(how="all")
            if parameters.empty:
                logger.warning(
                    "The %s yield curve has too few maturities to fit the %s model.",
                    country,
                    "Svensson" if svensson else "Nelson-Siegel",
                )
                continue

            if labels is None:
                fits[country] = parameters
                continue

            years = np.array([curve_model.maturity_in_years(label) for label in labels])
            fits[country] = pd.DataFrame(
                [
                    np.expm1(
                        curve_model.nelson_siegel_svensson_yield(
                            years,
                            row["Beta 0"],
                            row["Beta 1"],
                            row["Beta 2"],
                            row.get("Beta 3", 0.0),
                            row["Tau 1"],
                            row.get("Tau 2", 1.0),
                        )
                    )
                    for _, row in parameters.iterrows()
                ],
                index=parameters.index,
                columns=labels,
            )

        if not fits:
            return pd.DataFrame()

        result = pd.concat(fits, axis=1).sort_index()
        result.columns.names = [
            "Country",
            "Parameter" if labels is None else "Maturity",
        ]

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_carry_and_roll_down(
        self,
        countries: list[str] | str = "United States",
        period: str = "daily",
        horizon: float | str = 0.25,
        maturities: list[float | str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the carry and roll-down of zero-coupon government bonds on every date: the
        return in excess of financing that a bond of each maturity earns over the horizon if
        the yield curve stays as it is. The bond earns its own yield over the financing rate
        (the carry) and, as it ages, is priced at the yield of a shorter maturity, which on
        an upward sloping curve is lower, so its price rises (the roll-down). Together they
        are the return an unchanged curve gives, the bond carry of Koijen, Moskowitz,
        Pedersen and Vrugt (2018): the yield spread over the short rate plus the roll-down.
        It shows where on the curve a position is paid most for holding it, and where an
        inverted curve makes holding a bond cost money.

        The formula is as follows:

        - Carry = (1 + z(T))^h - (1 + r)^h
        - Roll-Down = (1 + z(T))^T / (1 + z(T - h))^(T - h) - (1 + z(T))^h
        - Total = Carry + Roll-Down = P(T - h) / P(T) - (1 + r)^h

        where:

        - z(m) = The zero-coupon rate of maturity m, annually compounded
        - P(m) = (1 + z(m))^(-m), the price of a zero-coupon bond paying 1 at m
        - T = The maturity of the bond, in years
        - h = The horizon, in years
        - r = The financing rate over the horizon: the zero-coupon rate of the horizon, z(h)

        The curve is the zero-coupon curve of `get_zero_coupon_curve`, bootstrapped from the
        government bond par yields of the day (or published as zero-coupon rates for the Euro
        Area and Germany) and annually compounded, with the zero rate of every coupon date the
        bootstrap solves for. z(T - h) and z(h) are interpolated linearly on it, as for
        `get_forward_rate`, and held flat below the shortest maturity, so where the curve
        starts beyond the horizon (Germany at one year) the financing rate is the shortest
        rate. Beyond the longest maturity L the curve is held flat as well, so a bond with
        T - h at or beyond L has a roll-down of 0 and only the carry
        (1 + z(L))^h - (1 + z(h))^h. Each bond is repriced exactly on the unchanged curve: the
        carry is what it earns over the financing if its own yield stays z(T), the roll-down
        the price change from its yield moving to z(T - h), valued at the horizon. Every
        value is a return over the horizon, not annualized. Germany and Japan quote
        maturities from one year, so a horizon of a year avoids the flat extrapolation there.
        For a curve or a financing rate of your own, use
        `yieldcurve_model.get_carry_and_roll_down`.

        To first order, with D = (T - h) / (1 + z(T)) the modified duration (annual
        compounding) of the bond at the horizon, as Koijen et al. (2018) approximate it:

        - Carry ≈ (z(T) - r) × h
        - Roll-Down ≈ -D × (z(T - h) - z(T))

        For more information, see the following sources:

        - Koijen, R.S.J., Moskowitz, T.J., Pedersen, L.H. and Vrugt, E.B. (2018). "Carry."
          Journal of Financial Economics, 127(2), 197-225.
        - Ilmanen, A. (2011). "Expected Returns: An Investor's Guide to Harvesting Market
          Rewards." Wiley, chapter 9.

        Also known as: bond carry, rolldown, roll yield, horizon return on an unchanged curve.

        Args:
            countries (list[str] | str, optional): The countries, see
                `get_zero_coupon_curve`. Defaults to "United States".
            period (str, optional): The frequency of the curves. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            horizon (float | str, optional): The holding period, in years or as a label such
                as "3M" or "1Y". Defaults to 0.25 (three months).
            maturities (list[float | str] | str | None, optional): The maturities of the
                bonds, in years or as labels such as "2Y" (a list or a comma-separated
                string), at least one longer than the horizon; a maturity shorter than the horizon
                is NaN. Defaults to None, every maturity of the curve longer than the horizon.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Carry, Roll-Down and Total as decimals (returns over the
            horizon), indexed by date with a column per country, component and maturity. A
            date whose curve has no maturity of a year or less to bootstrap from (such as
            Canada before its bill yields of the day are published) is left out.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-10-19', end_date='2026-10-23')

        carry_and_roll_down = fixedincome.get_carry_and_roll_down(maturities='2Y,5Y,10Y,30Y')

        carry_and_roll_down['United States']['Total']
        ```

        Which returns:

        |            |     2Y |     5Y |    10Y |    30Y |
        |:-----------|-------:|-------:|-------:|-------:|
        | 2026-10-19 | 0.0024 | 0.0022 | 0.0033 | 0.002  |
        | 2026-10-20 | 0.0025 | 0.0022 | 0.0031 | 0.0019 |
        | 2026-10-21 | 0.0026 | 0.0021 | 0.0032 | 0.0019 |
        | 2026-10-22 | 0.0021 | 0.0017 | 0.0028 | 0.002  |
        | 2026-10-23 | 0.0022 | 0.0019 | 0.003  | 0.002  |
        """
        horizon_years = fixedincome_helpers.maturity_in_years(horizon)
        if horizon_years <= 0:
            raise InvalidArgumentError(
                f"The horizon must be longer than zero, not {horizon!r}."
            )
        if maturities is not None and not any(
            maturity > horizon_years for maturity in parse_maturities(maturities, [])
        ):
            raise InvalidArgumentError(
                f"A bond is held over the horizon of {horizon!r}, so at least one of the maturities "
                f"must be longer than it, not {maturities!r}. Please pass longer maturities or a "
                "shorter horizon."
            )

        results = {}
        for country, curve in self._yield_curves(countries, period).items():
            quoted = [curve_model.maturity_in_years(label) for label in curve.columns]
            years = parse_maturities(
                maturities,
                [maturity for maturity in quoted if maturity > horizon_years],
            )
            if not years:
                logger.warning(
                    "The %s yield curve has no maturity longer than the horizon of %s years, "
                    "so no bond is held over it.",
                    country,
                    horizon_years,
                )
                continue
            measures = {}
            for date, par_curve in curve.iterrows():
                try:
                    zero_curve = self._zero_curve(par_curve, country)
                except ValueError:
                    # A day without a maturity of a year or less has no zero-coupon curve.
                    continue
                measures[date] = yieldcurve_model.get_carry_and_roll_down(
                    zero_curve, years, horizon_years
                ).T.stack()
            if not measures:
                logger.warning(
                    "The %s yield curve has no maturity of a year or less to bootstrap "
                    "zero-coupon rates from.",
                    country,
                )
                continue
            result = pd.DataFrame.from_dict(measures, orient="index")
            result.columns = pd.MultiIndex.from_tuples(
                [
                    (component, fixedincome_helpers.maturity_label(maturity))
                    for component, maturity in result.columns
                ]
            )
            results[country] = result

        if not results:
            return pd.DataFrame()

        carry_and_roll_down = pd.concat(results, axis=1).sort_index()
        carry_and_roll_down.columns.names = ["Country", "Component", "Maturity"]

        return apply_rounding(
            carry_and_roll_down, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_yield_curve_principal_components(
        self,
        country: str = "United States",
        period: str = "daily",
        components: int = 3,
        changes: bool = True,
        scores: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the principal components of a country's government bond yield curve: the
        few independent movements that explain nearly all of how the curve changes. Litterman
        and Scheinkman (1991) found that three components explain over 95% of the changes of
        Treasury yields: a parallel shift (level, PC1, loading about equally on every
        maturity), a steepening (slope, PC2, with opposite signs at the short and long end)
        and a bending (curvature, PC3, moving the middle against both ends). Hedging a bond
        portfolio against the first two covers most of its interest rate risk.

        The components are the eigenvectors of the covariance matrix of the yield changes
        (or levels), sorted by the share of the variance they explain. Their sign is
        arbitrary, so PC1 is oriented as a rise in all yields (its loadings sum to a
        positive number), PC2 as a steepening (the longest maturity loads above the shortest)
        and the others so that their loadings sum to a positive number.
        Maturities that are only published for part of the period (such as the 4-month
        Treasury bill from 2022) are left out, so that the analysis covers the full period.

        For more information, see the following paper:

        - Litterman, R. and Scheinkman, J. (1991). "Common Factors Affecting Bond Returns."
          The Journal of Fixed Income, 1(1), 54-61.

        Also known as: PCA of the yield curve, level slope curvature, term structure factors.

        Args:
            country (str, optional): The country, see `get_government_bond_yield_curve`.
                Defaults to "United States".
            period (str, optional): The frequency of the curves. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            components (int, optional): The number of components, at least 1 and at most the number of
                maturities of the curve. Defaults to 3.
            changes (bool, optional): Whether to analyze the changes of the yields, the usual
                choice as yields themselves are not stationary, or their levels. Defaults to True.
            scores (bool, optional): Whether to return the value of every component per date
                (the scores) instead of the loadings. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The loadings per maturity and component, with the share of the
            variance each component explains as the last row, or the scores per date.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2016-01-01', end_date='2025-12-31')

        fixedincome.get_yield_curve_principal_components(period='monthly')
        ```

        Which returns:

        | Maturity           |    PC1 |     PC2 |     PC3 |
        |:-------------------|-------:|--------:|--------:|
        | 1M                 | 0.1332 | -0.6209 |  0.3084 |
        | 3M                 | 0.1713 | -0.4809 |  0.1636 |
        | 6M                 | 0.225  | -0.3809 | -0.0592 |
        | 1Y                 | 0.2789 | -0.2545 | -0.2647 |
        | 2Y                 | 0.3538 | -0.0249 | -0.4383 |
        | 3Y                 | 0.3676 |  0.0701 | -0.3617 |
        | 5Y                 | 0.3733 |  0.1595 | -0.1347 |
        | 7Y                 | 0.3636 |  0.1907 |  0.0399 |
        | 10Y                | 0.3434 |  0.1972 |  0.2136 |
        | 20Y                | 0.3052 |  0.1888 |  0.4277 |
        | 30Y                | 0.2855 |  0.1773 |  0.4831 |
        | Explained Variance | 0.732  |  0.1938 |  0.0501 |
        """
        validate_window("components", components)
        curve = self._yield_curves(country, period)[country]

        # A maturity published for only part of the period would cut the analysis short,
        # as only the dates with every maturity can be used.
        coverage = curve.notna().sum()
        curve = curve.loc[:, coverage >= CURVE_COVERAGE * coverage.max()]
        # A curve of one maturity has no components whatever is asked, which the model reports.
        if 1 < curve.shape[1] < components:
            raise InvalidArgumentError(
                f"The {country} yield curve has {curve.shape[1]} maturities, so at most "
                f"{curve.shape[1]} components, not {components}. Please pass fewer components."
            )

        loadings, component_scores, explained = curve_model.get_principal_components(
            curve, components=components, changes=changes
        )

        if scores:
            result = component_scores
        else:
            result = pd.concat([loadings, explained.to_frame().T])
            result.index.name = "Maturity"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @staticmethod
    def _continuous_short_rate(rates: pd.Series, country: str, label: str) -> pd.Series:
        """
        The yields of one maturity as continuously compounded rates, the short rate the
        Vasicek, CIR and Hull-White models describe, converted the way `_to_zero_coupon`
        converts the bills.

        Args:
            rates (pd.Series): The yields over time.
            country (str): The country, which sets the quote convention.
            label (str): The maturity label, such as "3M".

        Returns:
            pd.Series: The continuously compounded rates.
        """
        rates = rates.dropna()
        if country in ZERO_COUPON_CURVES:
            return rates
        maturity = curve_model.maturity_in_years(label)
        annual = curve_model.bill_yields_to_annual(
            np.full(len(rates), maturity),
            rates.to_numpy(dtype=float),
            COUPON_FREQUENCIES.get(country, 2),
            BILL_CONVENTIONS.get(country),
        )
        return pd.Series(np.log1p(annual), index=rates.index, name=rates.name)

    @handle_errors
    def get_term_premium(
        self,
        period: str = "daily",
        maturity_years: list[int] | str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the Treasury term premia of Adrian, Crump and Moench (2013): the extra
        yield investors demand for holding a long Treasury instead of rolling over short
        ones, as the Federal Reserve Bank of New York estimates it daily from 1961 for
        maturities of 1 to 10 years.

        A Treasury yield is split into the average short rate investors expect over the
        bond's life and the term premium. The split tells whether a move in long yields
        comes from expected monetary policy or from compensation for risk, such as inflation
        and fiscal uncertainty.

        Adrian, T., Crump, R. K., & Moench, E. (2013). Pricing the Term Structure with Linear
        Regressions. Journal of Financial Economics, 110(1), 110-138.
        https://doi.org/10.1016/j.jfineco.2013.04.009

        Three components are returned per maturity: the "Term Premium", the "Expected Short
        Rate" (the risk-neutral yield, the yield if investors were indifferent to risk) and
        the "Fitted Yield" of the model, which is their sum. The model is fitted to the
        zero-coupon Treasury curve of Gürkaynak, Sack and Wright (2007), so every value is a
        continuously compounded zero-coupon rate, as a decimal fraction (0.0067 for 0.67%).
        Weekly and monthly periods take the last day of each period,
        matching the New York Fed's own month-end series.

        The term premium is a model estimate rather than an observed price: it rests on the
        model's five pricing factors (principal components of the yield curve), and other
        models, such as that of Kim and Wright (2005), give different levels with similar
        movements.

        See definition: https://www.newyorkfed.org/research/data_indicators/term-premia-tabs

        Also known as: ACM term premium, Treasury term premium, risk-neutral yield, expected
        short rate, bond risk premium.

        Args:
            period (str, optional): The frequency of the data. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            maturity_years (list[int] | str | None, optional): The maturities in years to keep,
                from 1 to 10, e.g. [2, 5, 10] (or "2,5,10"). Defaults to None, which keeps every
                maturity.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The rates, indexed by date with a column per component ("Term
            Premium", "Expected Short Rate" and "Fitted Yield") and maturity ("1Y" to "10Y").

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-05-01', end_date='2026-10-31')

        term_premium = fixedincome.get_term_premium(period='monthly', maturity_years=[2, 5, 10])

        term_premium['Term Premium']
        ```

        Which returns:

        |         |     2Y |     5Y |    10Y |
        |:--------|-------:|-------:|-------:|
        | 2026-05 | 0.0011 | 0.003  | 0.0066 |
        | 2026-06 | 0.0004 | 0.0018 | 0.0051 |
        | 2026-07 | 0.0018 | 0.0043 | 0.0083 |
        | 2026-08 | 0.0015 | 0.0038 | 0.0076 |
        | 2026-09 | 0.0042 | 0.0057 | 0.0089 |
        | 2026-10 | 0.0024 | 0.0037 | 0.0068 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "ACM term premium"
        )
        term_premium = resample_to_period(nyfed_model.get_term_premium(), period)

        if isinstance(maturity_years, str):
            maturity_years = [
                int(float(maturity))
                for maturity in maturity_years.split(",")
                if maturity.strip()
            ]
        if maturity_years is not None and not term_premium.empty:
            term_premium = term_premium.loc[
                :,
                term_premium.columns.get_level_values(1).isin(
                    [f"{maturity}Y" for maturity in maturity_years]
                ),
            ]

        return finalize_dataset(
            dataset=term_premium,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_eiopa_risk_free_rate(
        self,
        countries: list[str] | str | None = None,
        curve: str = "spot_no_va",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        maturity_years: list[int] | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the risk-free interest rate term structures the European Insurance and
        Occupational Pensions Authority (EIOPA) publishes monthly: the curves European
        insurers must discount their liabilities with under Solvency II.

        They cover the euro and the currencies of every country of the European Economic
        Area, plus Switzerland, the United Kingdom, Australia, Canada, China, Colombia, Hong
        Kong, Japan, Taiwan and the United States, for maturities of 1 to 150 years. Beyond
        the last liquid maturity each curve converges to the ultimate forward rate (UFR), as
        Solvency II prescribes.

        Four curves are published: the basic spot curve, the spot curve with the volatility
        adjustment, and the basic curve after the upward and downward interest rate shocks of
        the Solvency II standard formula. EIOPA ships the shocked worksheets as formulas without computed
        values, so they are computed here with those formulas: upwards by the relative shock
        per maturity and at least one percentage point, downwards by the relative shock with
        negative rates left unchanged. Every value is a decimal fraction (0.0358 for 3.58%),
        at the end of the month the release is for, and an annually compounded spot rate, as
        Solvency II defines the term structure: a rate r discounts a payment in t years with
        (1 + r)^(-t).

        EIOPA's page links the releases from January 2023; each is a
        separate file, so only the months between the start and end date are downloaded, and
        a release is cached for a year since it does not change.

        Up to the last liquid point the curve follows market rates (for the euro, swap rates
        minus a credit risk adjustment); beyond it the curve is extrapolated towards the
        ultimate forward rate, so long maturities reflect the regulatory assumption more than
        market prices. Insurers that are allowed to apply the volatility adjustment discount
        with that curve, and the shocked curves give the interest rate risk capital
        requirement of the standard formula.

        See definition: https://www.eiopa.europa.eu/tools-and-data/risk-free-interest-rate-term-structures_en

        Also known as: Solvency II discount curve, EIOPA RFR, risk-free rate term structure,
        UFR curve.

        Args:
            countries (list[str] | str | None, optional): The currencies or countries to include,
                as EIOPA names them, e.g. 'Euro Area', 'United Kingdom' or 'United States'.
                Defaults to None, which returns all of them.
            curve (str, optional): The curve. Options are: "spot_no_va" (the basic spot curve),
                "spot_with_va" (the spot curve with the volatility adjustment), "shock_up" (the
                basic curve after the upward shock), "shock_down" (the basic curve after the
                downward shock). Defaults to "spot_no_va".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            maturity_years (list[int] | None, optional): The maturities in years to keep, e.g.
                [1, 2, 5, 10, 20, 30] (or "1,2,5,10,20,30") for the key points of the curve. Defaults to None, which
                keeps every maturity from 1 to 150 years.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The rates, indexed by month with a column per currency or country and
            maturity ("1Y" to "150Y").

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-07-01', end_date='2026-09-30')

        curves = fixedincome.get_eiopa_risk_free_rate(countries=['Euro Area', 'United Kingdom'])

        curves.loc[:, (slice(None), ['1Y', '10Y', '30Y', '60Y', '150Y'])]
        ```

        Which returns:

        |         |   ('Euro Area', '1Y') |   ('United Kingdom', '1Y') |   ('Euro Area', '10Y') |   ('United Kingdom', '10Y') |   ('Euro Area', '30Y') |   ('United Kingdom', '30Y') |   ('Euro Area', '60Y') |   ('United Kingdom', '60Y') |   ('Euro Area', '150Y') |   ('United Kingdom', '150Y') |
        |:--------|----------------------:|---------------------------:|-----------------------:|----------------------------:|-----------------------:|----------------------------:|-----------------------:|----------------------------:|------------------------:|-----------------------------:|
        | 2026-07 |                0.0283 |                     0.0415 |                 0.0316 |                      0.0468 |                 0.0337 |                      0.0518 |                 0.0335 |                      0.0443 |                  0.0332 |                       0.0375 |
        | 2026-09 |                0.0327 |                     0.0445 |                 0.0358 |                      0.0501 |                 0.0351 |                      0.0533 |                 0.0342 |                      0.0447 |                  0.0335 |                       0.0376 |
        """
        check_option(
            "curve", curve, {**eiopa_model.CURVES, **eiopa_model.SHOCKED_CURVES}
        )
        if countries is not None and not isinstance(countries, str | list | tuple):
            raise TypeError(
                "The countries must be a name or a list of names, such as 'Euro Area' or "
                f"['Euro Area', 'Japan'], not a {type(countries).__name__} ({countries!r})."
            )

        term_structures = eiopa_model.get_risk_free_rate_term_structures(
            curve, self._start_date, self._end_date
        )

        if countries is not None and not term_structures.empty:
            requested = [countries] if isinstance(countries, str) else list(countries)
            available = list(dict.fromkeys(term_structures.columns.get_level_values(0)))

            if unknown := [name for name in requested if name not in available]:
                logger.warning(
                    "EIOPA publishes no risk-free rates for %s. It covers %s.",
                    ", ".join(unknown),
                    ", ".join(available),
                )

            term_structures = term_structures.loc[
                :, term_structures.columns.get_level_values(0).isin(requested)
            ]

        if isinstance(maturity_years, str):
            maturity_years = [
                int(float(maturity))
                for maturity in maturity_years.split(",")
                if maturity.strip()
            ]
        if maturity_years is not None:
            # 150 maturities per country and month are rarely all needed.
            term_structures = term_structures.loc[
                :,
                term_structures.columns.get_level_values(1).isin(
                    [f"{maturity}Y" for maturity in maturity_years]
                ),
            ]

        return finalize_dataset(
            dataset=term_structures,
            indicator_name="EIOPA Risk-Free Rate",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_eiopa_symmetric_adjustment(
        self,
        period: str = "daily",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the symmetric adjustment of the equity capital charge of Solvency II: how
        much the "equity dampener" EIOPA publishes, daily from 1991, raises or lowers the
        equity charges of the standard formula.

        The adjustment raises the charge when equity prices are above their three-year
        average and lowers it when they are below, between -10% and +10%, so insurers are
        not forced to sell equities into a falling market. With get_eiopa_risk_free_rate it
        completes the market inputs of the standard formula's market risk module.

        The formula is as follows:

        - Type 1 Equity Charge = 0.39 + Symmetric Adjustment, for equities listed in the EEA or OECD
        - Type 2 Equity Charge = 0.49 + Symmetric Adjustment, for other equities

        Every value is a decimal fraction (0.0771 for 7.71%). Weekly and monthly periods take the value on
        the last day of each period, which is the value insurers apply at that reporting date.

        See definition: https://www.eiopa.europa.eu/tools-and-data/symmetric-adjustment-equity-capital-charge_en

        Also known as: equity dampener, symmetric adjustment, Solvency II equity charge.

        Args:
            period (str, optional): The frequency of the data. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The symmetric adjustment and the type 1 and type 2 equity charges,
            indexed by date.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-04-01', end_date='2026-09-30')

        fixedincome.get_eiopa_symmetric_adjustment(period='monthly')
        ```

        Which returns:

        |         |   Symmetric Adjustment |   Type 1 Equity Charge |   Type 2 Equity Charge |
        |:--------|-----------------------:|-----------------------:|-----------------------:|
        | 2026-04 |                 0.0767 |                 0.4667 |                 0.5667 |
        | 2026-05 |                 0.0868 |                 0.4768 |                 0.5768 |
        | 2026-06 |                 0.0894 |                 0.4794 |                 0.5794 |
        | 2026-07 |                 0.0951 |                 0.4851 |                 0.5851 |
        | 2026-08 |                 0.0938 |                 0.4838 |                 0.5838 |
        | 2026-09 |                 0.0771 |                 0.4671 |                 0.5671 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "EIOPA symmetric adjustment"
        )
        symmetric_adjustment = resample_to_period(
            eiopa_model.get_symmetric_adjustment(), period
        )
        if not symmetric_adjustment.empty:
            symmetric_adjustment.index.name = None

        return finalize_dataset(
            dataset=symmetric_adjustment,
            indicator_name="EIOPA Symmetric Adjustment",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_bank_of_england_yield_curve(
        self,
        curve: str = "nominal",
        period: str = "daily",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the UK yield curves the Bank of England estimates daily from 1985: the
        spot rates of UK government bonds (gilts) and of overnight index swaps on SONIA
        (OIS, from 2009), for maturities from 0.5 years to 40 years (25 years for OIS).

        Unlike the par yields of 5, 10 and 20 years in `get_government_bond_yield_curve`,
        these are zero-coupon spot rates over the full range of maturities with history back
        to 1985. The Bank of England publishes them continuously compounded, and they are
        returned that way: a spot rate r discounts a payment in t years with exp(-r t).

        The Bank of England's real and implied inflation curves are part of
        `economics.get_real_yield_curve` and `economics.get_breakeven_inflation_expectations`
        for the United Kingdom.

        The archive is a set of large files, of which only those
        covering the requested years are read, and they are cached since past years do not
        change. Weekly and monthly periods take the curve on the last day of each period.

        See definition: https://www.bankofengland.co.uk/statistics/yield-curves

        Also known as: gilt curve, UK spot curve, SONIA OIS curve.

        Args:
            curve (str, optional): The curve. Options are: "nominal" (the gilt spot curve), "ois" (the
                SONIA overnight index swap spot curve). Defaults to "nominal".
            period (str, optional): The frequency of the data. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The spot rates as decimals, indexed by date with a column per
            maturity in years, in steps of half a year.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-04-01', end_date='2026-09-30')

        fixedincome.get_bank_of_england_yield_curve(period='monthly')[
            ['1Y', '3Y', '5Y', '10Y', '20Y', '30Y']
        ]
        ```

        Which returns:

        |         |     1Y |     3Y |     5Y |    10Y |    20Y |    30Y |
        |:--------|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2026-04 | 0.0426 | 0.0433 | 0.0447 | 0.0506 | 0.0573 | 0.0581 |
        | 2026-05 | 0.0403 | 0.0414 | 0.043  | 0.0486 | 0.0555 | 0.0563 |
        | 2026-06 | 0.0402 | 0.041  | 0.0427 | 0.0483 | 0.0552 | 0.0561 |
        | 2026-07 | 0.0417 | 0.0438 | 0.0456 | 0.051  | 0.058  | 0.0589 |
        | 2026-08 | 0.0418 | 0.0441 | 0.046  | 0.0514 | 0.0582 | 0.0591 |
        | 2026-09 | 0.0442 | 0.0478 | 0.0494 | 0.054  | 0.0598 | 0.06   |
        """
        if curve in ("real", "inflation"):
            raise ValueError(
                f"The {curve} curve of the Bank of England is part of economics."
                + (
                    "get_real_yield_curve"
                    if curve == "real"
                    else "get_breakeven_inflation_expectations"
                )
                + "(countries='United Kingdom')."
            )
        check_option("curve", curve, ("nominal", "ois"))
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "Bank of England yield curve"
        )

        spot_curve = resample_to_period(
            boe_model.get_spot_curve(
                curve, buffered_start_date(self._start_date, period), self._end_date
            ),
            period,
        )

        return finalize_dataset(
            dataset=spot_curve,
            indicator_name="Bank of England Yield Curve",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )
