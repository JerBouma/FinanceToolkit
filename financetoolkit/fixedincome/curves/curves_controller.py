"""Yield Curves Module"""

__docformat__ = "google"

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from financetoolkit import (
    helpers,
)
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
from financetoolkit.fixedincome.base.base_controller import (
    FixedIncomeBase,
    _maturities,
    _maturity_key,
)
from financetoolkit.fixedincome.curves import curve_model, yieldcurve_model
from financetoolkit.fixedincome.sources import eiopa_model, fmp_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
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
class CurvesFixedIncome(FixedIncomeBase):
    """
    Yield curves: government bond yields and curves, Treasury rates, forward rates, par yields, spreads, breakeven inflation, zero-coupon curves, Nelson-Siegel-Svensson fits, principal components, the term premium, the Bank of England curves and the EIOPA risk-free rates.
    """

    def _latest_zero_curve(self, country: str, date: str | None = None) -> pd.Series:
        """
        The zero-coupon curve of a country on a day, bootstrapped from its par yield curve
        as `get_zero_coupon_curve` does.

        Args:
            country (str): The country, see get_zero_coupon_curve.
            date (str | None): The day (YYYY-MM-DD), or the last day before it with a curve.
                Defaults to None, the end date.

        Returns:
            pd.Series: The annually compounded zero-coupon rates by maturity in years.
        """
        par_curve = self._latest_yield_curve(country, date)
        zero_curve = self._to_zero_coupon(par_curve.to_frame().T, country)
        if zero_curve.empty:
            raise ValueError(
                f"The {country} yield curve has no maturity of a year or less to bootstrap "
                "zero-coupon rates from."
            )
        zero_curve = zero_curve.iloc[0].dropna()

        return pd.Series(
            zero_curve.to_numpy(dtype=float),
            index=[_maturity_key(label) for label in zero_curve.index],
        )

    @handle_errors
    def get_forward_rate(
        self,
        near_maturity: float | str | list | None = None,
        far_maturity: float | str | list | None = None,
        country: str = "United States",
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the implied forward rates between pairs of maturities of a country's
        zero-coupon (spot) yield curve on a day. The forward rate is the interest rate, implied by
        that day's curve, for a loan that starts at a future date: it follows from no-arbitrage
        pricing rather than being a forecast of future rates, and contains a term premium.

        The formula is as follows:

        - Forward Rate = ((1 + r2)^t2 / (1 + r1)^t1)^(1 / (t2 - t1)) - 1

        where:

        - r1 = Zero-coupon rate at the near maturity, interpolated linearly from the curve
        - t1 = Near maturity, in years
        - r2 = Zero-coupon rate at the far maturity, interpolated linearly from the curve
        - t2 = Far maturity, in years

        The curve is the zero-coupon curve of `get_zero_coupon_curve`, bootstrapped from the
        country's government bond par yields and annually compounded, so the forward rate is
        annually compounded as well. For a curve of your own, use
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
        zero_curve = self._latest_zero_curve(country, date)
        near = _maturities(near_maturity, range(1, 6))
        far = _maturities(far_maturity, range(5, 11))

        def zero_rate(maturity: float) -> float:
            return float(np.interp(maturity, zero_curve.index, zero_curve.to_numpy()))

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
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the par yields of any maturity from a country's zero-coupon (spot) yield
        curve on a day. The par yield for a maturity is the coupon rate a newly issued bond of
        that maturity needs so that, discounted with the spot curve, its price equals par.

        The government bond yield curve quotes par yields at its own maturities; this gives the
        par yield of any maturity in between (such as 4 or 15 years) with the interpolation done
        on the zero-coupon rates, where it belongs, rather than on the par yields.

        The formula is as follows:

        - DF(t) = 1 / (1 + Spot Rate(t / Frequency) / Frequency)^t, the discount factor of a cash flow after t
          coupon periods
        - Par Yield = Frequency * (1 - DF(t_n)) / (SUM(DF(t_i)) - a), over the remaining coupons i = 1, ..., n

        where Spot Rate(τ) is the zero-coupon rate of `get_zero_coupon_curve` at τ years, restated from annual
        compounding to the country's coupon frequency (twice a year, once for Sweden and Norway) and interpolated
        linearly, a is the elapsed fraction of the current coupon period (0 on a coupon date) and t_i = i - a is
        the time of the i-th coupon in periods. The par yield is a nominal annual coupon rate paid at the coupon
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
        |                  15 |      0.0507 |
        |                  25 |      0.0527 |
        """
        frequency = COUPON_FREQUENCIES.get(country, 2)
        zero_curve = self._latest_zero_curve(country, date)
        spot_rates = frequency * ((1 + zero_curve) ** (1 / frequency) - 1)

        par_yields = pd.Series(
            {
                years: yieldcurve_model.get_par_yield(
                    spot_rates=spot_rates,
                    years_to_maturity=years,
                    frequency=frequency,
                )
                for years in _maturities(years_to_maturity, range(1, 31))
            },
            name="Par Yield",
            dtype=float,
        )
        par_yields.index.name = "Years to Maturity"

        return apply_rounding(
            par_yields, rounding if rounding is not None else self._rounding
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
    ):
        """
        Calculates the spreads between pairs of maturities of a country's government bond yield
        curve on a day, e.g. the widely followed 10-year minus 2-year and 10-year minus 3-month
        Treasury spreads. A positive spread indicates a "normal" upward-sloping curve, while a
        negative spread ("inversion") has historically been used as a leading indicator of an
        economic slowdown.

        The formula is as follows:

        - Yield Curve Spread = Long-Term Yield - Short-Term Yield

        Where both yields are the par yields of `get_government_bond_yield_curve`, as the
        Treasury and FRED (T10Y2Y, T10Y3M) calculate the spreads, interpolated linearly for a
        maturity the curve does not quote. For the history of these spreads, take the
        difference of two columns of `get_government_bond_yield_curve`; for yields of your own,
        use `yieldcurve_model.get_yield_curve_spread`.

        Also known as: term spread, yield curve slope, 10Y-2Y spread.

        Args:
            long_maturity (float | str | list | None, optional): The longer maturity (or
                maturities), in years or as labels such as '10Y'. Defaults to None, 5, 10 and
                30 years.
            short_maturity (float | str | list | None, optional): The shorter maturity (or
                maturities), in years or as labels such as '3M'. Defaults to None, 3 months
                and 2 years.
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

        Returns:
            pd.DataFrame: The spread (decimal) for each combination of long (rows) and short
            maturity (columns).

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
        maturities = [_maturity_key(label) for label in par_curve.index]

        def par_yield(maturity: float) -> float:
            return float(
                np.interp(maturity, maturities, par_curve.to_numpy(dtype=float))
            )

        long = _maturities(long_maturity, [5, 10, 30])
        short = _maturities(short_maturity, [0.25, 2])
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

        Where both yields are the Treasury's par yield curve rates (nominal and real), the simple
        difference FRED publishes as T5YIE and T10YIE, not (1 + Nominal) / (1 + Real) - 1.

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
        Get the government bond yield for a variety of countries over time from the OECD. By
        default this is the long-term (10-year) government bond yield; set short_term=True to
        get the short-term (3-month) rate instead. The two maturities are described below.

        Long-term (short_term=False): long-term interest rates refer to government bonds maturing
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

        Short-term (short_term=True): short-term interest rates are the rates at which short-term
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
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Defaults to None, which returns the monthly data (quarterly when the class was
                created with quarterly=True).
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
                short-term (3-month) interest rate when short_term=True.

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
        Retrieves the government bond yield curve of a variety of countries, every maturity
        a country's government borrows at, from treasury bills to bonds of 30 years and
        more. The yield curve is the basis for discounting cash flows and pricing bonds,
        and its shape is one of the most watched signals in markets: an inverted curve,
        with short rates above long rates, has preceded most recessions.

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

        With period="monthly", every other European Union member, such as France, Italy,
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
                "Sweden", "Norway" and "Australia", and with period="monthly" any European
                Union member.
                Defaults to None, which retrieves every country.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
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
        U.S. Department of the Treasury, covering every maturity from 1 Month through 30 Year in
        a single dataset. This is the official, risk-free curve widely used as the discount curve
        for bond valuation and as the benchmark for credit spreads.

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
            columns = {_maturity_key(label): column for column, label in labels.items()}
            requested = _maturities(maturities, [])
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
        if country not in sources:
            raise ValueError(
                f"No daily yield curve is available for {country}. Choose from {', '.join(sources)}."
            )
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
        """
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

    @handle_errors
    def get_zero_coupon_curve(
        self,
        countries: list[str] | str = "United States",
        period: str = "daily",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the zero-coupon (spot) yield curve of a country by bootstrapping its
        government bond par yields. A par yield is the coupon at which a bond trades at par,
        so it mixes the discount rates of every coupon date; the zero-coupon rate of a
        maturity is the single rate that discounts a payment at that date. Zero-coupon rates
        are what bonds and other cash flows are discounted with, and what forward rates and
        the curve models (Nelson-Siegel, Vasicek) are built on.

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

        Also known as: spot curve, zero curve, bootstrapped curve, discount curve.

        Args:
            countries (list[str] | str, optional): The countries, see
                `get_government_bond_yield_curve`. Defaults to "United States".
            period (str, optional): Whether to use the daily, weekly or monthly curves.
                Defaults to "daily".
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
            zero_curve = self._to_zero_coupon(curve, country)
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
        svensson: bool = True,
        maturities: list[float | str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Fits the Nelson-Siegel-Svensson model to the zero-coupon yield curve of a country on
        every date. The model describes a whole curve with a few parameters, which smooths
        out the noise of individual maturities, gives a yield for any maturity in between
        and turns the history of curves into a history of a handful of factors. It is the
        model central banks such as the ECB, the Bundesbank and the Federal Reserve
        (Gürkaynak, Sack and Wright) publish their curves with.

        The Nelson-Siegel model has a level β0 (the yield at an infinite maturity), a slope
        β1 (the short end minus the long end) and a curvature β2 (a hump that peaks at about
        1.8 τ1 years):

            y(m) = β0 + β1 (1 - e^(-m/τ1)) / (m/τ1) + β2 [(1 - e^(-m/τ1)) / (m/τ1) - e^(-m/τ1)]

        Svensson adds a second curvature β3 with its own decay τ2, which fits the long end
        better. With the dozen maturities a government curve has, the six Svensson
        parameters are weakly identified, as the level and the second curvature can offset
        each other; set svensson=False for the three factors to read as level, slope and
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
            period (str, optional): Whether to fit the daily, weekly or monthly curves.
                Defaults to "daily".
            svensson (bool, optional): Whether to fit the Svensson model (True) or the
                Nelson-Siegel model (False). Defaults to True.
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
            period (str, optional): Whether to use the daily, weekly or monthly curves.
                Defaults to "daily".
            components (int, optional): The number of components. Defaults to 3.
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
        curve = self._yield_curves(country, period)[country]

        # A maturity published for only part of the period would cut the analysis short,
        # as only the dates with every maturity can be used.
        coverage = curve.notna().sum()
        curve = curve.loc[:, coverage >= CURVE_COVERAGE * coverage.max()]

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
        Retrieves the Treasury term premia of Adrian, Crump and Moench (2013), as the
        Federal Reserve Bank of New York estimates them daily from 1961 for maturities of 1
        to 10 years. A Treasury yield is split into the average short rate investors expect
        over the bond's life and the term premium: the extra yield they demand for holding a
        long bond instead of rolling over short ones. The split tells whether a move in long
        yields comes from expected monetary policy or from compensation for risk, such as
        inflation and fiscal uncertainty.

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
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
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
        insurers must discount their liabilities with under Solvency II. They cover the euro
        and the currencies of every country of the European Economic Area, plus Switzerland,
        the United Kingdom, Australia, Canada, China, Colombia, Hong Kong, Japan, Taiwan and
        the United States, for maturities of 1 to 150 years. Beyond the last liquid maturity
        each curve converges to the ultimate forward rate (UFR), as Solvency II prescribes.

        Four curves are published: the basic spot curve (curve="spot_no_va"), the spot curve
        with the volatility adjustment (curve="spot_with_va"), and the basic curve after the
        interest rate shocks of the Solvency II standard formula (curve="shock_up" and
        curve="shock_down"). EIOPA ships the shocked worksheets as formulas without computed
        values, so they are computed here with those formulas: upwards by the relative shock
        per maturity and at least one percentage point, downwards by the relative shock with
        negative rates left unchanged. Every value is a decimal fraction (0.0358 for 3.58%),
        at the end of the month the release is for.

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
            curve (str, optional): "spot_no_va", "spot_with_va", "shock_up" or "shock_down".
                Defaults to "spot_no_va".
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
        if curve not in {**eiopa_model.CURVES, **eiopa_model.SHOCKED_CURVES}:
            raise ValueError(
                f"The curve must be one of {', '.join({**eiopa_model.CURVES, **eiopa_model.SHOCKED_CURVES})}, not {curve!r}."
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
        Retrieves the symmetric adjustment of the equity capital charge of Solvency II, the
        "equity dampener" EIOPA publishes, daily from 1991, and the equity charges of the
        standard formula it results in. The adjustment raises the charge when equity prices
        are above their three-year average and lowers it when they are below, between -10%
        and +10%, so insurers are not forced to sell equities into a falling market. With
        get_eiopa_risk_free_rate it completes the market inputs of the standard formula's
        market risk module.

        The formula is as follows:

        - Type 1 Equity Charge = 0.39 + Symmetric Adjustment, for equities listed in the EEA or OECD
        - Type 2 Equity Charge = 0.49 + Symmetric Adjustment, for other equities
        Every value is a decimal fraction (0.0771 for 7.71%). Weekly
        and monthly periods take the value on the last day of each period, which is the
        value insurers apply at that reporting date.

        See definition: https://www.eiopa.europa.eu/tools-and-data/symmetric-adjustment-equity-capital-charge_en

        Also known as: equity dampener, symmetric adjustment, Solvency II equity charge.

        Args:
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
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
        Retrieves the UK yield curves the Bank of England estimates daily from 1985, for
        maturities from 0.5 years to 40 years (25 years for OIS): the nominal spot curve of
        UK government bonds (gilts) and the spot curve of overnight index swaps on SONIA (OIS,
        from 2009). Unlike the par yields of 5, 10 and 20 years in
        `get_government_bond_yield_curve`, these are zero-coupon spot rates over the full
        range of maturities with history back to 1985.

        The Bank of England's real and implied inflation curves are part of
        `economics.get_real_yield_curve` and `economics.get_breakeven_inflation_expectations`
        (countries='United Kingdom').

        The archive is a set of large files, of which only those
        covering the requested years are read, and they are cached since past years do not
        change. Weekly and monthly periods take the curve on the last day of each period.

        See definition: https://www.bankofengland.co.uk/statistics/yield-curves

        Also known as: gilt curve, UK spot curve, SONIA OIS curve.

        Args:
            curve (str, optional): "nominal" or "ois". Defaults to "nominal".
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
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
        if curve not in ("nominal", "ois"):
            raise ValueError(f"The curve must be 'nominal' or 'ois', not {curve!r}.")
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
