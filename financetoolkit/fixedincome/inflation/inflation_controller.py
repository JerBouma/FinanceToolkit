"""Inflation Swaps Module"""

__docformat__ = "google"

from datetime import (
    date as date_type,
    datetime,
    timedelta,
)

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from financetoolkit.economics.sources import treasury_model
from financetoolkit.fixedincome.curves import curve_model
from financetoolkit.fixedincome.helpers import parse_maturities
from financetoolkit.fixedincome.inflation import inflation_swap_model
from financetoolkit.fixedincome.markets import inflation_bond_model
from financetoolkit.fixedincome.markets.markets_controller import MarketsFixedIncome
from financetoolkit.fixedincome.sources import bundesbank_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import check_option, handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = logger_model.get_logger()

# The coupons per year of the inflation-linked bonds whose real yields make up each country's
# real yield curve: US TIPS pay twice a year, German inflation-linked German federal securities once.
REAL_COUPON_FREQUENCIES = {"United States": 2, "Germany": 1}


@filter_shown_columns
@track_sources
class InflationFixedIncome(MarketsFixedIncome):
    """
    Inflation swaps: zero-coupon and year-on-year inflation swap rates implied by the nominal and real government bond curves, and the forward consumer price index with its monthly seasonality.
    """

    def _real_yield_curve(self, country: str, date: str | None) -> pd.Series:
        """
        The real par yields of a country's inflation-linked government bonds on the last day
        on or before the date.

        Args:
            country (str): "United States" (the Treasury's real par yield curve of TIPS) or
                "Germany" (the Bundesbank's real yields of inflation-linked German federal securities at constant
                maturities).
            date (str | None): The day (YYYY-MM-DD). Defaults to None, the end date.

        Returns:
            pd.Series: The real yields (decimals) by maturity in years, named after the day.
        """
        check_option("country", country, REAL_COUPON_FREQUENCIES)
        end_date = date or self._end_date
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=21)
        ).strftime("%Y-%m-%d")

        if country == "United States":
            real = treasury_model.get_yield_curve("real", start_date, end_date)
            real = real.rename(
                columns={
                    columns["real"]: float(maturity.split()[0])
                    for maturity, columns in treasury_model.MATURITIES.items()
                }
            )
        else:
            real = bundesbank_model.get_real_yield_curve(start_date, end_date)
            real = real.rename(columns=lambda label: float(str(label)[:-1]))

        real = real.loc[: pd.Period(end_date, "D")].dropna(how="all")
        if real.empty:
            raise ValueError(
                f"The real yield curve of {country} has no observations in the weeks before {end_date}."
            )
        latest = real.iloc[-1]
        latest = latest[
            [column for column in latest.index if isinstance(column, float)]
        ].dropna()
        if latest.empty:
            raise ValueError(
                f"The real yield curve of {country} on {pd.Period(real.index[-1], 'D')} has no "
                "maturity with a yield to bootstrap from."
            )
        latest.name = str(pd.Period(real.index[-1], "D"))

        return latest.sort_index()

    def _inflation_curves(
        self, country: str, date: str | None
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        The nominal and real zero-coupon curves of a country on the last day on or before the
        date that has a real yield curve.

        The real par yields are bootstrapped like the nominal ones, with the coupons of the
        country's inflation-linked bonds, and held at the shortest quoted real yield before
        its maturity (five years in the US, about three in Germany).

        Args:
            country (str): "United States" or "Germany".
            date (str | None): The day (YYYY-MM-DD). Defaults to None, the end date.

        Returns:
            tuple[pd.Series, pd.Series, pd.Series]: The annually compounded nominal and real
            zero rates by maturity in years, and the real par yields, named after the day of
            the curves.
        """
        real_par = self._real_yield_curve(country, date)
        day = str(real_par.name)
        nominal_zero = self._zero_curve(self._latest_yield_curve(country, day), country)
        maturities, real_rates = curve_model.bootstrap_zero_rates(
            real_par.index.to_numpy(dtype=float),
            real_par.to_numpy(dtype=float),
            REAL_COUPON_FREQUENCIES[country],
            None,
            return_grid=True,
        )

        return nominal_zero, pd.Series(real_rates, index=maturities), real_par

    @handle_errors
    def get_inflation_swap_rate(
        self,
        country: str = "United States",
        date: str | None = None,
        maturities: list[float] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the zero-coupon and year-on-year inflation swap rates the government bond
        market implies: the inflation the market expects, as the fair fixed rates of swaps
        on US CPI-U inflation (from Treasury and TIPS yields) or euro area HICP inflation
        excluding tobacco (from nominal and inflation-linked German federal securities).

        A zero-coupon inflation swap pays the growth of the price index over its life against
        a fixed rate compounded over the same years; a year-on-year swap pays each year's
        inflation against a fixed rate every year. With nominal and real zero-coupon curves,
        the forward index of maturity T is I(0) * P_real(T) / P_nominal(T) (Jarrow and
        Yildirim, 2003), so the zero-coupon swap rate is the zero-coupon breakeven inflation
        rate and the year-on-year swap rate the discount-weighted average of the forward
        year-on-year rates (Kerkhof, 2005; Brigo and Mercurio, 2006, ch. 15).

        The formula is as follows:

        - Zero-Coupon Inflation Swap Rate: K(T) = (1 + n(T)) / (1 + r(T)) - 1
        - Forward Year-on-Year Rate: f_i = (1 + K(i))^i / (1 + K(i - 1))^(i - 1) - 1
        - Year-on-Year Inflation Swap Rate = SUM(P(i) f_i) / SUM(P(i)), over i = 1, ..., T

        where n(T) and r(T) are the annually compounded nominal and real zero-coupon rates of
        maturity T and P(i) = (1 + n(i))^(-i) the nominal discount factor. The nominal zero
        rates are bootstrapped from the government par yield curve (see
        `get_zero_coupon_curve`), the real ones from the real par yields of the
        inflation-linked bonds with their coupons (twice a year for TIPS, once for German
        inflation-linked German federal securities), held at the shortest quoted real yield before its maturity.
        Every rate is annually compounded. The year-on-year rates carry no convexity
        adjustment, which needs the volatilities of real and nominal rates: it is a few basis
        points up to ten years.

        Referenced against:
        - QuantLib, ZeroCouponInflationSwap.fairRate: https://www.quantlib.org/reference/class_quant_lib_1_1_zero_coupon_inflation_swap.html
        - QuantLib, YearOnYearInflationSwap.fairRate: https://www.quantlib.org/reference/class_quant_lib_1_1_year_on_year_inflation_swap.html

        Also known as: ZCIS rate, YoY swap rate, zero-coupon breakeven inflation, inflation
        swap curve.

        Args:
            country (str, optional): The country. Options are: "United States" (CPI-U, Treasury and
                TIPS curves), "Germany" (euro area HICP excluding tobacco, Bundesbank curves).
                Defaults to "United States".
            date (str | None, optional): The day (YYYY-MM-DD), or the last day before it with a
                real yield curve. Defaults to None, the end date.
            maturities (list[float] | str | None, optional): The swap maturities in years, or
                labels such as "5Y", as a list or a comma-separated string, all positive. A
                maturity beyond the longest of the nominal and real curves holds their zero
                rates flat from there, which is logged. Defaults to None, the maturities of
                the real yield curve.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per maturity (in years), the Nominal Zero Rate, Real Zero Rate,
            Zero-Coupon Inflation Swap Rate and Year-on-Year Inflation Swap Rate (decimals,
            annually compounded; the year-on-year rate only for whole years) and the Date of
            the curves.

        Notes:
        - The rates are breakevens of government bonds. Traded inflation swaps are quoted a
          few basis points above them (the inflation swap basis), which reflects the
          liquidity of inflation-linked bonds and the funding of the hedge.
        - The breakevens include the inflation risk premium, so they are not a pure forecast
          of inflation.
        - The index of the swaps is observed with the lag of the bonds (three months for TIPS
          and inflation-linked German federal securities).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date="2026-10-01", end_date="2026-11-20")

        fixedincome.get_inflation_swap_rate(
            show_columns=["Zero-Coupon Inflation Swap Rate", "Year-on-Year Inflation Swap Rate", "Date"]
        )
        ```

        Which returns:

        |   Maturity |   Zero-Coupon Inflation Swap Rate |   Year-on-Year Inflation Swap Rate | Date       |
        |-----------:|----------------------------------:|-----------------------------------:|:-----------|
        |          5 |                            0.0217 |                             0.0216 | 2026-11-20 |
        |          7 |                            0.0215 |                             0.0215 | 2026-11-20 |
        |         10 |                            0.0215 |                             0.0215 | 2026-11-20 |
        |         20 |                            0.0242 |                             0.0234 | 2026-11-20 |
        |         30 |                            0.0211 |                             0.022  | 2026-11-20 |
        """
        nominal_zero, real_zero, real_par = self._inflation_curves(country, date)
        years = parse_maturities(maturities, list(real_par.index))
        if not len(years) or min(years) <= 0:
            raise ValueError(
                f"The maturities must be positive numbers of years, received {maturities!r}."
            )
        longest = float(min(nominal_zero.index.max(), real_zero.index.max()))
        if beyond := [year for year in years if year > longest]:
            logger.info(
                "The maturities %s lie beyond the longest maturity of the curves (%g years), "
                "whose zero rates are held flat beyond it.",
                ", ".join(f"{year:g}" for year in beyond),
                longest,
            )
        whole_years = int(np.ceil(max(years)))

        zero_inflation = inflation_swap_model.get_zero_coupon_inflation_swap_rate(
            nominal_zero, real_zero, np.arange(1, whole_years + 1)
        )
        year_on_year = inflation_swap_model.get_year_on_year_inflation_swap_rate(
            nominal_zero, zero_inflation, np.arange(1, whole_years + 1)
        )
        zero_coupon = inflation_swap_model.get_zero_coupon_inflation_swap_rate(
            nominal_zero, real_zero, years
        )
        rates = pd.DataFrame(
            {
                "Nominal Zero Rate": np.interp(
                    years, nominal_zero.index.to_numpy(dtype=float), nominal_zero
                ),
                "Real Zero Rate": np.interp(
                    years, real_zero.index.to_numpy(dtype=float), real_zero
                ),
                "Zero-Coupon Inflation Swap Rate": zero_coupon.to_numpy(),
                "Year-on-Year Inflation Swap Rate": [
                    (
                        year_on_year.get(int(year), np.nan)
                        if float(year).is_integer()
                        else np.nan
                    )
                    for year in years
                ],
                "Date": str(real_par.name),
            },
            index=pd.Index(years, name="Maturity"),
        )

        return apply_rounding(
            rates, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_forward_inflation_index(
        self,
        date: str | None = None,
        years: int = 5,
        seasonality: bool = True,
        seasonality_years: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the forward US consumer price index (CPI-U, not seasonally adjusted) of
        every coming month: the price level the inflation swap rates the Treasury and TIPS
        curves imply, with the monthly seasonality of the CPI.

        The zero-coupon inflation swap rates fix the growth of the index over whole years
        (see `get_inflation_swap_rate`). Between them, prices rise faster in some months than
        in others, so each month's forward index is moved by its seasonal factor relative to
        that of the base month (Belgrade, Benhamou and Koehler, 2004; Kerkhof, 2005). The
        seasonal factors are estimated from the CPI history: the average log change of each
        calendar month above the average of all months, accumulated over the year and scaled
        so that the twelve factors multiply to 1.

        The formula is as follows:

        - I(m) = I(0) * (1 + K(m / 12))^(m / 12) * S_(month of m) / S_(base month)
        - K(T) = (1 + n(T)) / (1 + r(T)) - 1
        - s_k = MEAN(ln(I_t / I_(t-1)) in calendar month k) - MEAN(ln(I_t / I_(t-1)))
        - S_k = exp(s_1 + ... + s_k), scaled so that PRODUCT(S_k) = 1

        where I(0) is the CPI of the base month, m the months after it, K the annually
        compounded zero-coupon inflation swap rate (interpolated linearly between whole
        years), n and r the annually compounded nominal and real zero rates of the Treasury
        and TIPS curves and S the seasonal factors (all 1 without seasonality). The base month
        is the third month before the date's month, the CPI the reference CPI of TIPS starts
        from on the first of the month (31 CFR 356), so the breakevens price the inflation from
        it on.

        Referenced against:
        - QuantLib, ZeroInflationIndex: https://www.quantlib.org/reference/class_quant_lib_1_1_zero_inflation_index.html
        - QuantLib, MultiplicativePriceSeasonality: https://www.quantlib.org/reference/class_quant_lib_1_1_multiplicative_price_seasonality.html

        Also known as: forward CPI, CPI fixing curve, inflation index curve, seasonal
        adjustment of inflation swaps.

        Args:
            date (str | None, optional): The day (YYYY-MM-DD) of the curves, or the last day
                before it with a real yield curve. Defaults to None, the end date.
            years (int, optional): The number of years of months after the base month.
                Defaults to 5.
            seasonality (bool, optional): Whether to apply the monthly seasonal factors.
                Defaults to True.
            seasonality_years (int, optional): The years of CPI history before the base month
                the seasonal factors are estimated on. The keyless BLS API covers about three
                years; a FRED API key gives the rest. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per month after the base month, the Forward CPI (with the seasonal
            factors when seasonality is True), the Trend CPI (without them), the Seasonal
            Factor relative to the base month and the Zero-Coupon Inflation Swap Rate of the
            horizon (annually compounded).

        Notes:
        - Without a FRED API key the seasonality is estimated on the CPI the BLS returns
          (about three years before the base month), which is logged.
        - The swap rates are breakevens of government bonds, see `get_inflation_swap_rate`.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date="2026-10-01", end_date="2026-11-20")

        fixedincome.get_forward_inflation_index(years=1)
        ```

        Which returns:

        | Month   |   Forward CPI |   Trend CPI |   Seasonal Factor |   Zero-Coupon Inflation Swap Rate |
        |:--------|--------------:|------------:|------------------:|----------------------------------:|
        | 2026-09 |       335.388 |     335.524 |            0.9996 |                            0.0196 |
        | 2026-10 |       335.709 |     336.068 |            0.9989 |                            0.0196 |
        | 2026-11 |       334.568 |     336.613 |            0.9939 |                            0.0196 |
        | 2026-12 |       334.306 |     337.16  |            0.9915 |                            0.0196 |
        | 2027-01 |       335.731 |     337.707 |            0.9941 |                            0.0196 |
        | 2027-02 |       337.16  |     338.255 |            0.9968 |                            0.0196 |
        | 2027-03 |       339.026 |     338.804 |            1.0007 |                            0.0196 |
        | 2027-04 |       340.487 |     339.354 |            1.0033 |                            0.0196 |
        | 2027-05 |       341.337 |     339.904 |            1.0042 |                            0.0196 |
        | 2027-06 |       341.073 |     340.456 |            1.0018 |                            0.0196 |
        | 2027-07 |       341.073 |     341.008 |            1.0002 |                            0.0196 |
        | 2027-08 |       341.562 |     341.562 |            1      |                            0.0196 |
        """
        nominal_zero, real_zero, real_par = self._inflation_curves(
            "United States", date
        )
        day_date = date_type.fromisoformat(str(real_par.name))
        base_day = day_date.replace(day=1) - relativedelta(
            months=inflation_bond_model.REFERENCE_LAG_MONTHS
        )
        base_month = pd.Period(base_day, freq="M")
        months = int(years) * inflation_swap_model.MONTHS_PER_YEAR

        try:
            cpi = self._consumer_price_index(
                base_day - relativedelta(years=int(seasonality_years))
                if seasonality
                else base_day
            )
        except ValueError as error:
            if not seasonality:
                raise
            # The keyless BLS API covers about three years: the seasonality is estimated on
            # those rather than not at all, and the shorter history is logged.
            cpi = self._consumer_price_index(base_day)
            logger.warning(
                "The seasonality is estimated on the CPI from %s: %s",
                cpi.index[0],
                logger_model.redact(error),
            )
        # Only the months up to the base month, which were published before the date.
        first_month = pd.Period(
            base_day - relativedelta(years=int(seasonality_years)), freq="M"
        )
        cpi = cpi.loc[first_month:base_month]
        if base_month not in cpi.index or not np.isfinite(cpi[base_month]):
            raise ValueError(f"The CPI of {base_month} is not available.")

        zero_inflation = inflation_swap_model.get_zero_coupon_inflation_swap_rate(
            nominal_zero, real_zero, np.arange(1, int(years) + 1)
        )
        factors = (
            inflation_swap_model.get_inflation_seasonality(cpi) if seasonality else None
        )
        trend = inflation_swap_model.get_forward_inflation_index(
            float(cpi[base_month]), zero_inflation, months, base_day.month
        )
        forward = inflation_swap_model.get_forward_inflation_index(
            float(cpi[base_month]), zero_inflation, months, base_day.month, factors
        )
        horizon = np.arange(1, months + 1) / inflation_swap_model.MONTHS_PER_YEAR

        result = pd.DataFrame(
            {
                "Forward CPI": forward.to_numpy(),
                "Trend CPI": trend.to_numpy(),
                "Seasonal Factor": (forward / trend).to_numpy(),
                "Zero-Coupon Inflation Swap Rate": np.interp(
                    horizon,
                    zero_inflation.index.to_numpy(dtype=float),
                    zero_inflation.to_numpy(),
                ),
            },
            index=pd.period_range(
                base_day + relativedelta(months=1), periods=months, freq="M"
            ),
        )
        result.index.name = "Month"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
