"""Swaps and Forwards Module"""

__docformat__ = "google"

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from financetoolkit import helpers
from financetoolkit.economics.sources import bis_model
from financetoolkit.fixedincome.helpers import (
    maturity_in_years,
    maturity_key,
    maturity_label,
    parse_maturities,
)
from financetoolkit.fixedincome.markets import bond_data_model
from financetoolkit.fixedincome.markets.markets_controller import MarketsFixedIncome
from financetoolkit.fixedincome.sources import eiopa_model
from financetoolkit.fixedincome.swaps import forward_model, swap_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import check_option, handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = logger_model.get_logger()

# The curves a swap, forward or spread can be priced on: the EIOPA risk-free rate term
# structure, which EIOPA derives from swap rates (monthly, 1 to 150 years), or the
# zero-coupon curve bootstrapped from the government bond yield curve (daily).
CURVES = ("eiopa", "government")

DEFAULT_SWAP_MATURITIES = (1, 2, 3, 5, 7, 10, 15, 20, 30)
DEFAULT_FRAS = ("1x4", "3x6", "6x9", "6x12", "9x12", "12x18", "12x24")
DEFAULT_FX_COUNTRIES = (
    "Euro Area",
    "United Kingdom",
    "Japan",
    "Switzerland",
    "Canada",
    "Australia",
)
DEFAULT_FX_TENORS = ("1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y")

# The bonds of get_bond_universe pay coupons twice a year and are priced per 100 of par.
BOND_COUPON_FREQUENCY = 2


@filter_shown_columns
@track_sources
class SwapsFixedIncome(MarketsFixedIncome):
    """
    Swaps and forwards on a single curve: par swap rates and annuities, forward rate agreements, the asset swap and I-spreads of corporate bonds and FX forward rates by covered interest parity.
    """

    def _swap_zero_curve(
        self, curve: str, country: str, date: str | None = None
    ) -> pd.Series:
        """
        The zero-coupon curve a swap or forward is priced on, on a day.

        Args:
            curve (str): "eiopa" for the EIOPA risk-free rate term structure (basic spot
                curve, without volatility adjustment) of the last month that ended on or
                before the day, or "government" for the zero-coupon curve of the government
                bond yield curve of the day (see get_zero_coupon_curve).
            country (str): The country or currency area, as EIOPA or
                get_government_bond_yield_curve names it.
            date (str | None): The day (YYYY-MM-DD). Defaults to None, the end date.

        Returns:
            pd.Series: The annually compounded zero-coupon rates by maturity in years.
        """
        check_option("curve", curve, CURVES)
        if curve == "government":
            return self._latest_zero_curve(country, date)

        end_date = date or self._end_date
        # A release is the curve at the end of its month; the months before cover the days
        # before EIOPA publishes it (early in the next month).
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=100)
        ).strftime("%Y-%m-%d")
        term_structures = eiopa_model.get_risk_free_rate_term_structures(
            "spot_no_va", start_date, end_date
        )
        if term_structures.empty:
            # The source has logged why (it could not be reached, or published nothing).
            raise ValueError(
                f"EIOPA returned no risk-free rate term structures between {start_date} and "
                f"{end_date}, so there is no curve to price on; curve='government' uses the "
                "government bond yield curve instead."
            )
        countries = list(dict.fromkeys(term_structures.columns.get_level_values(0)))
        check_option("country", country, countries)
        term_structure = term_structures[country].dropna(how="all")
        # Only the months that ended by the day, so the curve is not ahead of it.
        term_structure = term_structure[
            term_structure.index.end_time.normalize() <= pd.Timestamp(end_date)
        ]
        if term_structure.empty:
            raise ValueError(
                f"EIOPA has no risk-free rate term structure for {country} at a month end "
                f"in the 100 days before {end_date}."
            )
        latest = term_structure.iloc[-1].dropna()

        return pd.Series(
            latest.to_numpy(dtype=float),
            index=[maturity_in_years(label) for label in latest.index],
        ).sort_index()

    @handle_errors
    def get_swap_rate(
        self,
        maturities: float | str | list | None = None,
        country: str = "United States",
        curve: str = "eiopa",
        fixed_frequency: int = 1,
        forward_start: float | str = 0,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the par swap rates and annuities of interest rate swaps of any maturity on a
        single curve: the fixed rate at which a new fixed-for-floating swap is worth nothing,
        and the present value of receiving 1 a year on its fixed leg. With a forward start,
        these are the forward swap rates and annuities that price a swaption.

        On a single curve, which both projects the floating rate and discounts, a floating leg
        that resets at the start of each period is worth par at its start, so the swap rate
        follows from the discount factors alone (Hull, 2022, Options, Futures and Other
        Derivatives, ch. 7; Andersen and Piterbarg, 2010, Interest Rate Modeling, ch. 5).

        The formula is as follows:

        - Annuity = SUM(τ_i * P(t_i)), over the fixed payment dates t_1, ..., t_n
        - Swap Rate = (P(t_0) - P(t_n)) / Annuity

        where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded zero
        rate r(t) of the curve (interpolated linearly between its maturities), t_0 the start of
        the swap and τ_i = 1 / fixed_frequency the accrual of a fixed period (a short first
        period when the maturity is not a whole number of periods). The swap rate is a nominal
        annual rate paid fixed_frequency times a year on year fractions with no calendar (a
        regular 30/360 schedule). The value of a basis point on the fixed rate (PV01) of a swap
        is Annuity * Notional * 0.0001.

        The default curve is EIOPA's risk-free rate term structure, which EIOPA derives from
        the swap rates of each currency (for the euro, swap rates minus a credit risk
        adjustment) and publishes at each month end; the government curve is the zero-coupon
        curve of the government bond yield curve of the day, whose par rates are those
        of government bonds rather than swaps. To value a swap of your own, with a fixed rate,
        a current fixing and its DV01, use `swap_model.get_swap_value`.

        Referenced against:
        - QuantLib, VanillaSwap.fairRate: https://www.quantlib.org/reference/class_quant_lib_1_1_vanilla_swap.html
        - rateslib, IRS.rate: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.IRS.html#rateslib.instruments.IRS.rate

        Shares similar logic:
        - FinancePy, IborSwap.swap_rate: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/rates/ibor_swap.py#L280

        Also known as: par swap rate, swap curve, interest rate swap rate, forward swap rate,
        PV01, swap annuity, PVBP.

        Args:
            maturities (float | str | list | None, optional): The maturities (tenors) of the
                swaps, in years or as labels such as '10Y'. Defaults to None, 1, 2, 3, 5, 7, 10,
                15, 20 and 30 years.
            country (str, optional): The country or currency area of the curve, as EIOPA
                (for the EIOPA curve) or `get_government_bond_yield_curve` (for the government
                curve) names it. Defaults to "United States".
            curve (str, optional): The discount curve. Options are: "eiopa" (EIOPA's risk-free rate term structure of
                the last month end on or before the date), "government" (the zero-coupon curve of the government bond
                yield curve of the date, see `get_zero_coupon_curve`). Defaults to "eiopa".
            fixed_frequency (int, optional): The number of fixed payments per year. Defaults to
                1 (annual, as euro and US dollar overnight index swaps pay).
            forward_start (float | str, optional): The start of the swaps, in years or as a
                label such as '1Y', for forward starting swaps. Defaults to 0, swaps that start
                today.
            date (str | None, optional): The day of the curve (YYYY-MM-DD). Defaults to None,
                the end date.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Swap Rate (decimal, compounded fixed_frequency times a year) and
            the Annuity (years, per unit of notional) by maturity in years.

        Notes:
        - EIOPA's curves start at 1 year, so a shorter maturity discounts at the 1-year zero
          rate; beyond the last liquid point (20 years for the euro, 50 for the US dollar)
          they converge to the ultimate forward rate, a regulatory assumption rather than a
          market price.
        - A single curve leaves out the basis between the overnight and the term rates
          (SOFR against term SOFR, €STR against Euribor), so these are the rates of swaps on
          the curve's own rate.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-10-01', end_date='2026-11-20')

        fixedincome.get_swap_rate(maturities=[1, 2, 5, 10, 30], country='Euro Area')
        ```

        Which returns:

        |   Maturity |   Swap Rate |   Annuity |
        |-----------:|------------:|----------:|
        |          1 |      0.0314 |    0.9696 |
        |          2 |      0.0327 |    1.9072 |
        |          5 |      0.0329 |    4.5437 |
        |         10 |      0.0334 |    8.3984 |
        |         30 |      0.0332 |   18.7673 |
        """
        start = maturity_in_years(forward_start)
        zero_curve = self._swap_zero_curve(curve, country, date)

        swap_rates = pd.DataFrame(
            {
                tenor: swap_model.get_swap_rate(
                    zero_curve,
                    tenor=tenor,
                    fixed_frequency=fixed_frequency,
                    forward_start=start,
                )
                for tenor in parse_maturities(maturities, DEFAULT_SWAP_MATURITIES)
            },
            index=["Swap Rate", "Annuity"],
        ).T
        swap_rates.index.name = "Maturity"

        return apply_rounding(
            swap_rates, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_forward_rate_agreement(
        self,
        fras: str | list[str] | None = None,
        country: str = "United States",
        curve: str = "government",
        day_count: str = "act/360",
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the rates of forward rate agreements (FRAs) from a zero-coupon curve: the
        simple interest rate, fixed today, of a loan that starts and ends at future dates.

        A "3x6" FRA covers the three months that start three months from now.

        At no arbitrage the FRA rate is the simple forward rate implied by the discount factors
        of the two dates (Hull, 2022, Options, Futures and Other Derivatives, ch. 4.7). A FRA
        that pays a fixed rate K and receives the floating rate is worth
        Notional * τ * (FRA Rate - K) * P(end), see `forward_model.get_forward_rate_agreement`.

        The formula is as follows:

        - τ = (End - Start) * 365 / Days in Year
        - FRA Rate = (P(Start) / P(End) - 1) / τ

        where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded zero
        rate r(t) of the curve, interpolated linearly between its maturities, the times are
        years (a month is a twelfth of a year) and Days in Year is 360 on ACT/360, the money
        market day count of US dollar and euro rates, or 365 on ACT/365. The FRA rate is
        simple interest on that day count.

        The default curve is the zero-coupon curve of the government bond yield curve of the
        day (see `get_zero_coupon_curve`), whose bills price the first months; the EIOPA curve
        is EIOPA's swap-based risk-free rate term structure of the last month end, which
        starts at 1 year.

        Referenced against:
        - QuantLib, ForwardRateAgreement: https://www.quantlib.org/reference/class_quant_lib_1_1_forward_rate_agreement.html

        Shares similar logic:
        - FinancePy, IborFRA: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/rates/ibor_fra.py#L23
        - rateslib, FRA: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.FRA.html

        Also known as: FRA, forward rate agreement, forward-forward rate, simple forward rate.

        Args:
            fras (str | list[str] | None, optional): The FRAs as "start x end" in months, as a
                list or comma-separated, e.g. "3x6,6x12". Defaults to None, 1x4, 3x6, 6x9, 6x12,
                9x12, 12x18 and 12x24.
            country (str, optional): The country or currency area of the curve, see
                `get_government_bond_yield_curve` (or EIOPA's names for the EIOPA curve).
                Defaults to "United States".
            curve (str, optional): The discount curve. Options are: "eiopa" (EIOPA's risk-free rate term structure of
                the last month end on or before the date), "government" (the zero-coupon curve of the government bond
                yield curve of the date, see `get_zero_coupon_curve`). Defaults to "government".
            day_count (str, optional): The day count of the FRA rate. Options are: "act/360" (actual/360), "act/365"
                (actual/365). Defaults to "act/360".
            date (str | None, optional): The day of the curve (YYYY-MM-DD). Defaults to None,
                the end date.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per FRA the Start and End (years) and the FRA Rate (decimal, simple
            interest on the day count).

        Notes:
        - Government bill and bond yields sit below the interbank and overnight rates FRAs
          are settled on (the swap spread), so on the government curve these are risk-free
          forward rates rather than market FRA quotes.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-11-01', end_date='2026-11-20')

        fixedincome.get_forward_rate_agreement()
        ```

        Which returns:

        | FRA   |   Start |    End |   FRA Rate |
        |:------|--------:|-------:|-----------:|
        | 1x4   |  0.0833 | 0.3333 |     0.0429 |
        | 3x6   |  0.25   | 0.5    |     0.0433 |
        | 6x9   |  0.5    | 0.75   |     0.0449 |
        | 6x12  |  0.5    | 1      |     0.0458 |
        | 9x12  |  0.75   | 1      |     0.0461 |
        | 12x18 |  1      | 1.5    |     0.0463 |
        | 12x24 |  1      | 2      |     0.0475 |
        """
        check_option("day_count", day_count, forward_model.DAY_COUNTS)
        zero_curve = self._swap_zero_curve(curve, country, date)

        rows = {}
        for fra in helpers.split_values(fras) or DEFAULT_FRAS:
            try:
                start_months, end_months = (
                    float(months) for months in str(fra).lower().split("x")
                )
            except ValueError as error:
                raise ValueError(
                    f"A FRA is given as 'start x end' in months, such as '3x6', not {fra!r}."
                ) from error
            fra_rate, _ = forward_model.get_forward_rate_agreement(
                zero_curve, start_months / 12, end_months / 12, day_count=day_count
            )
            rows[str(fra).lower()] = {
                "Start": start_months / 12,
                "End": end_months / 12,
                "FRA Rate": fra_rate,
            }

        fra_rates = pd.DataFrame(rows).T
        fra_rates.index.name = "FRA"

        return apply_rounding(
            fra_rates, rounding if rounding is not None else self._rounding
        )

    def _bonds_on_swap_curve(
        self,
        curve: str,
        etfs: list[str] | str | None,
        issuer: list[str] | str | None,
        maturity_start: str | None,
        maturity_end: str | None,
        bonds: list[str] | str | None,
    ) -> tuple[pd.DataFrame, list[pd.Series]]:
        """
        The bonds of get_bond_universe with their years to maturity, and the US dollar zero
        curve of each bond's Price Date.

        Args:
            curve (str): The curve, see _swap_zero_curve.
            etfs, issuer, maturity_start, maturity_end, bonds: See get_bond_universe.

        Returns:
            tuple[pd.DataFrame, list[pd.Series]]: The bonds, unrounded, and the zero curve
            of each bond in the same order.
        """
        # Unrounded, as a yield rounded to a basis point would move the spreads.
        universe = self.get_bond_universe(
            etfs=etfs,
            issuer=issuer,
            maturity_start=maturity_start,
            maturity_end=maturity_end,
            bonds=bonds,
            rounding=10,
        )
        if universe.empty:
            raise ValueError("No bonds were found for the bonds and issuers given.")

        # The years to maturity on the 30/360 Bond Basis of US corporate bonds, as
        # get_bond_universe prices them.
        universe = universe.assign(
            Years=[
                bond_data_model.get_years_to_maturity(
                    pd.Timestamp(settlement).date(),
                    pd.Timestamp(maturity).date(),
                    day_count="30/360",
                )
                for settlement, maturity in zip(
                    universe["Price Date"], universe["Maturity"], strict=True
                )
            ]
        )
        universe = universe[universe["Years"].notna() & (universe["Years"] > 0)]
        curves = {
            price_date: self._swap_zero_curve(curve, "United States", price_date)
            for price_date in universe["Price Date"].unique()
        }

        return universe, [curves[price_date] for price_date in universe["Price Date"]]

    @handle_errors
    def get_asset_swap_spread(
        self,
        issuer: list[str] | str | None = None,
        bonds: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        etfs: list[str] | str | None = None,
        curve: str = "eiopa",
        float_frequency: int = 4,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the par and the market value asset swap spreads of US dollar corporate bonds
        over the swap curve: the spread over the floating rate at which a swap that pays away
        the bond's coupons turns the bond into a floating rate asset worth what the investor
        pays. It measures a bond's credit and liquidity against swaps rather than against
        government bonds, as banks that fund at floating rates see it (O'Kane, 2000,
        "Introduction to Asset Swaps", Lehman Brothers; Tuckman and Serrat, 2022, Fixed Income
        Securities, ch. 18).

        In a par asset swap the investor pays par for the bond and the swap together, and the
        swap settles the difference between par and the bond's price; in a market value asset
        swap the investor pays the bond's price and the floating leg runs on that amount.

        The formula is as follows:

        - Curve Price = 100 * (SUM(c / 2 * P(t_i)) + P(t_n)), the dirty price of the bond on the swap curve
        - Annuity = SUM(δ_j * P(s_j)), over the floating payment dates s_j
        - Par Asset Swap Spread = (Curve Price - Dirty Price) / (100 * Annuity)
        - Market Value Asset Swap Spread = (Curve Price - Dirty Price) / (Dirty Price * Annuity)

        where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded US dollar
        zero rate r(t) of the curve on the bond's Price Date, c the coupon rate, t_i the
        semi-annual coupon dates counted back from maturity in years on the 30/360 Bond Basis
        (as `get_bond_universe` prices the bonds), Dirty Price = Price + Accrued Interest per
        100 of par, and δ_j = 1 / float_frequency the accrual of the floating periods from the
        Price Date to maturity (a short first period). The spreads are simple annual rates on
        the floating leg.

        The default curve is EIOPA's US dollar risk-free rate term structure, which EIOPA
        derives from US dollar swap rates, of the last month end on or before each bond's Price
        Date; the government curve is the Treasury zero-coupon curve of the Price Date instead,
        which gives the asset swap spread over Treasuries. For a bond and curve of your own, use
        `swap_model.get_asset_swap_spread`.

        Referenced against:
        - QuantLib, AssetSwap.fairSpread: https://www.quantlib.org/reference/class_quant_lib_1_1_asset_swap.html

        Also known as: ASW, asset swap spread, par-par asset swap, proceeds asset swap, market
        value asset swap.

        Args:
            issuer (list[str] | str | None, optional): Keeps the bonds whose issuer contains this
                text (case-insensitive), e.g. "APPLE", or any of several. Defaults to None,
                every issuer.
            bonds (list[str] | str | None, optional): Keeps the bonds with these ISINs or
                CUSIPs, as a list or a comma-separated string. Defaults to None, every bond.
            maturity_start (str | None, optional): Keeps the bonds maturing on or after this
                date (YYYY-MM-DD). Defaults to None.
            maturity_end (str | None, optional): Keeps the bonds maturing on or before this
                date (YYYY-MM-DD). Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs to read the bonds from (see
                `get_bond_universe`). Defaults to None, LQD, IGSB and USHY.
            curve (str, optional): The swap curve. Options are: "eiopa" (EIOPA's US dollar risk-free rate term
                structure), "government" (the Treasury zero-coupon curve). Defaults to "eiopa".
            float_frequency (int, optional): The number of floating payments per year of the
                swap. Defaults to 4 (quarterly).
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond, indexed by ISIN, with the Issuer, Coupon (decimal),
            Maturity, Price (clean, per 100 of par), the Par Asset Swap Spread and Market Value
            Asset Swap Spread (decimals) and the Price Date.

        Notes:
        - EIOPA publishes its curve at each month end, so the curve can be up to a month and
          a half older than the bond prices; the government curve is that of the same day.
        - The bonds and their prices are those of `get_bond_universe`; a callable bond is
          treated as a bullet bond to maturity.
        - Requires a FinancialModelingPrep API key for the holdings.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date='2026-11-20', api_key="FINANCIAL_MODELING_PREP_KEY")

        spreads = fixedincome.get_asset_swap_spread(issuer="APPLE INC")

        spreads[["Coupon", "Maturity", "Price", "Par Asset Swap Spread", "Market Value Asset Swap Spread"]].head()
        ```

        Which returns:

        | ISIN         |   Coupon | Maturity   |   Price |   Par Asset Swap Spread |   Market Value Asset Swap Spread |
        |:-------------|---------:|:-----------|--------:|------------------------:|---------------------------------:|
        | US037833EC07 |   0.012  | 2028-02-08 |   95.85 |                  0.0039 |                           0.0041 |
        | US037833ET32 |   0.04   | 2028-05-10 |   98.9  |                  0.0042 |                           0.0042 |
        | US037833EY27 |   0.04   | 2028-05-12 |   98.87 |                  0.0043 |                           0.0044 |
        | US037833EH93 |   0.014  | 2028-08-05 |   94.55 |                  0.0036 |                           0.0038 |
        | US037833FD70 |   0.0465 | 2028-11-09 |   99.74 |                  0.0039 |                           0.0039 |
        """
        universe, curves = self._bonds_on_swap_curve(
            curve, etfs, issuer, maturity_start, maturity_end, bonds
        )
        spreads = [
            swap_model.get_asset_swap_spread(
                zero_curve,
                coupon_rate=coupon,
                years_to_maturity=years,
                clean_price=price,
                frequency=BOND_COUPON_FREQUENCY,
                float_frequency=float_frequency,
            )
            for zero_curve, coupon, years, price in zip(
                curves,
                universe["Coupon"],
                universe["Years"],
                universe["Price"],
                strict=True,
            )
        ]
        universe["Par Asset Swap Spread"] = [spread[0] for spread in spreads]
        universe["Market Value Asset Swap Spread"] = [spread[1] for spread in spreads]

        return apply_rounding(
            universe[
                [
                    "Issuer",
                    "Coupon",
                    "Maturity",
                    "Price",
                    "Par Asset Swap Spread",
                    "Market Value Asset Swap Spread",
                    "Price Date",
                ]
            ],
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_i_spread(
        self,
        issuer: list[str] | str | None = None,
        bonds: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        etfs: list[str] | str | None = None,
        curve: str = "eiopa",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the I-spread (interpolated spread) of US dollar corporate bonds: the yield to
        maturity over the par swap rate of the bond's maturity. The I-spread compares a bond
        with the swap curve the way the G-spread compares it with Treasuries (Tuckman and
        Serrat, 2022, Fixed Income Securities, ch. 18); the difference between the two is the
        swap spread at the bond's maturity.

        The formula is as follows:

        - I-Spread = Yield to Maturity - Swap Rate(T)

        where the Yield to Maturity is that of `get_bond_universe` (a nominal annual rate
        compounded twice a year) and Swap Rate(T) the par swap rate of `get_swap_rate` for the
        bond's years to maturity T on the 30/360 Bond Basis, with semi-annual fixed payments so
        that both rates are compounded alike (a short first period when T is not a whole
        number of half years). The swap rate is priced on the annually compounded US dollar
        zero curve of the bond's Price Date.

        The default curve is EIOPA's US dollar risk-free rate term structure, which EIOPA
        derives from US dollar swap rates, of the last month end on or before each bond's Price
        Date; the government curve is the Treasury zero-coupon curve, which makes the I-spread
        a spread over the Treasury par curve, close to the G-Spread. For a bond and curve of
        your own, use `swap_model.get_i_spread`.

        Also known as: interpolated spread, I-spread, spread over swaps, swap spread of a bond.

        Args:
            issuer (list[str] | str | None, optional): Keeps the bonds whose issuer contains this
                text (case-insensitive), e.g. "APPLE", or any of several. Defaults to None,
                every issuer.
            bonds (list[str] | str | None, optional): Keeps the bonds with these ISINs or
                CUSIPs, as a list or a comma-separated string. Defaults to None, every bond.
            maturity_start (str | None, optional): Keeps the bonds maturing on or after this
                date (YYYY-MM-DD). Defaults to None.
            maturity_end (str | None, optional): Keeps the bonds maturing on or before this
                date (YYYY-MM-DD). Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs to read the bonds from (see
                `get_bond_universe`). Defaults to None, LQD, IGSB and USHY.
            curve (str, optional): The swap curve. Options are: "eiopa" (EIOPA's US dollar risk-free rate term
                structure), "government" (the Treasury zero-coupon curve). Defaults to "eiopa".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond, indexed by ISIN, with the Issuer, Coupon (decimal),
            Maturity, the Yield to Maturity, Swap Rate, I-Spread and G-Spread (decimals) and
            the Price Date.

        Notes:
        - EIOPA publishes its curve at each month end, so the curve can be up to a month and
          a half older than the bond prices; the government curve is that of the same day.
        - The yield is to maturity: a callable bond priced above its call price yields less
          to its call date.
        - Requires a FinancialModelingPrep API key for the holdings.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date='2026-11-20', api_key="FINANCIAL_MODELING_PREP_KEY")

        spreads = fixedincome.get_i_spread(issuer="APPLE INC")

        spreads[["Coupon", "Maturity", "Yield to Maturity", "Swap Rate", "I-Spread", "G-Spread"]].head()
        ```

        Which returns:

        | ISIN         |   Coupon | Maturity   |   Yield to Maturity |   Swap Rate |   I-Spread |   G-Spread |
        |:-------------|---------:|:-----------|--------------------:|------------:|-----------:|-----------:|
        | US037833EC07 |   0.012  | 2028-02-08 |              0.0475 |      0.0434 |     0.0041 |     0.0021 |
        | US037833ET32 |   0.04   | 2028-05-10 |              0.0478 |      0.0436 |     0.0042 |     0.0022 |
        | US037833EY27 |   0.04   | 2028-05-12 |              0.048  |      0.0436 |     0.0044 |     0.0023 |
        | US037833EH93 |   0.014  | 2028-08-05 |              0.0476 |      0.0437 |     0.0038 |     0.0016 |
        | US037833FD70 |   0.0465 | 2028-11-09 |              0.0479 |      0.044  |     0.0039 |     0.0016 |
        """
        universe, curves = self._bonds_on_swap_curve(
            curve, etfs, issuer, maturity_start, maturity_end, bonds
        )
        universe["Swap Rate"] = [
            swap_model.get_swap_rate(zero_curve, years, BOND_COUPON_FREQUENCY)[0]
            for zero_curve, years in zip(curves, universe["Years"], strict=True)
        ]
        universe["I-Spread"] = universe["Yield to Maturity"] - universe["Swap Rate"]

        return apply_rounding(
            universe[
                [
                    "Issuer",
                    "Coupon",
                    "Maturity",
                    "Yield to Maturity",
                    "Swap Rate",
                    "I-Spread",
                    "G-Spread",
                    "Price Date",
                ]
            ],
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_fx_forward_rates(
        self,
        countries: list[str] | str | None = None,
        base_country: str = "United States",
        tenors: float | str | list | None = None,
        curve: str = "eiopa",
        points: bool = False,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates FX forward rates by covered interest parity: the exchange rate, agreed
        today, at which currencies are exchanged at a future date.

        Exchanging today and depositing abroad, or depositing at home and exchanging
        forward, must give the same, so the forward rate is the spot rate carried at the
        difference between the two currencies' interest rates: the currency with the higher
        rate trades at a forward discount (Hull, 2022, Options, Futures and Other
        Derivatives, ch. 5.10).

        The formula is as follows:

        - Forward Rate = Spot Rate * ((1 + r_country(T)) / (1 + r_base(T)))^T
        - Forward Points = (Forward Rate - Spot Rate) * 10,000

        where the Spot Rate is in units of the country's currency per unit of the base
        country's currency, from the daily exchange rates against the US dollar the Bank for
        International Settlements compiles (the last day with a rate on or before the date),
        and r_country(T) and r_base(T) are the annually compounded zero rates of the two
        currencies at the tenor T in years (a month is a twelfth of a year). The value of a
        forward you hold follows from `forward_model.get_fx_forward_rate`.

        The default curves are EIOPA's risk-free rate term structures, which EIOPA derives from
        the swap rates of each currency and publishes at each month end for the euro and about 20 other currencies;
        they start at 1 year, so a shorter tenor uses the 1-year zero rate. The government
        curves are the zero-coupon curves of the government bond yield curves of the date, whose
        bills price the first months, for the countries that have one (see
        `get_zero_coupon_curve`).

        Referenced against:
        - rateslib, forward_fx: https://rateslib.com/py/en/3.0.x/api/rateslib.fx.forward_fx.html

        Shares similar logic:
        - QuantLib, FxForward: https://www.quantlib.org/reference/class_quant_lib_1_1_fx_forward.html
        - FinancePy, FXForward: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/fx/fx_forward.py#L20

        Also known as: FX forward, currency forward, forward exchange rate, forward points,
        swap points, covered interest parity, outright forward.

        Args:
            countries (list[str] | str | None, optional): The countries or currency areas whose
                currency is quoted, as EIOPA and the BIS name them (e.g. "Euro Area", "Japan").
                Defaults to None, the Euro Area, the United Kingdom, Japan, Switzerland, Canada
                and Australia.
            base_country (str, optional): The country of the base currency, one unit of which
                the rates are quoted against. Defaults to "United States".
            tenors (float | str | list | None, optional): The tenors, in years or as labels such
                as '3M' or '5Y'. Defaults to None, 1, 3 and 6 months and 1, 2, 5 and 10 years.
            curve (str, optional): The discount curves. Options are: "eiopa" (EIOPA's risk-free rate term structures of
                the last month end on or before the date), "government" (the zero-coupon curves of the government bond
                yield curves of the date). Defaults to "eiopa".
            points (bool, optional): Whether to return the forward points, (Forward Rate - Spot
                Rate) * 10,000, instead of the forward rates. Defaults to False.
            date (str | None, optional): The day (YYYY-MM-DD). Defaults to None, the end date.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per country the Spot Rate and the forward rate (or the forward points)
            of each tenor, in units of the country's currency per unit of the base currency.

        Notes:
        - The market quotes forward points of yen pairs per 100 rather than 10,000; the points
          here are per 10,000 for every pair.
        - The spot rate is the BIS rate of the day, while EIOPA's curves are those of the last
          month end; the government curves are those of the same day.
        - Covered interest parity leaves out the cross-currency basis, which makes US dollar
          forwards trade some basis points of carry away from the parity rate.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-11-01', end_date='2026-11-20')

        fixedincome.get_fx_forward_rates()
        ```

        Which returns:

        | Country        |   Spot Rate |       1M |       3M |       6M |       1Y |       2Y |       5Y |      10Y |
        |:---------------|------------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
        | Euro Area      |      0.91   |   0.9091 |   0.9073 |   0.9046 |   0.8993 |   0.8896 |   0.8609 |   0.8101 |
        | United Kingdom |      0.7581 |   0.7582 |   0.7584 |   0.7587 |   0.7593 |   0.7621 |   0.7694 |   0.7826 |
        | Japan          |    159.296  | 158.945  | 158.247  | 157.205  | 155.141  | 151.547  | 143.133  | 133.421  |
        | Switzerland    |      0.8402 |   0.8373 |   0.8315 |   0.8229 |   0.806  |   0.7752 |   0.6944 |   0.5763 |
        | Canada         |      1.4291 |   1.4275 |   1.4243 |   1.4196 |   1.4103 |   1.3957 |   1.3589 |   1.3098 |
        | Australia      |      1.4848 |   1.4857 |   1.4875 |   1.4903 |   1.4958 |   1.5011 |   1.5256 |   1.5828 |
        """
        end_date = date or self._end_date
        requested = helpers.split_values(countries) or list(DEFAULT_FX_COUNTRIES)
        tenor_years = parse_maturities(tenors, DEFAULT_FX_TENORS)

        exchange_rates = bis_model.get_exchange_rates(
            "daily",
            (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime(
                "%Y-%m-%d"
            ),
            end_date,
        )
        if exchange_rates.empty:
            # The source has logged why (it could not be reached, or published nothing).
            raise ValueError(
                f"The BIS returned no exchange rates in the 30 days before {end_date}, so there "
                "is no spot rate to start the forwards from."
            )
        exchange_rates = exchange_rates.loc[: pd.Period(end_date, "D")]

        def spot_per_dollar(country: str) -> float:
            check_option("country", country, exchange_rates)
            rates = exchange_rates[country].dropna()
            if rates.empty:
                raise ValueError(
                    f"The BIS has no exchange rate for {country} in the 30 days before {end_date}."
                )
            return float(rates.iloc[-1])

        base_spot = spot_per_dollar(base_country)
        base_curve = self._swap_zero_curve(curve, base_country, end_date)

        forwards = {}
        for country in requested:
            try:
                spot = spot_per_dollar(country) / base_spot
                zero_curve = self._swap_zero_curve(curve, country, end_date)
            except ValueError as error:
                logger.warning(
                    "No FX forward rates for %s: %s",
                    country,
                    logger_model.redact(error),
                )
                continue

            row = {"Spot Rate": spot}
            for years in tenor_years:
                forward_rate, _ = forward_model.get_fx_forward_rate(
                    spot_rate=spot,
                    domestic_rate=float(np.interp(years, zero_curve.index, zero_curve)),
                    foreign_rate=float(np.interp(years, base_curve.index, base_curve)),
                    years_to_maturity=years,
                )
                row[maturity_label(maturity_key(years))] = (
                    (forward_rate - spot) * 10_000 if points else forward_rate
                )
            forwards[country] = row

        if not forwards:
            raise ValueError(
                f"No FX forward rates could be calculated for {', '.join(requested)}."
            )

        fx_forwards = pd.DataFrame(forwards).T
        fx_forwards.index.name = "Country"

        return apply_rounding(
            fx_forwards, rounding if rounding is not None else self._rounding
        )
