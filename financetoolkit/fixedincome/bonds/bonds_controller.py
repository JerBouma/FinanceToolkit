"""Bond Analytics Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit import (
    helpers,
)
from financetoolkit.fixedincome import helpers as fixedincome_helpers
from financetoolkit.fixedincome.base.base_controller import _maturities
from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.markets import bond_data_model
from financetoolkit.fixedincome.markets.markets_controller import MarketsFixedIncome
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = logger_model.get_logger()


def _benchmark_treasuries(treasuries: pd.DataFrame) -> pd.DataFrame:
    """
    The benchmark Treasuries: per maturity of TREASURY_BENCHMARKS the note (up to ten years)
    or bond maturing closest to that many years after the price date.

    The latest auction of each maturity sells the issue maturing closest to that date; until
    its first price is published the issue a few months shorter is the closest one.

    Args:
        treasuries (pd.DataFrame): The Treasuries, see get_treasury_securities.

    Returns:
        pd.DataFrame: The benchmark Treasuries, from the shortest to the longest maturity.
    """
    maturities = pd.to_datetime(treasuries["Maturity"])
    price_date = pd.Timestamp(treasuries["Price Date"].iloc[0])
    benchmarks = []
    for years in TREASURY_BENCHMARKS:
        candidates = maturities[
            treasuries["Security Type"]
            == ("Note" if years <= 10 else "Bond")  # noqa: PLR2004
        ]
        if not candidates.empty:
            target = price_date + pd.DateOffset(years=years)
            benchmarks.append((candidates - target).abs().idxmin())

    return treasuries.loc[list(dict.fromkeys(benchmarks))]


def _bond_terms(bond: pd.Series) -> dict:
    """
    The terms `bond_model` takes for a bond: par of 100, coupons twice a year.

    Args:
        bond (pd.Series): A bond, see FixedIncome._select_bonds.

    Returns:
        dict: The par_value, coupon_rate, years_to_maturity and frequency.
    """
    return {
        "par_value": PAR_VALUE,
        "coupon_rate": float(bond["Coupon"]),
        "years_to_maturity": float(bond["Years to Maturity"]),
        "frequency": FREQUENCY,
    }


def _curve_measures(bond: pd.Series, terms: dict, spot_rates: pd.Series) -> dict:
    """
    The present value of a bond on a spot curve and its Z-spread over it. TIPS are priced
    on their inflation-adjusted principal, so they have neither on a nominal curve.

    Args:
        bond (pd.Series): A bond, see FixedIncome._select_bonds.
        terms (dict): The bond's terms, see _bond_terms.
        spot_rates (pd.Series): The spot rates by maturity in years, compounded at the
            coupon frequency.

    Returns:
        dict: The Present Value and Z-Spread.
    """
    if bond["Security Type"] == "TIPS":
        return {"Present Value": np.nan, "Z-Spread": np.nan}

    return {
        "Present Value": bond_model.get_bond_price_from_curve(
            **terms, spot_rates=spot_rates
        ),
        "Z-Spread": bond_model.get_z_spread(
            **terms, bond_price=float(bond["Price"]), spot_rates=spot_rates
        ),
    }


# The bond calculators price per 100 of par with coupons twice a year, as US Treasuries and
# corporate bonds pay them.
PAR_VALUE = 100

FREQUENCY = bond_data_model.FREQUENCY

# The maturities of the benchmark Treasuries, those the Treasury auctions notes (up to ten
# years) and bonds of.
TREASURY_BENCHMARKS = (2, 3, 5, 7, 10, 20, 30)

# What the bond calculators show of every bond before its measures.
BOND_COLUMNS = ["Issuer", "Security Type", "Coupon", "Maturity", "Price", "Price Date"]

# The measures of a bond at its yield to maturity, by column name.
YIELD_MEASURES = {
    "Macaulay's Duration": bond_model.get_macaulays_duration,
    "Modified Duration": bond_model.get_modified_duration,
    "Effective Duration": bond_model.get_effective_duration,
    "Dollar Duration": bond_model.get_dollar_duration,
    "DV01": bond_model.get_dv01,
    "Convexity": bond_model.get_convexity,
}

DURATION_TYPES = {
    "macaulay": "Macaulay's Duration",
    "modified": "Modified Duration",
    "effective": "Effective Duration",
    "dollar": "Dollar Duration",
}


@filter_shown_columns
@track_sources
class BondsFixedIncome(MarketsFixedIncome):
    """
    Bond analytics: present value, duration, yield to maturity, Z-spread, key rate durations, the Taylor price change, the bond equivalent yield and bond statistics.
    """

    def _select_bonds(
        self,
        bonds: list[str] | str | None,
        issuer: list[str] | str | None,
        etfs: list[str] | str | None,
        maturity_start: str | None,
        maturity_end: str | None,
        date: str | None,
    ) -> pd.DataFrame:
        """
        The bonds the bond calculators work on, with their terms and market price.

        Treasuries (a CUSIP starting with 912, or its US912 ISIN) come from
        `get_treasury_securities`, every other bond from `get_bond_universe`, which an issuer
        or a set of bond ETFs selects as well. Without any of these the benchmark Treasuries
        are used: per maturity of TREASURY_BENCHMARKS the note (up to ten years) or bond
        maturing closest to that many years after the price date, the issue the latest
        auction of that maturity sold or one that matures a few months earlier. With only a
        maturity range every fixed-coupon Treasury (bills, notes and bonds) maturing within
        it is used instead.

        Args:
            bonds (list[str] | str | None): CUSIPs or ISINs.
            issuer (list[str] | str | None): Part of the issuer names, see get_bond_universe.
            etfs (list[str] | str | None): The bond ETFs to read bonds from, see get_bond_universe.
            maturity_start (str | None): The earliest maturity (YYYY-MM-DD).
            maturity_end (str | None): The latest maturity (YYYY-MM-DD).
            date (str | None): The day of the Treasury prices, see get_treasury_securities.

        Returns:
            pd.DataFrame: One row per bond, indexed by CUSIP (Treasuries) or ISIN, with the
            Issuer, Security Type, Coupon, Maturity, Price, Price Date, Years to Maturity
            and Yield to Maturity, unrounded.
        """
        identifiers = [value.upper() for value in helpers.split_values(bonds)]
        treasury_ids = [
            cusip
            for value in identifiers
            if (cusip := fixedincome_helpers.treasury_cusip(value))
        ]
        other_ids = [
            value
            for value in identifiers
            if fixedincome_helpers.treasury_cusip(value) is None
        ]
        use_universe = bool(other_ids) or issuer is not None or etfs is not None

        selections = []
        if use_universe:
            universe = self.get_bond_universe(
                etfs=etfs,
                issuer=issuer,
                maturity_start=maturity_start,
                maturity_end=maturity_end,
                bonds=other_ids or None,
                rounding=10,
            )
            if not isinstance(universe, pd.DataFrame) or universe.empty:
                raise ValueError(
                    "No bond in the bond ETF holdings matches the bonds, issuer and "
                    "maturities given."
                )
            selections.append(universe.assign(**{"Security Type": "Corporate"}))

        if treasury_ids or not use_universe:
            treasuries = self.get_treasury_securities(
                date=date, bonds=treasury_ids or None, rounding=10
            )
            if not isinstance(treasuries, pd.DataFrame) or treasuries.empty:
                raise ValueError(
                    "No Treasury prices are available for the bonds and date given."
                )
            floating = treasuries.index[treasuries["Security Type"] == "FRN"]
            if treasury_ids and not floating.empty:
                logger.info(
                    "Left out %s: floating rate notes reset their coupon every week.",
                    ", ".join(floating),
                )
            treasuries = treasuries.drop(floating)
            maturities = pd.to_datetime(treasuries["Maturity"])
            if maturity_start is not None:
                treasuries = treasuries[maturities >= pd.Timestamp(maturity_start)]
            if maturity_end is not None:
                treasuries = treasuries[
                    pd.to_datetime(treasuries["Maturity"]) <= pd.Timestamp(maturity_end)
                ]
            if not treasury_ids:
                treasuries = treasuries[
                    treasuries["Security Type"].isin(["Bill", "Note", "Bond"])
                ]
                if maturity_start is None and maturity_end is None:
                    treasuries = _benchmark_treasuries(treasuries)
            if treasuries.empty:
                raise ValueError("No Treasury matches the bonds and maturities given.")
            selections.append(treasuries.assign(Issuer="United States Treasury"))

        selection = pd.concat(selections)
        selection.index.name = "Bond"
        selection["Coupon"] = selection["Coupon"].fillna(0.0)
        selection["Years to Maturity"] = [
            (
                # A bill pays no coupon and counts its days on a 365-day year, as the
                # Treasury's bond-equivalent yield does.
                (pd.Timestamp(maturity) - pd.Timestamp(price_date)).days / 365
                if security_type == "Bill"
                else bond_data_model.get_years_to_maturity(
                    pd.Timestamp(price_date).date(), pd.Timestamp(maturity).date()
                )
            )
            for security_type, maturity, price_date in zip(
                selection["Security Type"],
                selection["Maturity"],
                selection["Price Date"],
                strict=True,
            )
        ]

        return selection[
            [
                "Issuer",
                "Security Type",
                "Coupon",
                "Maturity",
                "Price",
                "Price Date",
                "Years to Maturity",
                "Yield to Maturity",
            ]
        ]

    def _bond_spot_curves(self, selection: pd.DataFrame) -> dict[str, pd.Series]:
        """
        The Treasury zero-coupon curve of every price date of the bonds, compounded twice
        a year like the coupons, so that `bond_model` discounts with it directly.

        Args:
            selection (pd.DataFrame): The bonds, see _select_bonds.

        Returns:
            dict[str, pd.Series]: The spot rates by maturity in years, per price date.
        """
        curves = {}
        for price_date in selection["Price Date"].unique():
            zero_curve = self._latest_zero_curve("United States", str(price_date))
            curves[price_date] = FREQUENCY * ((1 + zero_curve) ** (1 / FREQUENCY) - 1)

        return curves

    @staticmethod
    def _bond_table(
        selection: pd.DataFrame, measures: pd.DataFrame, rounding: int | None
    ) -> pd.DataFrame:
        """
        The bonds' description followed by the measures calculated for them.

        Args:
            selection (pd.DataFrame): The bonds, see _select_bonds.
            measures (pd.DataFrame): The measures, with the same index.
            rounding (int | None): The number of decimals to round to.

        Returns:
            pd.DataFrame: The Issuer, Security Type, Coupon, Maturity, Price and Price Date,
            then the measures.
        """
        table = pd.concat([selection[BOND_COLUMNS], measures], axis=1)

        return apply_rounding(table, rounding)

    @handle_errors
    def collect_bond_statistics(
        self,
        bonds: list[str] | str | None = None,
        issuer: list[str] | str | None = None,
        etfs: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Collects the statistics of individual bonds from their market price: the yield to
        maturity, current yield, the Macaulay, modified, effective and dollar duration, DV01
        and convexity, plus the present value on the Treasury zero-coupon curve and the
        Z-spread over it.

        Treasuries come from the Treasury's daily FedInvest prices (see
        `get_treasury_securities`), corporate bonds from the holdings of bond ETFs (see
        `get_bond_universe`). Without bonds or an issuer the statistics are those of the
        benchmark Treasuries: the note or bond maturing closest to 2, 3, 5, 7, 10, 20 and
        30 years after the price date, which is the latest issue of that maturity or one
        that matures a few months earlier. With only a maturity range they are those of
        every bill, note and bond maturing within it.

        The fields are:

            - Years to Maturity: counted on the bond's actual coupon dates (actual/actual).
            - Accrued Interest: the coupon earned since the previous coupon date.
            - Yield to Maturity: the rate that discounts the coupons and principal to the
                price plus the accrued interest, compounded twice a year.
            - Current Yield: the annual coupon divided by the price.
            - Macaulay's Duration: the weighted average time to receive the cash flows.
            - Modified Duration: Macaulay's duration divided by 1 + YTM / 2.
            - Effective Duration: the price change of a 1% move in the yield up and down.
            - Dollar Duration: the modified duration times the full price, divided by 100.
            - DV01: the price change, per 100 of par, of a one basis point move in the yield.
            - Convexity: the second derivative of the price to the yield, divided by the price.
            - Present Value: the clean price of the cash flows discounted with the Treasury
                zero-coupon curve of the price date.
            - Z-Spread: the spread over that zero-coupon curve that prices the bond at its
                market price.

        The formula is as follows:

        - PV(t) = Cash Flow(t) / (1 + YTM / 2)^(2t), with coupons of Coupon / 2 per 100 of
          par and the principal repaid with the last coupon
        - Dirty Price = SUM(PV(t)) = Price + Accrued Interest, over the remaining payment
          times t in years
        - Current Yield = Coupon * 100 / Price
        - Macaulay's Duration = SUM(t * PV(t)) / Dirty Price
        - Modified Duration = Macaulay's Duration / (1 + YTM / 2)
        - Effective Duration = (Dirty Price(YTM - 0.01) - Dirty Price(YTM + 0.01)) / (2 * Dirty Price * 0.01)
        - Dollar Duration = Modified Duration * Dirty Price / 100
        - DV01 = (Price(YTM - 0.0001) - Price(YTM + 0.0001)) / 2
        - Convexity = (Dirty Price(YTM + 0.0001) + Dirty Price(YTM - 0.0001) - 2 * Dirty Price) / (Dirty Price * 0.0001^2)
        - Present Value = SUM(Cash Flow(t) / (1 + Spot Rate(t) / 2)^(2t)) - Accrued Interest
        - Z-Spread solves Price = SUM(Cash Flow(t) / (1 + (Spot Rate(t) + Z-Spread) / 2)^(2t)) - Accrued Interest

        Where Spot Rate(t) is the Treasury zero-coupon rate of `get_zero_coupon_curve`,
        restated from annual to semi-annual compounding, interpolated linearly between its
        maturities. Bills pay no coupon and their yield is the bond-equivalent yield (see
        `get_bond_equivalent_yield`). TIPS are priced on their inflation-adjusted principal,
        so their yield is a real yield and they have no present value or Z-spread on the
        nominal curve. Floating rate notes are left out.

        To calculate these statistics for terms of your own, use the functions of
        `financetoolkit.fixedincome.bonds.bond_model` (get_bond_price, get_yield_to_maturity,
        get_macaulays_duration, get_modified_duration, get_effective_duration,
        get_dollar_duration, get_dv01, get_convexity and get_z_spread).

        Also known as: bond data, fixed income statistics, bond analytics.

        Args:
            bonds (list[str] | str | None, optional): CUSIPs or ISINs, a list or a
                comma-separated string. Treasuries (CUSIPs starting with 912) come from
                FedInvest, other bonds from the bond ETF holdings. Defaults to None.
            issuer (list[str] | str | None, optional): Part of an issuer's name, such as
                "Apple", to select its bonds from the bond ETF holdings. Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs whose holdings to select
                bonds from, see `get_bond_universe`. Defaults to None, which reads LQD, IGSB
                and USHY when bonds other than Treasuries or an issuer are given.
            maturity_start (str | None, optional): The earliest maturity (YYYY-MM-DD).
                Defaults to None.
            maturity_end (str | None, optional): The latest maturity (YYYY-MM-DD). Defaults
                to None.
            date (str | None, optional): The day of the Treasury prices (YYYY-MM-DD). The
                bond ETF holdings are always the latest. Defaults to None, the latest day
                with prices.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond, indexed by CUSIP (Treasuries) or ISIN, with the
            Issuer, Security Type, Coupon, Maturity, Price, Price Date and the statistics.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        statistics = fixedincome.collect_bond_statistics(date="2026-11-05")

        statistics.iloc[[0, 2, 4, 6]].loc[:, "Maturity":].T
        ```

        Which returns:

        |                     | 91282CRU7   | 91282CRT0   | 91282CRF0   | 912810UW6   |
        |:--------------------|:------------|:------------|:------------|:------------|
        | Maturity            | 2028-10-31  | 2031-10-31  | 2036-08-15  | 2056-08-15  |
        | Price               | 100.3125    | 100.6875    | 98.4375     | 98.125      |
        | Price Date          | 2026-11-05  | 2026-11-05  | 2026-11-05  | 2026-11-05  |
        | Years to Maturity   | 1.9862      | 4.9862      | 9.7772      | 29.7772     |
        | Accrued Interest    | 0.0639      | 0.0656      | 1.0306      | 1.142       |
        | Yield to Maturity   | 0.0446      | 0.0459      | 0.0483      | 0.0525      |
        | Current Yield       | 0.0461      | 0.0472      | 0.047       | 0.0522      |
        | Macaulay's Duration | 1.9195      | 4.4973      | 7.8771      | 15.2706     |
        | Modified Duration   | 1.8777      | 4.3963      | 7.6914      | 14.8801     |
        | Effective Duration  | 1.8779      | 4.3985      | 7.7036      | 15.0246     |
        | Dollar Duration     | 1.8847      | 4.4294      | 7.6505      | 14.7715     |
        | DV01                | 0.0188      | 0.0443      | 0.0765      | 0.1477      |
        | Convexity           | 4.5145      | 22.7527     | 71.6857     | 333.2043    |
        | Present Value       | 100.3147    | 100.7082    | 98.4008     | 98.028      |
        | Z-Spread            | 0.0         | 0.0         | 0.0         | -0.0001     |
        """
        selection = self._select_bonds(
            bonds, issuer, etfs, maturity_start, maturity_end, date
        )
        curves = self._bond_spot_curves(selection)

        statistics = {}
        for bond, row in selection.iterrows():
            terms = _bond_terms(row)
            yield_to_maturity = row["Yield to Maturity"]
            statistics[bond] = {
                "Years to Maturity": terms["years_to_maturity"],
                "Accrued Interest": bond_model.get_accrued_interest(**terms),
                "Yield to Maturity": yield_to_maturity,
                "Current Yield": bond_model.get_current_yield(
                    par_value=PAR_VALUE,
                    coupon_rate=terms["coupon_rate"],
                    bond_price=row["Price"],
                ),
                **{
                    name: function(**terms, yield_to_maturity=yield_to_maturity)
                    for name, function in YIELD_MEASURES.items()
                },
                **_curve_measures(row, terms, curves[row["Price Date"]]),
            }

        return self._bond_table(
            selection,
            pd.DataFrame.from_dict(statistics, orient="index"),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_present_value(
        self,
        bonds: list[str] | str | None = None,
        issuer: list[str] | str | None = None,
        etfs: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the present value of individual bonds: their coupons and principal
        discounted with the Treasury zero-coupon curve of the price date, next to the price
        they trade at. For a Treasury the two differ by how far the issue trades off the
        curve; for a corporate bond the present value is the price it would have without
        any credit or liquidity risk, and the gap with its price is what the Z-spread
        (`get_z_spread`) measures.

        The formula is as follows:

        - Dirty Price = SUM(C / (1 + s(t_i) / 2)^(2 t_i)) + F / (1 + s(t_n) / 2)^(2 t_n), over the
          remaining coupons i = 1, ..., n
        - Accrued Interest = C * a
        - Present Value = Dirty Price - Accrued Interest, the clean price

        where:

        - C = Coupon payment per half year, 100 * Coupon / 2
        - s(t) = Treasury zero-coupon rate at t years (`get_zero_coupon_curve`), restated from
          annual to semi-annual compounding and interpolated linearly between its maturities
        - a = Elapsed fraction of the current coupon period, 0 on a coupon date
        - t_i = Time of the i-th cash flow in years, on the bond's actual coupon dates
        - F = Face value of 100

        The bonds are selected as in `collect_bond_statistics`: Treasuries by CUSIP from
        FedInvest, other bonds from the bond ETF holdings and, without bonds or an issuer,
        the benchmark Treasuries. TIPS have no present value on the nominal curve. To price
        terms of your own, use `bond_model.get_bond_price` (at one yield) or
        `bond_model.get_bond_price_from_curve` (on a curve).

        Also known as: PV, bond pricing, discounted cash flows, theoretical price.

        Args:
            bonds (list[str] | str | None, optional): CUSIPs or ISINs, see
                `collect_bond_statistics`. Defaults to None.
            issuer (list[str] | str | None, optional): Part of an issuer's name, such as
                "Apple". Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs to select bonds from,
                see `get_bond_universe`. Defaults to None.
            maturity_start (str | None, optional): The earliest maturity (YYYY-MM-DD).
                Defaults to None.
            maturity_end (str | None, optional): The latest maturity (YYYY-MM-DD). Defaults
                to None.
            date (str | None, optional): The day of the Treasury prices (YYYY-MM-DD).
                Defaults to None, the latest day with prices.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond with the Issuer, Security Type, Coupon, Maturity,
            Price, Price Date and Present Value (per 100 of par).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        present_values = fixedincome.get_present_value(date="2026-11-05")

        present_values[["Coupon", "Maturity", "Price", "Present Value"]]
        ```

        Which returns:

        |           |   Coupon | Maturity   |    Price |   Present Value |
        |:----------|---------:|:-----------|---------:|----------------:|
        | 91282CRU7 |   0.0462 | 2028-10-31 | 100.312  |        100.315  |
        | 91282CLR0 |   0.0412 | 2029-10-31 |  98.875  |         98.8811 |
        | 91282CRT0 |   0.0475 | 2031-10-31 | 100.688  |        100.708  |
        | 91282CRV5 |   0.0488 | 2033-10-31 | 101.062  |        101.039  |
        | 91282CRF0 |   0.0462 | 2036-08-15 |  98.4375 |         98.4008 |
        | 912810RU4 |   0.0288 | 2046-11-15 |  70.0312 |         69.9175 |
        | 912810UW6 |   0.0512 | 2056-08-15 |  98.125  |         98.028  |
        """
        selection = self._select_bonds(
            bonds, issuer, etfs, maturity_start, maturity_end, date
        )
        curves = self._bond_spot_curves(selection)
        present_values = {
            bond: _curve_measures(row, _bond_terms(row), curves[row["Price Date"]])[
                "Present Value"
            ]
            for bond, row in selection.iterrows()
        }

        return self._bond_table(
            selection,
            pd.Series(present_values, name="Present Value", dtype=float),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_duration(
        self,
        duration_type: str = "modified",
        bonds: list[str] | str | None = None,
        issuer: list[str] | str | None = None,
        etfs: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the duration of individual bonds at their yield to maturity. It has the
        option to calculate the following type of bond durations:

        - Macaulay's Duration: The weighted average time to receive the bond's cash flows.
        - Modified Duration: The Macaulay's duration divided by 1 plus the per-period yield (YTM / 2).
        - Effective Duration: The percentage change in the bond price for a 1% change in the yield to maturity.
        - Dollar Duration: The modified duration multiplied by the bond price, divided by 100.

        These duration measures can be used to estimate the sensitivity of a bond's price to changes in interest rates as well as
        to compare the risk of different bonds. The modified duration is particularly useful for estimating the percentage change
        in the bond price for a 1% change in the yield to maturity. Note that it is a percentage sensitivity and therefore not the
        same as the dollar duration, the price value of a basis point (PVBP) or the dollar value of a 0.01% change (DV01), which are
        all expressed as a currency amount instead. The DV01 is part of `collect_bond_statistics`.

        The formula is as follows (the default is the modified duration):

        - PV(t) = Cash Flow(t) / (1 + YTM / 2)^(2t), with coupons of Coupon / 2 per 100 of par and the
          principal repaid with the last coupon
        - Dirty Price = SUM(PV(t)), over the remaining payment times t in years on the actual coupon dates
        - Macaulay's Duration = SUM(t * PV(t)) / Dirty Price
        - Modified Duration = Macaulay's Duration / (1 + YTM / 2)
        - Effective Duration = (Dirty Price(YTM - 0.01) - Dirty Price(YTM + 0.01)) / (2 * Dirty Price * 0.01)
        - Dollar Duration = Modified Duration * Dirty Price / 100

        Where YTM is the bond's yield to maturity at its market price, compounded twice a year (see
        `get_yield_to_maturity`). The bonds are selected as in `collect_bond_statistics`.

        Modified duration is the first estimate of interest rate risk: a modified duration of 7 means the price falls
        by about 7% if the yield rises by one percentage point. The estimate worsens for larger moves, which convexity
        corrects (see `get_taylor_price_change`), and it assumes the cash flows are fixed. For callable bonds the
        relevant measure is an effective duration from a model that values the embedded option, which this method does
        not include. For terms of your own, use the duration functions of `bond_model`.

        Also known as: Macaulay duration, modified duration, bond price sensitivity.

        Args:
            duration_type (str, optional): The duration: "modified", "macaulay", "effective" or
                "dollar". Defaults to "modified".
            bonds (list[str] | str | None, optional): CUSIPs or ISINs, see
                `collect_bond_statistics`. Defaults to None.
            issuer (list[str] | str | None, optional): Part of an issuer's name, such as
                "Apple". Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs to select bonds from,
                see `get_bond_universe`. Defaults to None.
            maturity_start (str | None, optional): The earliest maturity (YYYY-MM-DD).
                Defaults to None.
            maturity_end (str | None, optional): The latest maturity (YYYY-MM-DD). Defaults
                to None.
            date (str | None, optional): The day of the Treasury prices (YYYY-MM-DD).
                Defaults to None, the latest day with prices.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond with the Issuer, Security Type, Coupon, Maturity,
            Price, Price Date and the duration (in years, or per 100 of par for the dollar
            duration).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        durations = fixedincome.get_duration(date="2026-11-05")

        durations[["Security Type", "Coupon", "Maturity", "Price", "Modified Duration"]]
        ```

        Which returns:

        |           | Security Type   |   Coupon | Maturity   |    Price |   Modified Duration |
        |:----------|:----------------|---------:|:-----------|---------:|--------------------:|
        | 91282CRU7 | Note            |   0.0462 | 2028-10-31 | 100.312  |              1.8777 |
        | 91282CLR0 | Note            |   0.0412 | 2029-10-31 |  98.875  |              2.7748 |
        | 91282CRT0 | Note            |   0.0475 | 2031-10-31 | 100.688  |              4.3963 |
        | 91282CRV5 | Note            |   0.0488 | 2033-10-31 | 101.062  |              5.8687 |
        | 91282CRF0 | Note            |   0.0462 | 2036-08-15 |  98.4375 |              7.6914 |
        | 912810RU4 | Bond            |   0.0288 | 2046-11-15 |  70.0312 |             13.6289 |
        | 912810UW6 | Bond            |   0.0512 | 2056-08-15 |  98.125  |             14.8801 |
        """
        name = DURATION_TYPES.get(duration_type.lower())
        if name is None:
            raise ValueError(
                "Please input a valid duration type ('macaulay', 'modified', 'effective' or 'dollar')"
            )
        selection = self._select_bonds(
            bonds, issuer, etfs, maturity_start, maturity_end, date
        )
        durations = {
            bond: YIELD_MEASURES[name](
                **_bond_terms(row), yield_to_maturity=row["Yield to Maturity"]
            )
            for bond, row in selection.iterrows()
        }

        return self._bond_table(
            selection,
            pd.Series(durations, name=name, dtype=float),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_yield_to_maturity(
        self,
        bonds: list[str] | str | None = None,
        issuer: list[str] | str | None = None,
        etfs: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the yield to maturity of individual bonds from their market price. The yield to maturity is the
        internal rate of return of the bond, the discount rate that equates the present value of the bond's cash flows
        to its price.

        The formula is as follows, solved for YTM:

        - Price + C * a = SUM(C / (1 + YTM / 2)^(2 t_i)) + F / (1 + YTM / 2)^(2 t_n), over the remaining coupons
          i = 1, ..., n

        where:

        - C = Coupon payment per half year, 100 * Coupon / 2
        - a = Elapsed fraction of the current coupon period, in actual days (actual/actual), 0 on a coupon date
        - t_i = Time of the i-th cash flow in years, on the bond's actual coupon dates
        - F = Face value of 100

        The yield is a nominal annual rate compounded twice a year (bond-equivalent), as the Treasury and US corporate
        bond markets quote it, solved with the secant method. For a bill it is the bond-equivalent yield of its discount
        price (see `get_bond_equivalent_yield`), for TIPS the real yield. The bonds are selected as in
        `collect_bond_statistics`. For terms of your own, use `bond_model.get_yield_to_maturity`.

        The yield to maturity is the return of a bond bought at this price and held to maturity, if every coupon is reinvested
        at the same yield and the issuer pays in full. It is the standard way to quote and compare bonds, but it is one average
        rate for all cash flows: bonds with the same maturity but different coupons have different yields even when both are
        priced in line with the same spot curve, which the Z-spread (`get_z_spread`) accounts for. For callable bonds the
        yield to worst is quoted instead.

        Also known as: YTM, bond return to maturity, redemption yield.

        Args:
            bonds (list[str] | str | None, optional): CUSIPs or ISINs, see
                `collect_bond_statistics`. Defaults to None.
            issuer (list[str] | str | None, optional): Part of an issuer's name, such as
                "Apple". Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs to select bonds from,
                see `get_bond_universe`. Defaults to None.
            maturity_start (str | None, optional): The earliest maturity (YYYY-MM-DD).
                Defaults to None.
            maturity_end (str | None, optional): The latest maturity (YYYY-MM-DD). Defaults
                to None.
            date (str | None, optional): The day of the Treasury prices (YYYY-MM-DD).
                Defaults to None, the latest day with prices.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond with the Issuer, Security Type, Coupon, Maturity,
            Price, Price Date and Yield to Maturity (decimal).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        yields = fixedincome.get_yield_to_maturity(date="2026-11-05")

        yields[["Security Type", "Coupon", "Maturity", "Price", "Yield to Maturity"]]
        ```

        Which returns:

        | Bond      | Security Type   |   Coupon | Maturity   |    Price |   Yield to Maturity |
        |:----------|:----------------|---------:|:-----------|---------:|--------------------:|
        | 91282CRU7 | Note            |   0.0462 | 2028-10-31 | 100.312  |              0.0446 |
        | 91282CLR0 | Note            |   0.0412 | 2029-10-31 |  98.875  |              0.0453 |
        | 91282CRT0 | Note            |   0.0475 | 2031-10-31 | 100.688  |              0.0459 |
        | 91282CRV5 | Note            |   0.0488 | 2033-10-31 | 101.062  |              0.0469 |
        | 91282CRF0 | Note            |   0.0462 | 2036-08-15 |  98.4375 |              0.0483 |
        | 912810RU4 | Bond            |   0.0288 | 2046-11-15 |  70.0312 |              0.0533 |
        | 912810UW6 | Bond            |   0.0512 | 2056-08-15 |  98.125  |              0.0525 |
        """
        selection = self._select_bonds(
            bonds, issuer, etfs, maturity_start, maturity_end, date
        )

        return self._bond_table(
            selection,
            selection["Yield to Maturity"],
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_z_spread(
        self,
        bonds: list[str] | str | None = None,
        issuer: list[str] | str | None = None,
        etfs: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the zero-volatility spread (Z-spread) of individual bonds over the Treasury
        zero-coupon curve. The Z-spread is the constant spread that, when added uniformly to every
        point of the curve, makes the present value of the bond's cash flows equal to its price.

        Unlike a simple yield spread (the bond's yield to maturity minus a Treasury yield of the
        same maturity, the G-spread of `get_bond_universe`), the Z-spread is measured against the
        entire curve rather than a single point, which makes it a more accurate measure of the
        compensation an investor receives for a bond's credit and liquidity risk. For a Treasury it
        shows how far the issue trades off the curve, typically a few basis points.

        The formula is as follows, solved for the Z-Spread with the secant method:

        - Price = SUM(Cash Flow(t) / (1 + (Spot Rate(t) + Z-Spread) / 2)^(2t)) - Accrued Interest

        Where each coupon is 100 * Coupon / 2, the principal is repaid with the last coupon, t is
        each remaining payment time in years on the bond's actual coupon dates and Spot Rate(t) is
        the Treasury zero-coupon rate of the price date (`get_zero_coupon_curve`), restated from
        annual to semi-annual compounding, interpolated linearly between its maturities and held
        flat beyond its first and last maturity.

        The bonds are selected as in `collect_bond_statistics`. TIPS have no Z-spread on the nominal
        curve. For a bond with an embedded option the Z-spread includes the option's value, which an
        option-adjusted spread removes: the ICE BofA option-adjusted spreads of bond indices are in
        `get_ice_bofa_option_adjusted_spread`, and `short_rate_model.get_option_adjusted_spread`
        calculates one for a call schedule of your own. For terms of your own, use
        `bond_model.get_z_spread`.

        Also known as: zero-volatility spread, static spread, credit spread.

        Args:
            bonds (list[str] | str | None, optional): CUSIPs or ISINs, see
                `collect_bond_statistics`. Defaults to None.
            issuer (list[str] | str | None, optional): Part of an issuer's name, such as
                "Apple". Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs to select bonds from,
                see `get_bond_universe`. Defaults to None.
            maturity_start (str | None, optional): The earliest maturity (YYYY-MM-DD).
                Defaults to None.
            maturity_end (str | None, optional): The latest maturity (YYYY-MM-DD). Defaults
                to None.
            date (str | None, optional): The day of the Treasury prices (YYYY-MM-DD).
                Defaults to None, the latest day with prices.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond with the Issuer, Security Type, Coupon, Maturity,
            Price, Price Date and Z-Spread (decimal, compounded twice a year).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(api_key="FINANCIAL_MODELING_PREP_KEY")

        z_spreads = fixedincome.get_z_spread(
            issuer="Apple", maturity_start="2030-01-01", maturity_end="2035-12-31"
        )

        z_spreads[["Issuer", "Coupon", "Maturity", "Price", "Price Date", "Z-Spread"]]
        ```

        Which returns:

        |              | Issuer    |   Coupon | Maturity   |   Price | Price Date   |   Z-Spread |
        |:-------------|:----------|---------:|:-----------|--------:|:-------------|-----------:|
        | US037833EU05 | APPLE INC |   0.0415 | 2030-05-10 |   97.95 | 2026-11-05   |     0.0024 |
        | US037833DU14 | APPLE INC |   0.0165 | 2030-05-11 |   89.69 | 2026-11-05   |     0.0033 |
        | US037833EZ91 | APPLE INC |   0.042  | 2030-05-12 |   98.25 | 2026-11-05   |     0.002  |
        | US037833DY36 | APPLE INC |   0.0125 | 2030-08-20 |   87.72 | 2026-11-05   |     0.0027 |
        | US037833ED89 | APPLE INC |   0.0165 | 2031-02-08 |   87.78 | 2026-11-05   |     0.0029 |
        | US037833EJ59 | APPLE INC |   0.017  | 2031-08-05 |   86.74 | 2026-11-05   |     0.0027 |
        | US037833FG02 | APPLE INC |   0.049  | 2031-11-09 |   99.83 | 2026-11-05   |     0.0035 |
        | US037833FA32 | APPLE INC |   0.045  | 2032-05-12 |   98.35 | 2026-11-05   |     0.0022 |
        | US037833EP10 | APPLE INC |   0.0335 | 2032-08-08 |   92.53 | 2026-11-05   |     0.0022 |
        | US037833EV87 | APPLE INC |   0.043  | 2033-05-10 |   95.84 | 2026-11-05   |     0.0038 |
        | US037833FH84 | APPLE INC |   0.051  | 2033-11-09 |   99.77 | 2026-11-05   |     0.0044 |
        | US037833FB15 | APPLE INC |   0.0475 | 2035-05-12 |   97.91 | 2026-11-05   |     0.0028 |
        """
        selection = self._select_bonds(
            bonds, issuer, etfs, maturity_start, maturity_end, date
        )
        curves = self._bond_spot_curves(selection)
        z_spreads = {
            bond: _curve_measures(row, _bond_terms(row), curves[row["Price Date"]])[
                "Z-Spread"
            ]
            for bond, row in selection.iterrows()
        }

        return self._bond_table(
            selection,
            pd.Series(z_spreads, name="Z-Spread", dtype=float),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_key_rate_duration(
        self,
        key_rate_maturity: float | str | list | None = None,
        yield_change: float = 0.0001,
        bonds: list[str] | str | None = None,
        issuer: list[str] | str | None = None,
        etfs: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the key rate durations of individual bonds on the Treasury zero-coupon curve:
        the price sensitivity to a shock at a single maturity ("key rate") of the curve while every
        other point is held fixed. Whereas `get_duration` with `duration_type='effective'` assumes the
        entire curve shifts in parallel, key rate durations show which segment of the curve a bond's
        price is exposed to. Because cash flows are discounted using linear interpolation between the
        curve's maturities, a shock at an inner maturity tapers off towards its neighbours and has no
        effect beyond them. The curve is held flat beyond its first and last maturities, so a shock at
        the first maturity moves every cash flow paid before it in full (and a shock at the last
        maturity every cash flow paid after it).

        Summing the key rate durations across every maturity of the curve approximately reproduces the
        bond's effective (parallel-shift) duration on the curve. A barbell and a bullet with the same
        duration have very different key rate profiles and react differently when the curve steepens
        or flattens, which is why key rate durations are hedged with instruments at each key maturity.
        The profile also depends on which maturities the curve contains: the same bond shows different
        key rate durations on a curve with more or fewer points.

        The formula is as follows:

        - Dirty Price = SUM(Cash Flow(t) / (1 + Spot Rate(t) / 2)^(2t))
        - Key Rate Duration = -(Dirty Price(Key Rate + Δy) - Dirty Price(Key Rate - Δy)) / (2 * Dirty Price * Δy)

        Where each coupon is 100 * Coupon / 2, the principal is repaid with the last coupon, t is each
        remaining payment time in years on the bond's actual coupon dates, Spot Rate(t) is the Treasury
        zero-coupon rate of the price date (`get_zero_coupon_curve`), restated from annual to
        semi-annual compounding and interpolated linearly between its maturities, and Δy is the shock
        applied to the key rate only (1 basis point by default).

        The bonds are selected as in `collect_bond_statistics`; TIPS have no key rate durations on the
        nominal curve. For terms or a curve of your own, use `bond_model.get_key_rate_duration`.

        Also known as: partial duration, rate-specific duration, curve risk.

        Args:
            key_rate_maturity (float | str | list | None, optional): The maturities of the Treasury
                curve to shock, in years or as labels such as '2Y' (one of the curve's maturities:
                1M, 1.5M, 2M, 3M, 4M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y and 30Y). Defaults to None,
                which shocks every maturity of the curve.
            yield_change (float, optional): The shock to each key rate (decimal). Defaults to 0.0001.
            bonds (list[str] | str | None, optional): CUSIPs or ISINs, see
                `collect_bond_statistics`. Defaults to None.
            issuer (list[str] | str | None, optional): Part of an issuer's name, such as
                "Apple". Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs to select bonds from,
                see `get_bond_universe`. Defaults to None.
            maturity_start (str | None, optional): The earliest maturity (YYYY-MM-DD).
                Defaults to None.
            maturity_end (str | None, optional): The latest maturity (YYYY-MM-DD). Defaults
                to None.
            date (str | None, optional): The day of the Treasury prices (YYYY-MM-DD).
                Defaults to None, the latest day with prices.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond with the Issuer, Security Type, Coupon, Maturity,
            Price, Price Date and a key rate duration per key rate maturity (in years).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        key_rates = fixedincome.get_key_rate_duration(
            key_rate_maturity=["2Y", "5Y", "10Y", "30Y"], date="2026-11-05"
        )

        key_rates[["Coupon", "Maturity", "2Y", "5Y", "10Y", "30Y"]]
        ```

        Which returns:

        |           |   Coupon | Maturity   |     2Y |     5Y |    10Y |    30Y |
        |:----------|---------:|:-----------|-------:|-------:|-------:|-------:|
        | 91282CRU7 |   0.0462 | 2028-10-31 | 1.8043 | 0      | 0      | 0      |
        | 91282CLR0 |   0.0412 | 2029-10-31 | 0.1101 | 0      | 0      | 0      |
        | 91282CRT0 |   0.0475 | 2031-10-31 | 0.0842 | 4.0372 | 0      | 0      |
        | 91282CRV5 |   0.0488 | 2033-10-31 | 0.0861 | 0.4068 | 0      | 0      |
        | 91282CRF0 |   0.0462 | 2036-08-15 | 0.083  | 0.3598 | 5.9559 | 0      |
        | 912810RU4 |   0.0288 | 2046-11-15 | 0.072  | 0.3122 | 1.6649 | 0.0255 |
        | 912810UW6 |   0.0512 | 2056-08-15 | 0.0922 | 0.3998 | 2.1327 | 7.5183 |
        """
        selection = self._select_bonds(
            bonds, issuer, etfs, maturity_start, maturity_end, date
        )
        curves = self._bond_spot_curves(selection)

        # The maturities of the curve to shock, all of them when none are given.
        key_rates = (
            None if key_rate_maturity is None else _maturities(key_rate_maturity, [])
        )

        durations: dict[str, dict] = {}
        for bond, row in selection.iterrows():
            curve = curves[row["Price Date"]]
            durations[bond] = {
                fixedincome_helpers.maturity_label(key_rate): (
                    np.nan
                    if row["Security Type"] == "TIPS"
                    else bond_model.get_key_rate_duration(
                        **_bond_terms(row),
                        spot_rates=curve,
                        key_rate_maturity=key_rate,
                        yield_change=yield_change,
                    )
                )
                for key_rate in (curve.index if key_rates is None else key_rates)
            }

        return self._bond_table(
            selection,
            pd.DataFrame.from_dict(durations, orient="index"),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_taylor_price_change(
        self,
        yield_change: float | None = None,
        bonds: list[str] | str | None = None,
        issuer: list[str] | str | None = None,
        etfs: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        date: str | None = None,
        from_date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Estimates the percentage change in the price of individual bonds for a change in their
        yield, using a second-order Taylor series expansion that combines modified duration and
        convexity. With a from_date, the yield change is each bond's own change in yield from that
        day to the price date, and the realised change of its market price sits next to the
        estimate.

        Modified duration alone only captures the first-order (linear) relationship between a
        bond's price and its yield, which understates the price increase for a yield decrease and
        overstates the price decrease for a yield increase because the true price-yield
        relationship is curved (convex), not linear. Adding a convexity term corrects for this and
        produces a substantially more accurate estimate, especially for larger yield changes.

        The formula is as follows:

        - Modified Duration = SUM(t * PV(t)) / Dirty Price / (1 + YTM / 2), see `get_duration`
        - Convexity = (Dirty Price(YTM + 0.0001) + Dirty Price(YTM - 0.0001) - 2 * Dirty Price) / (Dirty Price *
          0.0001^2)
        - %ΔPrice ≈ -Modified Duration * Δy + 0.5 * Convexity * Δy^2

        Where YTM is the bond's yield to maturity at its market price, compounded twice a year (see
        `get_yield_to_maturity`), Δy is the yield change (0.01 by default) and %ΔPrice is the change of
        the full (dirty) price as a decimal. The bonds are selected as in `collect_bond_statistics`. For
        terms of your own, use `bond_model.get_taylor_price_change`.

        With a from_date, the Treasuries are selected on from_date (so the benchmark Treasuries are those
        that already traded then) and priced again on the price date (`date`):

        - Yield Change = YTM(date) - YTM(from_date)
        - Price Change = the Taylor estimate above, with the modified duration, convexity and YTM of
          from_date and Δy = Yield Change
        - Realised Price Change = Price(date) / Price(from_date) - 1, on the clean FedInvest prices

        The Taylor estimate holds the time to maturity fixed, while the realised change also includes
        the pull of the price towards par over the period and is measured on the clean price, so the two
        differ slightly even for a small yield change; over a few days the difference is mostly the
        approximation itself. Only Treasuries have prices on earlier days, so other bonds are left out.

        Also known as: duration-convexity approximation, second-order price approximation, realised
        bond price change.

        Args:
            yield_change (float | None, optional): The change in the yield to maturity (decimal),
                negative for a fall. Defaults to None, which is 0.01, or each bond's own yield change
                since from_date when that is given.
            bonds (list[str] | str | None, optional): CUSIPs or ISINs, see
                `collect_bond_statistics`. Defaults to None.
            issuer (list[str] | str | None, optional): Part of an issuer's name, such as
                "Apple". Defaults to None.
            etfs (list[str] | str | None, optional): The bond ETFs to select bonds from,
                see `get_bond_universe`. Defaults to None.
            maturity_start (str | None, optional): The earliest maturity (YYYY-MM-DD).
                Defaults to None.
            maturity_end (str | None, optional): The latest maturity (YYYY-MM-DD). Defaults
                to None.
            date (str | None, optional): The day of the Treasury prices (YYYY-MM-DD); a weekend or
                holiday uses the business day before it. Defaults to None, the latest day with prices.
            from_date (str | None, optional): The day (YYYY-MM-DD) to measure each Treasury's yield
                and price change from, up to `date`; a weekend or holiday uses the business day before
                it. Defaults to None, which estimates the price change for yield_change instead.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond with the Issuer, Security Type, Coupon, Maturity,
            Price, Price Date and Price Change (decimal); with a from_date the From Price and
            From Date before the Price and Price Date, followed by the Yield Change, Price Change
            and Realised Price Change, all decimals.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        price_changes = fixedincome.get_taylor_price_change(yield_change=0.01, date="2026-11-05")

        price_changes[["Security Type", "Coupon", "Maturity", "Price", "Price Change"]]
        ```

        Which returns:

        |           | Security Type   |   Coupon | Maturity   |    Price |   Price Change |
        |:----------|:----------------|---------:|:-----------|---------:|---------------:|
        | 91282CRU7 | Note            |   0.0462 | 2028-10-31 | 100.312  |        -0.0186 |
        | 91282CLR0 | Note            |   0.0412 | 2029-10-31 |  98.875  |        -0.0273 |
        | 91282CRT0 | Note            |   0.0475 | 2031-10-31 | 100.688  |        -0.0428 |
        | 91282CRV5 | Note            |   0.0488 | 2033-10-31 | 101.062  |        -0.0567 |
        | 91282CRF0 | Note            |   0.0462 | 2036-08-15 |  98.4375 |        -0.0733 |
        | 912810RU4 | Bond            |   0.0288 | 2046-11-15 |  70.0312 |        -0.1242 |
        | 912810UW6 | Bond            |   0.0512 | 2056-08-15 |  98.125  |        -0.1321 |

        The price changes of the benchmark Treasuries from 28 October to 6 November 2026, each
        for its own yield change:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        price_changes = fixedincome.get_taylor_price_change(from_date="2026-10-28", date="2026-11-06")

        price_changes.loc[:, "Maturity":].drop(columns=["From Date", "Price Date"])
        ```

        Which returns:

        |           | Maturity   |   From Price |   Price |   Yield Change |   Price Change |   Realised Price Change |
        |:----------|:-----------|-------------:|--------:|---------------:|---------------:|------------------------:|
        | 91282CDF5 | 2028-10-31 |      93.6562 | 94.0938 |        -0.002  |         0.0039 |                  0.0047 |
        | 91282CLR0 | 2029-10-31 |      98.0938 | 98.7188 |        -0.0022 |         0.0061 |                  0.0064 |
        | 91282CLU3 | 2031-10-31 |      96.6562 | 97.6562 |        -0.0023 |         0.01   |                  0.0103 |
        | 91282CJJ1 | 2033-11-15 |      97.3438 | 98.5312 |        -0.002  |         0.0119 |                  0.0122 |
        | 91282CRF0 | 2036-08-15 |      96.7812 | 98.2188 |        -0.0019 |         0.0147 |                  0.0149 |
        | 912810RU4 | 2046-11-15 |      68.6875 | 69.9062 |        -0.0012 |         0.0171 |                  0.0177 |
        | 912810UW6 | 2056-08-15 |      96.375  | 97.9688 |        -0.0011 |         0.0164 |                  0.0165 |
        """
        if yield_change is not None and from_date is not None:
            raise ValueError(
                "Pass a yield_change to estimate the price change for, or a from_date to use "
                "each bond's own yield change since that day, not both."
            )

        rounding = rounding if rounding is not None else self._rounding

        if from_date is None:
            selection = self._select_bonds(
                bonds, issuer, etfs, maturity_start, maturity_end, date
            )
            price_changes = {
                bond: bond_model.get_taylor_price_change(
                    **_bond_terms(row),
                    yield_to_maturity=row["Yield to Maturity"],
                    yield_change=0.01 if yield_change is None else yield_change,
                )
                for bond, row in selection.iterrows()
            }

            return self._bond_table(
                selection,
                pd.Series(price_changes, name="Price Change", dtype=float),
                rounding,
            )

        # The bonds are selected on the earlier day, so the benchmark Treasuries are those
        # that already traded then, and priced again on the later day.
        earlier = self._select_bonds(
            bonds, issuer, etfs, maturity_start, maturity_end, from_date
        )
        treasuries = earlier.index[earlier["Issuer"] == "United States Treasury"]
        if treasuries.empty:
            raise ValueError(
                "Only Treasuries have prices on earlier days, so a from_date needs Treasuries."
            )
        if len(treasuries) < len(earlier):
            logger.info(
                "Only Treasuries have prices on earlier days, so the other bonds are left out."
            )
        earlier = earlier.loc[treasuries]
        later = self._select_bonds(
            list(treasuries), None, None, None, None, date
        ).reindex(earlier.index)
        if (later["Price Date"].dropna() <= earlier["Price Date"].max()).any():
            raise ValueError(
                f"The from_date {earlier['Price Date'].max()} must be a business day before "
                f"the price date {later['Price Date'].max()}."
            )

        yield_changes = later["Yield to Maturity"] - earlier["Yield to Maturity"]
        price_changes = {
            bond: (
                bond_model.get_taylor_price_change(
                    **_bond_terms(row),
                    yield_to_maturity=row["Yield to Maturity"],
                    yield_change=yield_changes[bond],
                )
                if np.isfinite(yield_changes[bond])
                else np.nan
            )
            for bond, row in earlier.iterrows()
        }

        table = pd.concat(
            [
                earlier[["Issuer", "Security Type", "Coupon", "Maturity"]],
                earlier[["Price", "Price Date"]].rename(
                    columns={"Price": "From Price", "Price Date": "From Date"}
                ),
                later[["Price", "Price Date"]],
                pd.DataFrame(
                    {
                        "Yield Change": yield_changes,
                        "Price Change": pd.Series(price_changes, dtype=float),
                        "Realised Price Change": later["Price"] / earlier["Price"] - 1,
                    }
                ),
            ],
            axis=1,
        )

        return apply_rounding(table, rounding)

    @handle_errors
    def get_bond_equivalent_yield(
        self,
        bonds: list[str] | str | None = None,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the bond-equivalent yield (BEY) of every Treasury bill from its price.
        Bills are quoted on a discount-yield basis, which understates the return an investor
        earns because it is computed on face value rather than the (lower) purchase price,
        and uses a 360-day rather than a 365-day year. The bond-equivalent yield restates the
        discount yield on a basis that is comparable to coupon-bearing notes and bonds.

        The formula is as follows:

        - Discount Yield = (100 - Price) / 100 * 360 / Days to Maturity
        - BEY = 365 * Discount Yield / (360 - Days to Maturity * Discount Yield), for 182.5 days or less
        - BEY = (-2t + 2 * SQRT(t^2 - (2t - 1) * (1 - 100 / Price))) / (2t - 1), with t = Days to Maturity / 365,
          for more than 182.5 days

        A bill with half a year or less remaining takes the simple restatement onto the purchase price and a
        365-day year. Beyond that an equivalent coupon-bearing note would have paid a coupon at the six month
        point, so the U.S. Treasury's semi-annually compounded solution (31 CFR 356, Appendix B) is used instead.
        Applying the simple formula to a 52-week bill instead overstates its yield by roughly seven basis points.

        The prices are the Treasury's daily FedInvest prices, see `get_treasury_securities`. To convert a discount
        yield of your own, use `bond_model.get_bond_equivalent_yield`.

        Also known as: BEY, coupon-equivalent yield, investment rate, T-bill yield.

        Args:
            bonds (list[str] | str | None, optional): The CUSIPs (or US ISINs) of the bills,
                a list or a comma-separated string. Defaults to None, every bill.
            date (str | None, optional): The day of the prices (YYYY-MM-DD). Defaults to None,
                the latest day with prices.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bill, indexed by CUSIP, with the Maturity, Days to
            Maturity, Price, Discount Yield, Bond-Equivalent Yield and Price Date.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        bills = fixedincome.get_bond_equivalent_yield(date="2026-11-05")

        bills.iloc[::4, :-1]
        ```

        Which returns:

        | CUSIP     | Maturity   |   Days to Maturity |   Price |   Discount Yield |   Bond-Equivalent Yield |
        |:----------|:-----------|-------------------:|--------:|-----------------:|------------------------:|
        | 912797VW4 | 2026-11-10 |                  5 | 99.9576 |           0.0306 |                  0.031  |
        | 912797VY0 | 2026-11-24 |                 19 | 99.8075 |           0.0365 |                  0.0371 |
        | 912797WE3 | 2026-12-08 |                 33 | 99.6551 |           0.0376 |                  0.0383 |
        | 912797WG8 | 2026-12-22 |                 47 | 99.4978 |           0.0385 |                  0.0392 |
        | 912797WN3 | 2027-01-05 |                 61 | 99.3367 |           0.0391 |                  0.04   |
        | 912797WQ6 | 2027-01-19 |                 75 | 99.1819 |           0.0393 |                  0.0401 |
        | 912797WS2 | 2027-02-02 |                 89 | 99.0271 |           0.0394 |                  0.0403 |
        | 912797WZ6 | 2027-02-16 |                103 | 98.8582 |           0.0399 |                  0.0409 |
        | 912797XB8 | 2027-03-02 |                117 | 98.6982 |           0.0401 |                  0.0411 |
        | 912797WL7 | 2027-03-25 |                140 | 98.4401 |           0.0401 |                  0.0413 |
        | 912797WV5 | 2027-04-22 |                168 | 98.1073 |           0.0406 |                  0.0419 |
        | 912797VF1 | 2027-06-10 |                217 | 97.57   |           0.0403 |                  0.0418 |
        | 912797WJ2 | 2027-09-30 |                329 | 96.2827 |           0.0407 |                  0.0424 |
        """
        bills = self.get_treasury_securities(
            date=date,
            security_type="Bill",
            bonds=bonds,
            rounding=10,
        )
        if not isinstance(bills, pd.DataFrame) or bills.empty:
            raise ValueError(
                "No Treasury bill prices are available for the date given."
            )

        days = (
            pd.to_datetime(bills["Maturity"]) - pd.to_datetime(bills["Price Date"])
        ).dt.days
        discount_yield = (100 - bills["Price"]) / 100 * 360 / days
        result = pd.DataFrame(
            {
                "Maturity": bills["Maturity"],
                "Days to Maturity": days,
                "Price": bills["Price"],
                "Discount Yield": discount_yield,
                "Bond-Equivalent Yield": [
                    bond_model.get_bond_equivalent_yield(value, day)
                    for value, day in zip(discount_yield, days, strict=True)
                ],
                "Price Date": bills["Price Date"],
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
