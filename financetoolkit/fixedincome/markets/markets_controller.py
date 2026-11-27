"""Bond Markets Module"""

__docformat__ = "google"

from datetime import (
    date as date_type,
    datetime,
    timedelta,
)
from typing import Literal

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from financetoolkit import (
    fmp_model as toolkit_fmp_model,
    helpers,
    historical_model,
)
from financetoolkit.economics.sources import bls_model
from financetoolkit.fixedincome import helpers as fixedincome_helpers
from financetoolkit.fixedincome.curves.curves_controller import CurvesFixedIncome
from financetoolkit.fixedincome.helpers import maturity_key, parse_maturities
from financetoolkit.fixedincome.markets import (
    bond_data_model,
    bond_futures_model,
    fitted_curve_model,
    frn_model,
    inflation_bond_model,
)
from financetoolkit.fixedincome.sources import (
    fed_model,
    fiscaldata_model,
    fred_model as fixedincome_fred_model,
    treasurydirect_model,
)
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import check_option, handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = logger_model.get_logger()

TREASURY_SECURITY_TYPES = (
    "Bill",
    "Cash Management Bill",
    "Note",
    "Bond",
    "TIPS",
    "FRN",
)

# The kinds of marketable Treasury FedInvest prices.
TREASURY_PRICE_TYPES = ("Bill", "Note", "Bond", "TIPS", "FRN")

# Bond ETFs whose holdings make up the bond universe by default: iShares funds publish their
# holdings daily with each bond's coupon, maturity and par amount. Together they hold
# investment grade corporate bonds of every maturity (LQD, from three years; IGSB, up to
# five years) and high yield corporate bonds (USHY).
BOND_FUNDS = ["LQD", "IGSB", "USHY"]

# The maturities, in years, get_issuer_spreads compares issuers at by default.
DEFAULT_SPREAD_MATURITIES = (2, 3, 5, 7, 10, 20, 30)

# CME's US Treasury futures and the rules of their delivery baskets: the kind of security,
# the remaining term in months (at least, from the first day of the delivery month, and up
# to or below a number of months from its first or last day), the original terms a note must have been auctioned at,
# how the months to maturity are rounded in the conversion factor and the last delivery day.
TREASURY_FUTURES: dict[str, dict] = {
    "ZT": {
        "security_type": "Note",
        "remaining": (21, (24, "last", "up to")),
        "original_terms": ("2-Year", "3-Year", "5-Year"),
        "rounding": "month",
        "delivery": "third business day",
    },
    "ZF": {
        "security_type": "Note",
        "remaining": (50, (63, "first", "up to")),
        "original_terms": ("2-Year", "3-Year", "5-Year"),
        "rounding": "month",
        "delivery": "third business day",
    },
    "ZN": {
        "security_type": "Note",
        "remaining": (78, (96, "first", "up to")),
        "original_terms": (),
        "rounding": "quarter",
        "delivery": "last business day",
    },
    "TN": {
        "security_type": "Note",
        "remaining": (113, (120, "first", "up to")),
        "original_terms": ("10-Year",),
        "rounding": "quarter",
        "delivery": "last business day",
    },
    "ZB": {
        "security_type": "Bond",
        # Less than 25 years: from 25 years on a bond is deliverable into UB instead.
        "remaining": (180, (300, "first", "below")),
        "original_terms": (),
        "rounding": "quarter",
        "delivery": "last business day",
    },
    "UB": {
        "security_type": "Bond",
        "remaining": (300, None),
        "original_terms": (),
        "rounding": "quarter",
        "delivery": "last business day",
    },
}

# The letters futures symbols give the delivery months.
FUTURES_MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}


@filter_shown_columns
@track_sources
class MarketsFixedIncome(CurvesFixedIncome):
    """
    Bond markets: the bond universe, issuer spreads, Treasury securities and auctions.
    """

    @handle_errors
    def get_treasury_auctions(
        self,
        security_type: list[str] | str | None = None,
        security_term: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the results of the US Treasury auctions held between the start and end
        date: every bill, note, bond, TIPS and floating rate note the Treasury sold, with the
        yield it sold at, how many times the offering was bid (the bid-to-cover ratio) and
        who bought it. Auction results are the most direct read of the demand for US
        government debt: a lower bid-to-cover ratio or a smaller share for indirect bidders
        (which include foreign central banks) at a large auction is a sign of weak demand.

        The data comes from the Treasury's Fiscal Data API, from 1979 on.
        The Yield is the high yield of notes and bonds and the investment rate (the bond
        equivalent yield) of bills; the High Discount Rate is the discount rate bills are
        quoted at. The Bid-to-Cover Ratio is the Treasury's, which leaves out the Federal
        Reserve's (SOMA) bids, while the Total Tendered and Total Accepted include them, so
        their ratio is lower when the Federal Reserve rolls over maturing holdings. The
        bidder shares are each class's part of the competitive awards:
        primary dealers (who must bid at every auction), direct bidders (buying for their
        own account) and indirect bidders (bidding through a dealer, such as foreign and
        international monetary authorities and investment funds). Auctions that are
        announced but not yet held, and older auctions for which the Treasury did not
        publish a breakdown, show NaN for the missing results.

        See definition: https://fiscaldata.treasury.gov/datasets/treasury-securities-auctions-data/

        Also known as: Treasury auction results, bid-to-cover ratio, auction demand,
        indirect bidders, Treasury issuance.

        Args:
            security_type (list[str] | str | None, optional): The kind of security, or several (a list or
                a comma-separated string). Options are: "Bill", "Cash Management Bill", "Note", "Bond",
                "TIPS", "FRN", None (every kind). Defaults to None.
            security_term (str | None, optional): The term as the Treasury writes it, such as
                "13-Week", "10-Year" or "30-Year". Defaults to None, which returns every term.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which keeps the precision the Treasury publishes.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per auction, indexed by the auction date, with the security,
            its dates, the Coupon Rate, Yield and High Discount Rate as decimals, the
            Bid-to-Cover Ratio, the amounts in dollars and the bidder shares as decimals.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2025-01-01', end_date='2026-10-23')

        auctions = fixedincome.get_treasury_auctions(security_type='Note', security_term='10-Year')

        auctions[['Security Term', 'Reopening', 'Yield', 'Bid-to-Cover Ratio', 'Indirect Bidder Share']].tail()
        ```

        Which returns:

        | Auction Date   | Security Term   | Reopening   |   Yield |   Bid-to-Cover Ratio |   Indirect Bidder Share |
        |:---------------|:----------------|:------------|--------:|---------------------:|------------------------:|
        | 2025-08-06     | 10-Year         | No          | 0.04255 |                 2.35 |                  0.6423 |
        | 2025-11-12     | 10-Year         | No          | 0.04074 |                 2.43 |                  0.67   |
        | 2026-02-11     | 10-Year         | No          | 0.04177 |                 2.39 |                  0.6454 |
        | 2026-05-12     | 10-Year         | No          | 0.04468 |                 2.4  |                  0.6395 |
        | 2026-08-12     | 10-Year         | No          | 0.04683 |                 2.53 |                  0.7673 |
        """
        auctions = fiscaldata_model.get_treasury_auctions(
            self._start_date, self._end_date
        )

        if security_type is not None:
            requested = helpers.split_values(security_type)
            types = {kind.lower(): kind for kind in TREASURY_SECURITY_TYPES}
            if unknown := [kind for kind in requested if kind.lower() not in types]:
                raise ValueError(
                    f"The security type {', '.join(unknown)} does not exist. Choose from "
                    f"{', '.join(TREASURY_SECURITY_TYPES)}."
                )
            auctions = auctions[
                auctions["Security Type"].isin(
                    [types[kind.lower()] for kind in requested]
                )
            ]
        if security_term is not None:
            auctions = auctions[
                auctions["Security Term"].str.lower() == security_term.strip().lower()
            ]

        if auctions.empty:
            return pd.DataFrame()

        auctions = auctions.copy()
        auctions["Yield"] = auctions["High Yield"].fillna(
            auctions["High Investment Rate"]
        )
        bidders = {
            "Primary Dealer Share": "Primary Dealers Accepted",
            "Direct Bidder Share": "Direct Bidders Accepted",
            "Indirect Bidder Share": "Indirect Bidders Accepted",
        }
        competitive = auctions[list(bidders.values())].sum(axis=1, min_count=3)
        for share, accepted in bidders.items():
            auctions[share] = auctions[accepted] / competitive.where(competitive > 0)
            if rounding is None and self._rounding is not None:
                # The published figures keep their precision, the calculated shares do not
                # need more than the module's rounding.
                auctions[share] = auctions[share].round(self._rounding)

        auctions.index = pd.PeriodIndex(auctions.pop("Auction Date"), freq="D")
        auctions.index.name = "Auction Date"
        auctions = auctions[
            [
                "Security Type",
                "Security Term",
                "CUSIP",
                "Issue Date",
                "Maturity Date",
                "Reopening",
                "Coupon Rate",
                "Yield",
                "High Discount Rate",
                "Bid-to-Cover Ratio",
                "Offering Amount",
                "Total Tendered",
                "Total Accepted",
                *bidders,
            ]
        ]

        return apply_rounding(auctions, rounding)

    @handle_errors
    def get_bond_universe(
        self,
        etfs: list[str] | str | None = None,
        issuer: list[str] | str | None = None,
        maturity_start: str | None = None,
        maturity_end: str | None = None,
        bonds: list[str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves thousands of individual US dollar bonds with their price, yield to maturity,
        modified duration and spread over Treasuries, read from the holdings of bond ETFs.

        Individual bond prices are not published for free, but bond ETFs publish their
        holdings every day: each bond's coupon and maturity (in its name), its par amount and
        its market value, from which the price follows. By default this reads the iShares
        iBoxx $ Investment Grade Corporate Bond ETF (LQD), the iShares 0-5 Year Investment
        Grade Corporate Bond ETF (IGSB) and the iShares Broad USD High Yield Corporate Bond
        ETF (USHY): together about 8,000 investment grade and high yield corporate bonds. A
        bond held by several funds is listed once, with the funds that hold it, at the median
        of their prices on the latest Price Date among them.

        The formula is as follows:

        - Price = Market Value / Par Amount × 100 (a clean price per 100 of par)
        - Yield to Maturity: the rate y at which the coupons and principal, discounted at
          (1 + y / 2) per half year, are worth the clean price plus accrued interest
          (see `get_yield_to_maturity`)
        - Modified Duration = Macaulay Duration / (1 + y / 2) (see `get_duration`)
        - G-Spread = Yield to Maturity - Treasury Par Yield at the bond's maturity, on the
          bond's Price Date

        Coupons are semi-annual, the US convention for corporate bonds, and the yield is a
        nominal annual rate compounded twice a year (bond-equivalent), like the Treasury par
        yields it is compared with. The coupon periods fall on the bond's actual coupon dates,
        counted back from maturity (on the last day of the month for a bond maturing on a
        month's last day), and the time in the current period is counted on the 30/360 Bond
        Basis, the day count of US corporate bonds: 30 days to a month and 180 days to a
        coupon period. The Treasury par yield at a maturity is interpolated linearly between the
        maturities of the US Treasury par yield curve on the price date (held at the 1-month
        and 30-year yield beyond them).

        Also known as: corporate bond prices, bond yields, bond screener, credit spreads per
        bond, G-spread.

        Args:
            etfs (list[str] | str | None, optional): The bond ETFs to read the holdings of, as
                a list or a comma-separated string. The holdings must name each bond as
                "ISSUER COUPON% MM/DD/YYYY" with its ISIN and par amount, as iShares funds
                do. Defaults to None, which uses LQD, IGSB and USHY.
            issuer (list[str] | str | None, optional): Keeps the bonds whose issuer contains
                this text (case-insensitive), e.g. "APPLE" or "JPMORGAN", or any of several.
                Defaults to None, every issuer.
            maturity_start (str | None, optional): Keeps the bonds maturing on or after this
                date (YYYY-MM-DD). Defaults to None.
            maturity_end (str | None, optional): Keeps the bonds maturing on or before this
                date (YYYY-MM-DD). Defaults to None.
            bonds (list[str] | str | None, optional): Keeps the bonds with these ISINs or
                CUSIPs, as a list or a comma-separated string. Defaults to None, every bond.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per bond, indexed by ISIN, with the Issuer, CUSIP, Coupon
            (decimal), Maturity, whether it is a Rule 144A bond, the Price per 100 of par, the
            Yield to Maturity and G-Spread (decimals), the Modified Duration (years), the
            Price Date and the Funds that hold it, sorted by issuer and maturity.

        Notes:
        - The prices are the funds' own end-of-day valuations (iShares uses evaluated bid
          prices), not trades. iShares publish the holdings valued at the previous business
          day's close, which is the Price Date; FinancialModelingPrep serves only the latest
          holdings, so there is no history. For the Treasuries the funds hold, the yields are
          within about a basis point of the FedInvest prices of the same day (see
          `get_treasury_securities`).
        - Each bond is valued as of its Price Date: the accrued interest, the years to
          maturity and the Treasury par yield of the G-Spread are those of that day, not of a
          later settlement date.
        - Only holdings that publish an ISIN are read. A US ISIN shows that a bond is a US
          security, which an ISIN built from the CUSIP does not: the euro and sterling notes
          of US companies carry US CUSIPs as well. FinancialModelingPrep gives no ISINs for
          the funds of Vanguard, Schwab and Invesco, whose valuation dates also differ from
          the date of the feed (the Vanguard holdings are those of the last month end), so
          their bonds are left out, with a warning.
        - The yield is to maturity. A callable bond that trades above its call price
          yields less to its call date (the yield to worst), which is common among high
          yield bonds; call schedules are not in the holdings.
        - The fund prints the coupon to two decimals (2.125% shows as 2.13%), which moves the
          yield by well under a basis point a year of maturity.
        - Left out, with a count in the log: lines that are not fixed-coupon bonds (cash,
          futures, swaps), floating rate notes, bank notes marked fixed-to-floating
          ("FXD-FRN", which switch to a floating coupon before maturity), inflation-linked
          bonds, mortgage pools, perpetual bonds, holdings without an ISIN, bonds without a
          US ISIN (the funds of other currencies) and prices outside 1 to 300. A
          fixed-to-floating note the fund does not mark as such is kept, and its yield
          assumes the fixed coupon to maturity.
        - Requires a FinancialModelingPrep API key for the holdings; the Treasury curve needs
          none.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(api_key="FINANCIAL_MODELING_PREP_KEY")

        bonds = fixedincome.get_bond_universe(issuer="APPLE INC")

        bonds[["Coupon", "Maturity", "Price", "Yield to Maturity", "Modified Duration", "G-Spread"]].head()
        ```

        Which returns:

        | ISIN         |   Coupon | Maturity   |   Price |   Yield to Maturity |   Modified Duration |   G-Spread |
        |:-------------|---------:|:-----------|--------:|--------------------:|--------------------:|-----------:|
        | US037833EC07 |   0.012  | 2028-02-08 |   95.85 |              0.0475 |              1.1795 |     0.0021 |
        | US037833ET32 |   0.04   | 2028-05-10 |   98.9  |              0.0478 |              1.4091 |     0.0022 |
        | US037833EY27 |   0.04   | 2028-05-12 |   98.87 |              0.048  |              1.4144 |     0.0023 |
        | US037833EH93 |   0.014  | 2028-08-05 |   94.55 |              0.0476 |              1.6476 |     0.0016 |
        | US037833FD70 |   0.0465 | 2028-11-09 |   99.74 |              0.0479 |              1.8577 |     0.0016 |
        """
        helpers.require_api_key(self._api_key, "the bond universe")

        # The bonds asked for, before the name is reused for the table of bonds.
        identifiers = {value.upper() for value in helpers.split_values(bonds)}
        funds = [fund.upper() for fund in (helpers.split_values(etfs) or BOND_FUNDS)]
        holdings, missing = toolkit_fmp_model.get_etf_holdings(funds, self._api_key)
        if missing:
            logger.warning("No holdings were found for %s.", ", ".join(missing))
        if holdings.empty:
            raise ValueError(
                f"None of {', '.join(funds)} has holdings to read bonds from."
            )

        # Fund families count the par amount differently: iShares in hundreds (market value /
        # amount is the price per 100), others in dollars (the price per 1).
        ratios = pd.to_numeric(
            holdings["Market Value"], errors="coerce"
        ) / pd.to_numeric(holdings["Shares"], errors="coerce").where(
            lambda values: values > 0
        )
        parsed_names = holdings["Name"].map(bond_data_model.parse_bond_name).notna()
        scale = (
            ratios[parsed_names]
            .groupby(level=0)
            .median()
            .map(lambda median: 100 if median < 2 else 1)  # noqa: PLR2004
        )

        isins = holdings["ISIN"].fillna("").astype(str).str.strip()
        without_isins = [
            fund
            for fund, values in isins.groupby(level=0, sort=False)
            if not values.any()
        ]
        if without_isins:
            logger.warning(
                "The holdings of %s give no ISINs, so their bonds are left out: without one "
                "neither the currency nor the date the fund valued them at is known.",
                ", ".join(without_isins),
            )

        skipped: dict[str, int] = {}
        records = []
        # iShares publish the holdings valued at the previous business day's close.
        price_dates = (
            pd.to_datetime(holdings["Updated At"].astype(str).str[:10])
            - pd.offsets.BDay(1)
        ).dt.date
        for position, (fund, holding) in enumerate(
            zip(
                holdings.index.get_level_values(0),
                holdings.to_dict("records"),
                strict=True,
            )
        ):
            parsed = bond_data_model.parse_bond_name(holding["Name"])
            reason = (
                "not a fixed-coupon bond"
                if parsed is None
                else bond_data_model.exclusion_reason(
                    holding["Name"], parsed["Maturity"]
                )
            )
            isin = isins.iloc[position]
            # A bond without a US ISIN is a foreign security, such as the euro bonds of a
            # European fund, whose coupons and currency differ. Without a published ISIN the
            # currency is unknown: the euro notes of US companies carry US CUSIPs too.
            if reason is None and not isin:
                reason = "no ISIN"
            elif reason is None and not isin.startswith("US"):
                reason = "not a US security"
            price = ratios.iloc[position] * scale.get(fund, 1)
            low, high = bond_data_model.PRICE_RANGE
            if reason is None and not low < price < high:
                reason = "no usable price"
            if reason is not None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue

            records.append(
                {
                    "ISIN": isin,
                    "CUSIP": holding["CUSIP"],
                    **parsed,
                    "Price": price,
                    "Price Date": price_dates.iloc[position],
                    "Fund": fund,
                }
            )

        if skipped:
            logger.info(
                "Left out of the bond universe: %s.",
                ", ".join(
                    f"{count} {reason}" for reason, count in sorted(skipped.items())
                ),
            )
        if not records:
            raise ValueError(
                f"The holdings of {', '.join(funds)} hold no US fixed-coupon bonds with an ISIN."
            )

        held = pd.DataFrame(records)
        funds_held = held.groupby("ISIN", sort=False)["Fund"].agg(
            lambda values: ", ".join(dict.fromkeys(values))
        )
        # A bond held by several funds takes the median of their prices on the latest Price
        # Date among them, so the order of the funds does not decide its price.
        held = held[
            held["Price Date"] == held.groupby("ISIN")["Price Date"].transform("max")
        ]
        prices = held.groupby("ISIN")["Price"].median()
        bonds = held.drop_duplicates("ISIN").set_index("ISIN")
        bonds["Price"] = prices
        bonds["Funds"] = funds_held

        if identifiers:
            bonds = bonds[
                bonds.index.str.upper().isin(identifiers)
                | bonds["CUSIP"].astype(str).str.upper().isin(identifiers)
            ]
        if issuer is not None:
            names = [name.upper() for name in helpers.split_values(issuer)]
            bonds = bonds[
                bonds["Issuer"]
                .str.upper()
                .apply(lambda value: any(name in value for name in names))
            ]
        if maturity_start is not None:
            bonds = bonds[bonds["Maturity"] >= pd.Timestamp(maturity_start).date()]
        if maturity_end is not None:
            bonds = bonds[bonds["Maturity"] <= pd.Timestamp(maturity_end).date()]
        if bonds.empty:
            raise ValueError(
                "No bond in the holdings matches the bonds, issuer and maturity given."
            )

        # A bond that matured by its price date has no years left to price.
        bonds = bonds[
            [
                maturity > settlement
                for settlement, maturity in zip(
                    bonds["Price Date"], bonds["Maturity"], strict=True
                )
            ]
        ].copy()
        # US corporate bonds count their days on the 30/360 Bond Basis.
        bonds["Years"] = [
            bond_data_model.get_years_to_maturity(
                settlement, maturity, day_count="30/360"
            )
            for settlement, maturity in zip(
                bonds["Price Date"], bonds["Maturity"], strict=True
            )
        ]
        bonds = bonds[bonds["Years"].notna()]
        analytics = [
            bond_data_model.get_bond_analytics(coupon, years, price)
            for coupon, years, price in zip(
                bonds["Coupon"], bonds["Years"], bonds["Price"], strict=True
            )
        ]
        bonds["Yield to Maturity"] = [values[0] for values in analytics]
        bonds["Modified Duration"] = [values[1] for values in analytics]

        # Each bond is compared with the Treasury curve of its own Price Date.
        try:
            treasury_yields = pd.concat(
                [
                    bond_data_model.interpolate_curve(
                        self._latest_yield_curve("United States", date=str(price_date)),
                        years,
                    )
                    for price_date, years in bonds.groupby("Price Date")["Years"]
                ]
            )
            bonds["G-Spread"] = bonds["Yield to Maturity"] - treasury_yields
        except (ValueError, requests.exceptions.RequestException) as error:
            logger.warning(
                "The G-Spread is left empty: no Treasury curve (%s).",
                logger_model.redact(error),
            )
            bonds["G-Spread"] = np.nan

        bonds = bonds.sort_values(["Issuer", "Maturity"], kind="stable")
        for column in ["Maturity", "Price Date"]:
            bonds[column] = bonds[column].astype(str)

        bonds = bonds[
            [
                "Issuer",
                "CUSIP",
                "Coupon",
                "Maturity",
                "Rule 144A",
                "Price",
                "Yield to Maturity",
                "Modified Duration",
                "G-Spread",
                "Price Date",
                "Funds",
            ]
        ]

        return apply_rounding(
            bonds, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_issuer_spreads(
        self,
        issuer: list[str] | str | None = None,
        maturity_years: list[float] | str | float | None = None,
        etfs: list[str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compares the credit spreads of bond issuers at the same maturities: the G-Spread
        (yield to maturity minus the Treasury par yield) of each issuer at, for example, 2, 5
        and 10 years, interpolated between the issuer's own bonds from `get_bond_universe`.
        Picking one bond per issuer by hand compares bonds of different maturities, and
        spreads rise with maturity; at the same maturity the difference is the credit.

        The formula is as follows:

        - Spread (issuer, T) = s(a) + (s(b) - s(a)) * (T - a) / (b - a)

        Where a and b are the years to maturity of the issuer's bonds closest below and above
        T and s their G-Spreads (see `get_bond_universe`). Bonds of the same maturity count
        with their average spread, and a maturity outside the range of the issuer's bonds
        has no spread, as a spread curve is not extrapolated.

        Also known as: issuer spread curve, credit curve, spread at a maturity, interpolated
        G-spread, credit spread comparison.

        Args:
            issuer (list[str] | str | None, optional): The issuers, each kept by the text its
                name contains (case-insensitive), e.g. "APPLE,BOEING". Defaults to None, every
                issuer in the funds.
            maturity_years (list[float] | str | float | None, optional): The maturities in years,
                as a list or comma-separated, e.g. "2,5,10". Defaults to None, which uses 2, 3,
                5, 7, 10, 20 and 30 years.
            etfs (list[str] | str | None, optional): The bond ETFs to read the bonds from (see
                `get_bond_universe`). Defaults to None, LQD, IGSB and USHY.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per issuer (the name the funds give it) the G-Spread at each
            maturity (decimals, columns such as "5Y"), the number of Bonds and the Shortest
            and Longest years to maturity among them.

        Notes:
        - An issuer's bonds of one name can differ in seniority, and the funds list a
          finance subsidiary (such as Ford Motor Credit) under its own name, apart from its
          parent.
        - The spreads are those of `get_bond_universe`: to maturity, at the funds' evaluated
          prices of the previous business day. A callable bond priced above its call price
          has a lower yield to worst.
        - Requires a FinancialModelingPrep API key for the holdings.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(api_key="FINANCIAL_MODELING_PREP_KEY")

        fixedincome.get_issuer_spreads(issuer="APPLE INC,BOEING,INTEL", maturity_years="2,5,10")
        ```

        Which returns:

        | Issuer            |     2Y |     5Y |    10Y |   Bonds |   Shortest |   Longest |
        |:------------------|-------:|-------:|-------:|--------:|-----------:|----------:|
        | APPLE INC         | 0.0016 | 0.0031 | 0.0057 |      43 |     1.2167 |   35.7167 |
        | BOEING CO         | 0.0065 | 0.0091 | 0.0105 |      20 |     1.1972 |   37.4472 |
        | INTEL CORPORATION | 0.0074 | 0.0118 | 0.0157 |      35 |     1.2222 |   36.2222 |
        """
        maturities = [
            float(value) for value in helpers.split_values(maturity_years)
        ] or list(DEFAULT_SPREAD_MATURITIES)
        bonds = self.get_bond_universe(etfs=etfs, issuer=issuer, rounding=None)

        if bonds.empty:
            raise ValueError("No bonds were found for the issuers given.")

        bonds = bonds.assign(
            Years=[
                bond_data_model.get_years_to_maturity(
                    pd.Timestamp(settlement).date(),
                    pd.Timestamp(maturity).date(),
                    day_count="30/360",
                )
                for settlement, maturity in zip(
                    bonds["Price Date"], bonds["Maturity"], strict=True
                )
            ]
        )

        return apply_rounding(
            bond_data_model.get_spreads_at_maturities(bonds, maturities),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_treasury_securities(
        self,
        date: str | None = None,
        security_type: list[str] | str | None = None,
        bonds: list[str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves every marketable US Treasury (bills, notes, bonds, TIPS and floating rate
        notes) with its price, yield to maturity and modified duration on a day, without an
        API key: what each Treasury trades at and yields.

        Where the Treasury par yield curve gives one yield per constant maturity, this lists
        the roughly 460 securities themselves, from the Bureau of the Fiscal Service's daily
        FedInvest prices: the prices the Treasury values its own accounts at. That shows
        where individual issues trade off the curve (old and new issues of the same maturity,
        the cheapest notes to deliver into futures) and gives the yield of any maturity date.

        The formula is as follows:

        - Notes and bonds: Yield to Maturity is the rate y at which the coupons and principal,
          discounted at (1 + y / 2) per half year, are worth the clean price plus accrued
          interest (see `get_yield_to_maturity`)
        - Bills: Discount Yield = (100 - Price) / 100 × 360 / Days, restated as the
          bond-equivalent yield (see `get_bond_equivalent_yield`)
        - Modified Duration = Macaulay Duration / (1 + y / 2) (see `get_duration`)
        - Floating rate notes: Yield to Maturity = Index Rate + Discount Margin and the
          Modified Duration is the index duration (see `get_treasury_floating_rate_notes`)

        Coupons are semi-annual and fall on the actual coupon dates, counted back from
        maturity (on the last day of the month for a note maturing on a month's last day,
        such as 30 April); the time in the current period is counted in actual days (actual/actual,
        the Treasury convention). Yields are nominal annual rates compounded twice a year
        (bond-equivalent), as the Treasury quotes them. For TIPS the price is per 100 of
        inflation-adjusted principal, so the yield is the real yield (see
        `get_treasury_inflation_protected_securities` for the index ratio and breakeven).
        Floating rate notes accrue at the 13-week bill rate plus a spread, which resets
        every week: their yield is the index rate plus the discount margin, simple interest
        on actual/360 compounded at each quarterly payment, the rate at which the Treasury's
        pricing formula gives the price, and their duration that to the index rate, which
        their coupons follow, so it is close to zero. The Coupon of a floating rate note is
        the rate it accrues at, as FedInvest lists it.

        See definition: https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate

        Also known as: Treasury bond prices, Treasury yields per security, CUSIP yields,
        off-the-run Treasuries, FedInvest prices.

        Args:
            date (str | None, optional): The day (YYYY-MM-DD); a weekend or holiday uses the
                business day before it. Defaults to None, the latest day with prices.
            security_type (list[str] | str | None, optional): The kind of security, or several (a list or
                a comma-separated string). Options are: "Bill", "Note", "Bond", "TIPS", "FRN", None
                (every kind). Defaults to None.
            bonds (list[str] | str | None, optional): Keeps the securities with these CUSIPs
                (or their US ISINs), as a list or a comma-separated string. Defaults to None,
                every security.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per security, indexed by CUSIP, with the Security Type,
            Coupon (decimal), Maturity, Price per 100 of par, Yield to Maturity (decimal),
            Modified Duration (years) and Price Date, sorted by maturity.

        Notes:
        - The price is the end-of-day price. It is published the next business day, so the
          latest day uses the bid (FedInvest's Sell price) until then.
        - One request per day; a day with end-of-day prices is cached for good.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        treasuries = fixedincome.get_treasury_securities(date="2026-11-05", security_type="Note,Bond")

        treasuries.loc[treasuries["Maturity"].between("2035-01-01", "2036-12-31")]
        ```

        Which returns:

        | CUSIP     | Security Type   |   Coupon | Maturity   |   Price |   Yield to Maturity |   Modified Duration | Price Date   |
        |:----------|:----------------|---------:|:-----------|--------:|--------------------:|--------------------:|:-------------|
        | 91282CMM0 | Note            |   0.0462 | 2035-02-15 | 99.0312 |              0.0477 |              6.7323 | 2026-11-05   |
        | 91282CNC1 | Note            |   0.0425 | 2035-05-15 | 96.3125 |              0.0478 |              6.8997 | 2026-11-05   |
        | 91282CNT4 | Note            |   0.0425 | 2035-08-15 | 96.125  |              0.048  |              7.1425 | 2026-11-05   |
        | 91282CPJ4 | Note            |   0.04   | 2035-11-15 | 94.1562 |              0.0481 |              7.2897 | 2026-11-05   |
        | 912810FT0 | Bond            |   0.045  | 2036-02-15 | 98.875  |              0.0465 |              7.4268 | 2026-11-05   |
        | 91282CPZ8 | Note            |   0.0412 | 2036-02-15 | 94.9062 |              0.0481 |              7.5011 | 2026-11-05   |
        | 91282CQQ7 | Note            |   0.0438 | 2036-05-15 | 96.625  |              0.0482 |              7.5126 | 2026-11-05   |
        | 91282CRF0 | Note            |   0.0462 | 2036-08-15 | 98.4375 |              0.0483 |              7.6914 | 2026-11-05   |
        """
        prices, price_date = self._treasury_prices(date, security_type, bonds)

        values = prices.apply(
            bond_data_model.get_security_analytics,
            axis=1,
            result_type="expand",
            args=(price_date,),
        )
        prices["Yield to Maturity"] = values[0]
        prices["Modified Duration"] = values[1]

        floating = prices["Security Type"] == "FRN"
        if floating.any():
            # A floating rate note's yield is its index rate plus discount margin, and its
            # duration that to the index rate, which its coupons follow.
            try:
                notes = self._floating_rate_note_analytics(prices[floating], price_date)
                # Without the auctions (TreasuryDirect unreachable, which it reports) there
                # is no index rate to add, and nothing more to say about it.
                if not notes.empty:
                    cusips = prices.loc[floating, "CUSIP"]
                    prices.loc[floating, "Yield to Maturity"] = cusips.map(
                        notes["Yield"]
                    ).to_numpy()
                    prices.loc[floating, "Modified Duration"] = cusips.map(
                        notes["Index Duration"]
                    ).to_numpy()
            except (
                ValueError,
                KeyError,
                requests.exceptions.RequestException,
            ) as error:
                logger.warning(
                    "The floating rate notes are left without a yield and duration: %s",
                    logger_model.redact(error),
                )

        prices["Price Date"] = str(price_date)
        prices["Maturity"] = prices["Maturity"].dt.strftime("%Y-%m-%d")

        prices = prices.sort_values(
            ["Maturity", "Security Type"], kind="stable"
        ).set_index("CUSIP")[
            [
                "Security Type",
                "Coupon",
                "Maturity",
                "Price",
                "Yield to Maturity",
                "Modified Duration",
                "Price Date",
            ]
        ]

        return apply_rounding(
            prices, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_fitted_bond_curve(
        self,
        method: Literal[
            "svensson", "nelson_siegel", "exponential_spline", "cubic_spline"
        ] = "svensson",
        maturities: float | str | list | None = None,
        date: str | None = None,
        residuals: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Fits the US Treasury zero-coupon curve to the prices of every Treasury note and bond
        on a day, from the Treasury's FedInvest prices: the term structure the bonds
        themselves imply, with each bond's residual showing whether it is rich (priced above
        the curve) or cheap (below it), the relative value view of a bond desk.

        Where `get_zero_coupon_curve` bootstraps the dozen constant maturity par yields of
        the Treasury's own curve, this takes the roughly 330 notes and bonds themselves, so
        the curve reflects what trades.

        The curve minimizes the weighted squared differences between the bonds' prices and
        their prices on the curve, with weights inversely proportional to duration, as
        Gürkaynak, Sack and Wright (2007) fit the Federal Reserve's Treasury curve. A price
        error is about the duration times the price times the yield error, so the weights
        reduce the weight of the long bonds' larger price errors, while still giving a basis
        point of a long bond more weight than one of a short bond (equal weight in yield would
        need 1 / D²).

        The formula is as follows:

        - min Σ_i w_i (P_i^model - P_i)², with P_i^model = Σ_k CF_ik d(t_ik) and w_i = 1 / D_i

        where P_i is the clean price plus accrued interest of bond i, CF_ik its coupons and
        principal at times t_ik (actual days / 365 from the price date, on its actual coupon
        dates), D_i its modified duration and d(t) the discount function of the method:

        - Svensson: d(t) = e^(-y(t) t), with y(t) the Svensson (1994) zero rate
          β0 + β1 (1 - e^(-t/τ1)) / (t/τ1) + β2 [(1 - e^(-t/τ1)) / (t/τ1) - e^(-t/τ1)]
          + β3 [(1 - e^(-t/τ2)) / (t/τ2) - e^(-t/τ2)], the model of the Federal Reserve's
          curve (Gürkaynak, Sack and Wright, 2007)
        - Nelson-Siegel: the same without β3 (Nelson and Siegel, 1987)
        - Exponential spline: d(t) = Σ_{k=1}^{9} a_k e^(-k κ t) with Σ a_k = 1, the
          exponential basis of Vasicek and Fong (1982)
        - Cubic spline: d(t) a cubic spline with d(0) = 1 and about √n basis functions for
          n bonds, with knots at quantiles of the maturities so that as many bonds mature
          between every two knots (McCulloch, 1975)

        The Nelson-Siegel and Svensson fits are a bounded nonlinear least squares from fixed
        starts (the model fitted to the bonds' yields at their durations, and its betas with
        first decays of 0.5, 1, 2 and 5 years), with the bounds of
        `get_nelson_siegel_svensson`; a fit whose parameters are not identified (a flat ridge
        of equally good fits) raises an error rather than return an arbitrary point on it. The
        splines are linear in their coefficients and fitted exactly. As Gürkaynak, Sack and
        Wright do, bills, TIPS, floating rate notes and notes and bonds within three months of
        maturity are left out. Each day uses its own prices only. For prices of your own, use
        `fitted_curve_model.fit_bond_curve`.

        The fitted zero rates are annually compounded, like those of `get_zero_coupon_curve`,
        while the models are fitted on continuously compounded rates, so the betas compare
        with the Federal Reserve's; the instantaneous forward rate is continuously compounded.
        Yields of the bonds are bond-equivalent (compounded twice a year), as the Treasury
        quotes them.

        For more information, see the following papers:

        - McCulloch, J.H. (1975). "The Tax-Adjusted Yield Curve." The Journal of Finance,
          30(3), 811-830.
        - Vasicek, O.A. and Fong, H.G. (1982). "Term Structure Modeling Using Exponential
          Splines." The Journal of Finance, 37(2), 339-348.
        - Nelson, C.R. and Siegel, A.F. (1987). "Parsimonious Modeling of Yield Curves." The
          Journal of Business, 60(4), 473-489.
        - Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest Rates: Sweden
          1992-1994." NBER Working Paper 4871.
        - Gürkaynak, R.S., Sack, B. and Wright, J.H. (2007). "The U.S. Treasury Yield Curve:
          1961 to the Present." Journal of Monetary Economics, 54(8), 2291-2304.

        Shares similar logic:
        - QuantLib, FittedBondDiscountCurve: https://www.quantlib.org/reference/class_quant_lib_1_1_fitted_bond_discount_curve.html
        - QuantLib, SvenssonFitting (svensson): https://www.quantlib.org/reference/class_quant_lib_1_1_svensson_fitting.html
        - QuantLib, NelsonSiegelFitting (nelson_siegel): https://www.quantlib.org/reference/class_quant_lib_1_1_nelson_siegel_fitting.html
        - QuantLib, ExponentialSplinesFitting (exponential_spline): https://www.quantlib.org/reference/class_quant_lib_1_1_exponential_splines_fitting.html
        - QuantLib, CubicBSplinesFitting (cubic_spline): https://www.quantlib.org/reference/class_quant_lib_1_1_cubic_b_splines_fitting.html

        Also known as: fitted yield curve, Treasury curve fitting, rich cheap analysis,
        bond relative value, Svensson curve on bond prices, spline curve.

        Args:
            method (str, optional): The discount function. Options are: "svensson" (the
                Nelson-Siegel-Svensson zero rate), "nelson_siegel" (the Nelson-Siegel zero rate),
                "exponential_spline" (the exponential spline of Vasicek and Fong), "cubic_spline"
                (McCulloch's cubic spline). Defaults to "svensson".
            maturities (float | str | list | None, optional): The maturities to return the
                curve at, in years or as labels such as '6M' (a list or a comma-separated
                string). Defaults to None, 6 months and 1, 2, 3, 5, 7, 10, 20 and 30 years.
            date (str | None, optional): The day of the prices (YYYY-MM-DD); a weekend or
                holiday uses the business day before it. Defaults to None, the latest day
                with prices.
            residuals (bool, optional): Whether to return every bond with its price and yield
                on the curve and its residuals instead of the curve. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Zero Rate (annually compounded), the Discount Factor and the
            Instantaneous Forward Rate (continuously compounded) by maturity in years; or,
            with the residuals, per bond (by CUSIP) its Security Type, Coupon, Maturity,
            Price, Fitted Price (clean, on the curve), Price Residual (Price minus Fitted
            Price, positive for a bond that is rich), Yield to Maturity, Fitted Yield and Yield
            Residual (Yield to Maturity minus Fitted Yield, positive for a bond that is cheap).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        fixedincome.get_fitted_bond_curve(date="2026-11-05")
        ```

        Which returns:

        |   Maturity |   Zero Rate |   Discount Factor |   Instantaneous Forward Rate |
        |-----------:|------------:|------------------:|-----------------------------:|
        |        0.5 |      0.0428 |            0.9792 |                       0.0435 |
        |        1   |      0.0441 |            0.9577 |                       0.0451 |
        |        2   |      0.0455 |            0.9149 |                       0.046  |
        |        3   |      0.046  |            0.8738 |                       0.046  |
        |        5   |      0.0466 |            0.7964 |                       0.0472 |
        |        7   |      0.0475 |            0.7225 |                       0.0504 |
        |       10   |      0.0496 |            0.6162 |                       0.0555 |
        |       20   |      0.0549 |            0.3432 |                       0.0578 |
        |       30   |      0.0546 |            0.2032 |                       0.0461 |
        """
        check_option("method", method, fitted_curve_model.METHODS)
        prices, price_date = self._treasury_prices(date, "Note,Bond", None)
        prices = prices[
            prices["Maturity"].dt.date > price_date + relativedelta(months=3)
        ].reset_index(drop=True)

        schedules = []
        for coupon, maturity in zip(prices["Coupon"], prices["Maturity"], strict=True):
            _, coupon_dates = bond_data_model.get_coupon_dates(
                price_date, maturity.date()
            )
            times = np.array([(day - price_date).days / 365 for day in coupon_dates])
            flows = np.full(len(times), 100 * coupon / 2)
            flows[-1] += 100
            schedules.append((times, flows))
        width = max(len(times) for times, _ in schedules)
        cash_flow_times = np.zeros((len(schedules), width))
        cash_flows = np.zeros((len(schedules), width))
        for row, (times, flows) in enumerate(schedules):
            cash_flow_times[row, : len(times)] = times
            cash_flows[row, : len(flows)] = flows

        accrued = np.array(
            [
                bond_data_model.get_treasury_accrued_interest(
                    coupon, price_date, maturity.date()
                )
                for coupon, maturity in zip(
                    prices["Coupon"], prices["Maturity"], strict=True
                )
            ]
        )
        analytics = [
            bond_data_model.get_bond_analytics(
                coupon,
                bond_data_model.get_years_to_maturity(price_date, maturity.date()),
                price,
            )
            for coupon, maturity, price in zip(
                prices["Coupon"], prices["Maturity"], prices["Price"], strict=True
            )
        ]
        market_yields = np.array([value[0] for value in analytics])
        durations = np.array([value[1] for value in analytics])
        usable = np.isfinite(durations) & (durations > 0)

        parameters = fitted_curve_model.fit_bond_curve(
            cash_flow_times[usable],
            cash_flows[usable],
            (prices["Price"].to_numpy(dtype=float) + accrued)[usable],
            1 / durations[usable],
            method,
        )

        if not residuals:
            years = np.array(
                parse_maturities(maturities, [0.5, 1, 2, 3, 5, 7, 10, 20, 30]),
                dtype=float,
            )
            curve = fitted_curve_model.get_fitted_curve(parameters, years, method)
            curve.index = pd.Index(
                [maturity_key(value) for value in years], name="Maturity"
            )
            return apply_rounding(
                curve, rounding if rounding is not None else self._rounding
            )

        fitted_prices = (
            fitted_curve_model.get_fitted_bond_prices(
                parameters, cash_flow_times, cash_flows, method
            )
            - accrued
        )
        fitted_yields = np.array(
            [
                bond_data_model.get_bond_analytics(
                    coupon,
                    bond_data_model.get_years_to_maturity(price_date, maturity.date()),
                    price,
                )[0]
                for coupon, maturity, price in zip(
                    prices["Coupon"], prices["Maturity"], fitted_prices, strict=True
                )
            ]
        )
        bonds = pd.DataFrame(
            {
                "Security Type": prices["Security Type"].to_numpy(),
                "Coupon": prices["Coupon"].to_numpy(dtype=float),
                "Maturity": prices["Maturity"].dt.strftime("%Y-%m-%d").to_numpy(),
                "Price": prices["Price"].to_numpy(dtype=float),
                "Fitted Price": fitted_prices,
                "Price Residual": prices["Price"].to_numpy(dtype=float) - fitted_prices,
                "Yield to Maturity": market_yields,
                "Fitted Yield": fitted_yields,
                "Yield Residual": market_yields - fitted_yields,
            },
            index=pd.Index(prices["CUSIP"].to_numpy(), name="CUSIP"),
        )
        bonds = bonds[usable].sort_values(["Maturity", "Security Type"], kind="stable")

        return apply_rounding(
            bonds, rounding if rounding is not None else self._rounding
        )

    def _treasury_prices(
        self,
        date: str | None,
        security_type: list[str] | str | None,
        bonds: list[str] | str | None,
    ) -> tuple[pd.DataFrame, date_type]:
        """
        The FedInvest prices of the Treasuries of the kinds and CUSIPs asked for that have
        not matured on the price date.

        Args:
            date (str | None): The day (YYYY-MM-DD), see get_treasury_securities.
            security_type (list[str] | str | None): The kinds of security, see
                get_treasury_securities.
            bonds (list[str] | str | None): The CUSIPs or ISINs.

        Returns:
            tuple[pd.DataFrame, date]: The prices, one row per security, and the price date.

        Raises:
            ValueError: When a kind of security is unknown or no security matches.
        """
        requested_types = None
        if security_type is not None:
            types = {kind.lower(): kind for kind in TREASURY_PRICE_TYPES}
            requested = helpers.split_values(security_type)
            if unknown := [kind for kind in requested if kind.lower() not in types]:
                raise ValueError(
                    f"The security type {', '.join(unknown)} does not exist. Choose from "
                    f"{', '.join(TREASURY_PRICE_TYPES)}."
                )
            requested_types = [types[kind.lower()] for kind in requested]

        prices, price_date = treasurydirect_model.get_treasury_prices(date)
        if requested_types is not None:
            prices = prices[prices["Security Type"].isin(requested_types)]
        if bonds is not None:
            cusips = [
                fixedincome_helpers.treasury_cusip(value.upper()) or value.upper()
                for value in helpers.split_values(bonds)
            ]
            prices = prices[prices["CUSIP"].isin(cusips)]
        prices = prices[prices["Maturity"].dt.date > price_date].copy()
        if prices.empty:
            raise ValueError(
                f"FedInvest lists no security of the type and CUSIPs given on {price_date}."
            )

        return prices, price_date

    @staticmethod
    def _original_issues(auctions: pd.DataFrame) -> pd.DataFrame:
        """
        The terms of each security from its original auction (a reopening repeats them),
        indexed by CUSIP.

        Args:
            auctions (pd.DataFrame): The auctions, see
                treasurydirect_model.get_auctioned_securities.

        Returns:
            pd.DataFrame: One row per CUSIP, empty when TreasuryDirect could not be reached
            (which it reports).
        """
        if auctions.empty:
            return auctions

        return (
            auctions.sort_values("Reopening", kind="stable")
            .drop_duplicates("CUSIP")
            .set_index("CUSIP")
        )

    def _floating_rate_note_analytics(
        self, notes: pd.DataFrame, price_date: date_type
    ) -> pd.DataFrame:
        """
        The index rate, accrued interest, discount margin, yield and durations of Treasury
        floating rate notes on the price date, see get_treasury_floating_rate_notes.

        Args:
            notes (pd.DataFrame): The FedInvest prices of the notes (CUSIP, Maturity, Price).
            price_date (date): The price date, taken as the settlement date.

        Returns:
            pd.DataFrame: Per CUSIP the Spread, Index Rate, Accrued Interest, Discount
            Margin, Yield, Index Duration and Spread Duration, or an empty DataFrame when
            TreasuryDirect could not be reached for the auctions (which it reports).

        Raises:
            ValueError: When the terms of the notes or the 13-week bill auctions are missing.
        """
        terms = self._original_issues(
            treasurydirect_model.get_auctioned_securities("FRN")
        )
        bills = treasurydirect_model.get_auctioned_securities("Bill", "13-Week")

        # TreasuryDirect could not be reached, which it reports, so there is nothing to add.
        if terms.empty or bills.empty:
            return pd.DataFrame()

        bills = bills[
            ~bills["Cash Management Bill"]
            & bills["High Rate"].notna()
            & bills["Auction Date"].notna()
        ]
        # Only the auctions held before the price date (get_daily_index_rates takes those
        # before each day), one rate per auction date.
        index_rates = (
            pd.Series(
                [
                    frn_model.get_index_rate(rate, days if np.isfinite(days) else 91)
                    for rate, days in zip(
                        bills["High Rate"], bills["Days to Maturity"], strict=True
                    )
                ],
                index=bills["Auction Date"].to_numpy(),
            )
            .groupby(level=0)
            .last()
        )
        index_rates = index_rates[index_rates.index <= price_date]

        rows = {}
        for cusip, maturity, price in zip(
            notes["CUSIP"], notes["Maturity"], notes["Price"], strict=True
        ):
            if cusip not in terms.index:
                logger.warning("TreasuryDirect has no terms for the FRN %s.", cusip)
                continue
            spread = float(terms.loc[cusip, "Spread"])
            dated_date = terms.loc[cusip, "Dated Date"]
            previous, payments = bond_data_model.get_coupon_dates(
                price_date, maturity.date(), frequency=4
            )
            start = max(previous, dated_date) if dated_date else previous
            # The lockout applies only to the issue date and the interest payment dates
            # (31 CFR 356, Appendix B, IV: "applies only to the issue date and interest
            # payment dates"), so a secondary-market price date takes the latest auction
            # unless it is the note's dated date or an interest payment date. The index rate
            # that applies to the price date projects the coupons.
            index_rate = float(
                frn_model.get_daily_index_rates(
                    index_rates,
                    price_date,
                    price_date + timedelta(days=1),
                    [price_date] if price_date == start else [],
                ).iloc[0]
            )
            accrued = frn_model.get_floating_rate_note_accrued_interest(
                frn_model.get_daily_index_rates(
                    index_rates, start, price_date, [start]
                ),
                spread,
            )
            period_days = (
                np.diff([price_date, *payments]).astype("timedelta64[D]").astype(int)
            )
            margin = frn_model.get_discount_margin(
                price, index_rate, spread, period_days, accrued
            )
            index_duration, spread_duration = frn_model.get_frn_durations(
                index_rate, spread, margin, period_days, accrued
            )
            rows[cusip] = {
                "Spread": spread,
                "Index Rate": index_rate,
                "Accrued Interest": accrued,
                "Discount Margin": margin,
                "Yield": index_rate + margin,
                "Index Duration": index_duration,
                "Spread Duration": spread_duration,
            }

        if not rows:
            raise ValueError(
                "TreasuryDirect has no terms for these floating rate notes."
            )

        return pd.DataFrame.from_dict(rows, orient="index")

    @handle_errors
    def get_treasury_floating_rate_notes(
        self,
        date: str | None = None,
        bonds: list[str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the discount margin, yield and durations of every US Treasury floating
        rate note (FRN) from its FedInvest price: what each note pays over its index and how
        its price responds to the index and the margin, with the Treasury's own pricing
        formula (31 CFR 356, Appendix B, IV).

        A Treasury FRN pays, each quarter, the interest accrued day by day at the index rate
        (the high rate of the latest 13-week bill auction, as a money market yield) plus a
        fixed spread set at its first auction. Its price is close to par, and how far it is
        from par is read as the discount margin: the margin over the index rate at which the
        projected interest and principal are worth the price plus accrued interest. A note
        trading below par has a discount margin above its spread.

        The formula is as follows:

        - Index Rate = d / (1 - d * t / 360), from the high rate d of the 13-week bill auction
          of t days (see `frn_model.get_index_rate`)
        - Accrued Interest = Σ_j 100 * max(r_j + s, 0) / 360 over the days since the last
          interest payment, each at the index rate r_j of its day
        - Price + Accrued Interest = Σ_i A_i / Π_(k<=i) (1 + (r + m) * (T_k - T_(k-1)) / 360),
          with A_i = 100 * max(r + s, 0) / 360 * (T_i - T_(i-1)), the accrued interest added
          to the first payment and 100 to the last
        - Discount Margin m: the margin that solves the price equation
        - Yield = r + m, simple interest on actual/360, compounded at each payment
        - Index Duration = (P(r - 0.0001) - P(r + 0.0001)) / (2 * P * 0.0001)
        - Spread Duration = (P(m - 0.0001) - P(m + 0.0001)) / (2 * P * 0.0001)

        where s is the spread, r the index rate that applies to the price date (held for
        every later day), T_i the quarterly interest payment dates and P the price plus
        accrued interest. Each day accrues at the index rate of the most recent 13-week bill
        auction held before it, except that an auction held in the two business days before an
        issue or interest payment date applies from the day after that date (the lockout, which
        31 CFR 356 applies only to those dates, so not to a price date between them). The coupons follow the index, so the index duration is close to zero, while
        the spread duration, the price change for a change in the discount margin, is about
        the time to maturity.

        The pricing reproduces the Treasury's: the reopening of 91282CRD5 on 25 September
        2026 at a discount margin of 0.040% prices at 100.015499 with accrued interest of
        0.603670, as announced.

        Referenced against:
        - FinancePy, BondFRN.dirty_price_from_dm: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_frn.py#L107
        - FinancePy, BondFRN.modified_credit_duration: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_frn.py#L283

        Shares similar logic:
        - FinancePy, BondFRN.discount_margin: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_frn.py#L375

        See definition: https://www.ecfr.gov/current/title-31/part-356/appendix-Appendix%20B%20to%20Part%20356

        Also known as: FRN discount margin, floating rate note yield, Treasury FRN spread,
        spread duration, credit duration, quoted margin.

        Args:
            date (str | None, optional): The day of the FedInvest prices (YYYY-MM-DD); a weekend
                or holiday uses the business day before it. Defaults to None, the latest day
                with prices.
            bonds (list[str] | str | None, optional): Keeps the notes with these CUSIPs (or
                their US ISINs). Defaults to None, every floating rate note.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per note, indexed by CUSIP, with the Maturity, the clean Price
            and Accrued Interest per 100 of par, the Spread, Index Rate, Discount Margin and
            Yield (decimals), the Index Duration and Spread Duration (years) and the Price Date.

        Notes:
        - The price date is taken as the settlement date; the market settles a day later.
        - The spreads and the 13-week bill rates come from TreasuryDirect's auction data. The
          business days of the lockout are weekdays, without the holidays.
        - The Treasury pays a minimum daily interest of zero; the formula keeps that floor.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        frns = fixedincome.get_treasury_floating_rate_notes(date="2026-11-19")

        frns[["Maturity", "Price", "Spread", "Index Rate", "Discount Margin", "Yield", "Spread Duration"]]
        ```

        Which returns:

        | CUSIP     | Maturity   |   Price |   Spread |   Index Rate |   Discount Margin |   Yield |   Spread Duration |
        |:----------|:-----------|--------:|---------:|-------------:|------------------:|--------:|------------------:|
        | 91282CMJ7 | 2027-01-31 | 100.028 |   0.001  |        0.042 |           -0.0005 |  0.0415 |            0.2011 |
        | 91282CMX6 | 2027-04-30 | 100.1   |   0.0016 |        0.042 |           -0.0007 |  0.0413 |            0.4431 |
        | 91282CNQ0 | 2027-07-31 | 100.133 |   0.0016 |        0.042 |           -0.0004 |  0.0416 |            0.6905 |
        | 91282CPG0 | 2027-10-31 | 100.218 |   0.0019 |        0.042 |           -0.0004 |  0.0415 |            0.9352 |
        | 91282CPX3 | 2028-01-31 | 100.132 |   0.001  |        0.042 |           -0.0001 |  0.0418 |            1.1776 |
        | 91282CQM6 | 2028-04-30 | 100.154 |   0.001  |        0.042 |           -0.0001 |  0.0419 |            1.4119 |
        | 91282CRD5 | 2028-07-31 | 100.076 |   0.0005 |        0.042 |            0      |  0.042  |            1.6494 |
        | 91282CRS2 | 2028-10-31 | 100.023 |   0.0004 |        0.042 |            0.0003 |  0.0422 |            1.8841 |
        """
        prices, price_date = self._treasury_prices(date, "FRN", bonds)
        notes = self._floating_rate_note_analytics(prices, price_date)

        # TreasuryDirect could not be reached, which it reports.
        if notes.empty:
            return pd.DataFrame()

        prices = prices.set_index("CUSIP")
        notes["Maturity"] = prices["Maturity"].dt.strftime("%Y-%m-%d")
        notes["Price"] = prices["Price"]
        notes["Price Date"] = str(price_date)
        notes.index.name = "CUSIP"
        notes = notes.sort_values("Maturity", kind="stable")[
            [
                "Maturity",
                "Price",
                "Accrued Interest",
                "Spread",
                "Index Rate",
                "Discount Margin",
                "Yield",
                "Index Duration",
                "Spread Duration",
                "Price Date",
            ]
        ]

        return apply_rounding(
            notes, rounding if rounding is not None else self._rounding
        )

    def _consumer_price_index(self, first_day: date_type) -> pd.Series:
        """
        The US CPI-U, not seasonally adjusted, from the month of the first day on: the
        latest three years from the BLS, and with a FRED API key the history before that from
        FRED (CPIAUCNSA).

        Args:
            first_day (date): The first day of the first month needed.

        Returns:
            pd.Series: The CPI by month, with the months the BLS did not report filled as
            the Treasury does (see inflation_bond_model.fill_unreported_months).

        Raises:
            ValueError: When the CPI does not reach back to the first month.
        """
        index = bls_model.get_consumer_price_index()
        cpi = (
            index["all"].dropna() if "all" in index.columns else pd.Series(dtype=float)
        )

        def covered(series: pd.Series) -> bool:
            return not series.empty and series.index[0].start_time.date() <= first_day

        if not covered(cpi) and self._fred_api_key:
            # A year more, for the twelve-month change that fills an unreported month.
            history = fixedincome_fred_model.fetch_single_series(
                "CPIAUCNSA",
                str(first_day - relativedelta(months=13)),
                self._end_date,
                self._fred_api_key,
            )
            if not history.empty:
                history = history.iloc[:, 0].dropna()
                history.index = history.index.asfreq("M")
                cpi = history.combine_first(cpi) if not cpi.empty else history
        if not covered(cpi):
            raise ValueError(
                f"The CPI does not reach back to {first_day:%Y-%m}: the keyless BLS API "
                "returns about the last three years. Set a FRED API key for older dates."
            )

        return inflation_bond_model.fill_unreported_months(cpi)

    @handle_errors
    def get_treasury_inflation_protected_securities(
        self,
        date: str | None = None,
        bonds: list[str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the index ratio, inflation-adjusted price and breakeven inflation rate of
        every US Treasury Inflation-Protected Security (TIPS) from its FedInvest price: how
        much inflation has lifted each security's principal and what inflation its price
        implies.

        TIPS pay a fixed real coupon on a principal that grows with the Consumer Price Index
        (CPI-U, not seasonally adjusted). They are quoted per 100 of inflation-adjusted
        principal, a real price whose yield is a real yield; what a buyer pays per 100 of par
        is that price times the index ratio, the inflation accrued since the security's dated
        date. The nominal Treasury yield of the same maturity less the real yield is the
        inflation rate at which the two earn the same: the breakeven inflation rate of the
        security.

        The formula is as follows:

        - Ref CPI (Date) = CPI_(M-3) + (t - 1) / D * (CPI_(M-2) - CPI_(M-3)), for day t of
          month M with D days
        - Index Ratio = Ref CPI (Date) / Ref CPI (Dated Date)
        - Adjusted Price = Price * Index Ratio, and Adjusted Accrued Interest = Accrued
          Interest * Index Ratio
        - Real Yield: the rate y at which the real coupons and principal, discounted at
          (1 + y / 2) per half year, are worth the price plus accrued interest
          (see `get_yield_to_maturity`)
        - Breakeven Inflation Rate = Nominal Yield - Real Yield

        where the reference CPI and the index ratio are truncated to six decimal places and
        rounded to five (31 CFR 356, Appendix B, I.B), the reference CPI of the dated date is
        the one TreasuryDirect publishes with the security, and the Nominal Yield is the
        Treasury par yield (bond-equivalent, compounded semi-annually) at the security's
        maturity on the price date, interpolated linearly between the curve's maturities.
        Yields are compounded semi-annually; the time in the current coupon period is counted
        in actual days (actual/actual). A month the BLS did not report the CPI for (October
        2025) takes the index number the Treasury announced in its place, from the last
        twelve-month change.

        QuantLib's CPI interpolation is linear with three months' lag. The Treasury's own
        figures are reproduced: the reference CPI of 30 November 2026 is 336.87467, the index
        ratio of 91282CRE3 on that day 1.00870.

        See definition: https://www.treasurydirect.gov/marketable-securities/tips/

        Also known as: TIPS index ratio, inflation-adjusted principal, real yield, TIPS
        breakeven, inflation compensation, reference CPI.

        Referenced against:
        - QuantLib, CPI.laggedFixing: https://github.com/lballabio/QuantLib/blob/v1.44/ql/indexes/inflationindex.hpp#L39

        Shares similar logic:
        - QuantLib, CPIBond: https://www.quantlib.org/reference/class_quant_lib_1_1_c_p_i_bond.html
        - FinancePy, InflationBond.inflation_principal: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/inflation/InflationBond.py#L92

        Args:
            date (str | None, optional): The day of the FedInvest prices (YYYY-MM-DD); a weekend
                or holiday uses the business day before it. Defaults to None, the latest day
                with prices.
            bonds (list[str] | str | None, optional): Keeps the TIPS with these CUSIPs (or their
                US ISINs). Defaults to None, every TIPS.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per TIPS, indexed by CUSIP, with the Coupon (decimal), Maturity,
            the real Price per 100 of inflation-adjusted principal, the Reference CPI, Index
            Ratio, Adjusted Price and Adjusted Accrued Interest per 100 of par, the Real Yield,
            Nominal Yield and Breakeven Inflation Rate (decimals), the real Modified Duration
            (years) and the Price Date.

        Notes:
        - The price date is taken as the settlement date; the market settles a day later.
        - At maturity the Treasury repays at least the par amount (the deflation floor); the
          real yield leaves that option out, which matters only for a security whose index
          ratio is close to or below 1.
        - The CPI comes from the BLS's keyless API, which covers about the last three years;
          a date before that needs a FRED API key.
        - The breakeven rate compares with the par yield curve, not with a nominal note of the
          same maturity, and leaves out the inflation risk premium and the liquidity of TIPS.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        tips = fixedincome.get_treasury_inflation_protected_securities(date="2026-11-19")

        tips[["Maturity", "Index Ratio", "Real Yield", "Nominal Yield", "Breakeven Inflation Rate"]].tail()
        ```

        Which returns:

        | CUSIP     | Maturity   |   Index Ratio |   Real Yield |   Nominal Yield |   Breakeven Inflation Rate |
        |:----------|:-----------|--------------:|-------------:|----------------:|---------------------------:|
        | 912810TE8 | 2052-02-15 |        1.2076 |       0.0327 |          0.0541 |                     0.0214 |
        | 912810TP3 | 2053-02-15 |        1.1309 |       0.0329 |          0.0541 |                     0.0212 |
        | 912810TY4 | 2054-02-15 |        1.0953 |       0.0329 |          0.0541 |                     0.0212 |
        | 912810UH9 | 2055-02-15 |        1.0653 |       0.0328 |          0.0541 |                     0.0213 |
        | 912810US5 | 2056-02-15 |        1.0372 |       0.0326 |          0.054  |                     0.0214 |
        """
        prices, price_date = self._treasury_prices(date, "TIPS", bonds)
        terms = self._original_issues(
            treasurydirect_model.get_auctioned_securities("TIPS")
        )

        # TreasuryDirect could not be reached, which it reports.
        if terms.empty:
            return pd.DataFrame()

        cpi = self._consumer_price_index(
            price_date.replace(day=1)
            - relativedelta(months=inflation_bond_model.REFERENCE_LAG_MONTHS)
        )
        reference_cpi = inflation_bond_model.get_reference_cpi(cpi, price_date)
        if not np.isfinite(reference_cpi):
            raise ValueError(
                f"The CPI of the months the reference CPI of {price_date} needs is not "
                "published yet."
            )

        prices = prices.set_index("CUSIP")
        prices["Maturity Date"] = prices["Maturity"].dt.date
        prices["Years"] = [
            bond_data_model.get_years_to_maturity(price_date, maturity)
            for maturity in prices["Maturity Date"]
        ]
        analytics = [
            bond_data_model.get_bond_analytics(coupon, years, price)
            for coupon, years, price in zip(
                prices["Coupon"], prices["Years"], prices["Price"], strict=True
            )
        ]
        prices["Real Yield"] = [values[0] for values in analytics]
        prices["Modified Duration"] = [values[1] for values in analytics]
        prices["Reference CPI"] = reference_cpi
        prices["Index Ratio"] = [
            inflation_bond_model.get_index_ratio(
                reference_cpi,
                terms["Reference CPI on Dated Date"].get(cusip, np.nan),
            )
            for cusip in prices.index
        ]
        adjusted = [
            inflation_bond_model.get_inflation_adjusted_price(
                price,
                bond_data_model.get_treasury_accrued_interest(
                    coupon, price_date, maturity
                ),
                ratio,
            )
            for price, coupon, maturity, ratio in zip(
                prices["Price"],
                prices["Coupon"],
                prices["Maturity Date"],
                prices["Index Ratio"],
                strict=True,
            )
        ]
        prices["Adjusted Price"] = [values[0] for values in adjusted]
        prices["Adjusted Accrued Interest"] = [values[1] for values in adjusted]

        try:
            curve = self._latest_yield_curve("United States", date=str(price_date))
            prices["Nominal Yield"] = bond_data_model.interpolate_curve(
                curve,
                pd.Series(
                    [
                        (maturity - price_date).days / 365.25
                        for maturity in prices["Maturity Date"]
                    ],
                    index=prices.index,
                ),
            )
        except (ValueError, requests.exceptions.RequestException) as error:
            logger.warning(
                "The breakeven inflation rate is left empty: no Treasury curve (%s).",
                logger_model.redact(error),
            )
            prices["Nominal Yield"] = np.nan
        prices["Breakeven Inflation Rate"] = (
            prices["Nominal Yield"] - prices["Real Yield"]
        )

        prices["Price Date"] = str(price_date)
        prices["Maturity"] = prices["Maturity"].dt.strftime("%Y-%m-%d")
        prices = prices.sort_values("Maturity", kind="stable")[
            [
                "Coupon",
                "Maturity",
                "Price",
                "Reference CPI",
                "Index Ratio",
                "Adjusted Price",
                "Adjusted Accrued Interest",
                "Real Yield",
                "Nominal Yield",
                "Breakeven Inflation Rate",
                "Modified Duration",
                "Price Date",
            ]
        ]

        return apply_rounding(
            prices, rounding if rounding is not None else self._rounding
        )

    @staticmethod
    def _repo_rate(price_date: date_type) -> float:
        """
        The tri-party general collateral rate (TGCR) of the last day on or before the price
        date: the overnight rate of repo against Treasury collateral, published by the
        Federal Reserve Bank of New York.

        Args:
            price_date (date): The price date.

        Returns:
            float: The rate (decimal, simple interest on actual/360).

        Raises:
            ValueError: When no rate is published on or before the price date.
        """
        rates = fed_model.get_tri_party_general_collateral_rate()
        if rates.empty:
            # The source has logged that it could not be reached.
            raise ValueError(
                "No repo rate (TGCR) could be retrieved; pass repo_rate to price the forwards."
            )
        rates = rates["Rate"].loc[: pd.Period(price_date, freq="D")].dropna()
        if rates.empty:
            raise ValueError(f"No repo rate (TGCR) is published up to {price_date}.")

        return float(rates.iloc[-1])

    @staticmethod
    def _carry_terms(
        coupon: float, price_date: date_type, end: date_type, maturity: date_type
    ) -> tuple[float, float, list[tuple[float, int]]]:
        """
        The accrued interest per 100 of par of a Treasury note or bond on the price date and
        on a later date, and the coupons paid after the price date up to that date with the
        days to each.

        Args:
            coupon (float): The annual coupon rate (decimal).
            price_date (date): The price date.
            end (date): The forward or delivery date, before maturity.
            maturity (date): The maturity date.

        Returns:
            tuple[float, float, list[tuple[float, int]]]: The accrued interest on both dates
            and the coupons in between.
        """
        _, coupon_dates = bond_data_model.get_coupon_dates(price_date, maturity)
        coupons = [
            (100 * coupon / 2, (paid - price_date).days)
            for paid in coupon_dates
            if paid <= end
        ]

        return (
            bond_data_model.get_treasury_accrued_interest(coupon, price_date, maturity),
            bond_data_model.get_treasury_accrued_interest(coupon, end, maturity),
            coupons,
        )

    @handle_errors
    def get_treasury_forward_prices(
        self,
        forward_date: str | None = None,
        repo_rate: float | None = None,
        date: str | None = None,
        security_type: list[str] | str | None = None,
        bonds: list[str] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the forward price of every US Treasury bill, note and bond: the price for
        delivery at a later date at which buying the security today with money borrowed in
        the repo market and selling it forward breaks even (Tuckman and Serrat, 2022, Fixed
        Income Securities, ch. 13). The difference with today's price is the carry: the
        coupon income less the cost of financing the position.

        The formula is as follows:

        - Forward Price = (P + AI_0) * (1 + r * d / 360) - Σ_k c_k * (1 + r * (d - d_k) / 360) - AI_T
        - Carry = Price - Forward Price
        - Forward Yield: the yield to maturity, compounded semi-annually, at the forward price
          on the forward date (see `get_yield_to_maturity`; for a bill its bond-equivalent
          yield)

        where P is the clean price, AI_0 and AI_T the accrued interest (actual/actual) on the
        price date and the forward date, r the repo rate (simple interest, actual/360), d the
        days to the forward date and c_k the coupons paid d_k days from the price date, which
        earn the repo rate until the forward date. A positive carry (coupon income above the
        repo cost) makes the forward price lower than the price. The implied repo rate of a
        quoted forward price inverts this formula (see `bond_futures_model.get_implied_repo_rate`).

        QuantLib's BondForward is equal without a coupon in between and within 0.0001 per 100
        with one, as QuantLib discounts the coupon on the repo curve rather than reinvesting it
        at simple interest.

        Also known as: bond forward price, repo carry, forward drop, cost of carry, forward
        yield, bond forward.

        Referenced against:
        - QuantLib, BondForward: https://www.quantlib.org/reference/class_quant_lib_1_1_bond_forward.html

        Shares similar logic:
        - rateslib, FixedRateBond.fwd_from_repo: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.FixedRateBond.html#rateslib.instruments.FixedRateBond.fwd_from_repo

        Args:
            forward_date (str | None, optional): The forward (delivery) date (YYYY-MM-DD).
                Defaults to None, three months after the price date.
            repo_rate (float | None, optional): The repo rate (decimal, simple interest on
                actual/360) to the forward date. Defaults to None, which uses the tri-party
                general collateral rate (TGCR) of the price date, the overnight rate of repo
                against Treasuries, held for the whole term.
            date (str | None, optional): The day of the FedInvest prices (YYYY-MM-DD); a weekend
                or holiday uses the business day before it. Defaults to None, the latest day
                with prices.
            security_type (list[str] | str | None, optional): The kind of security, or several. Options
                are: "Bill", "Note", "Bond", None (all three). Defaults to None.
            bonds (list[str] | str | None, optional): Keeps the securities with these CUSIPs
                (or their US ISINs). Defaults to None, every security.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per security maturing after the forward date, indexed by
            CUSIP, with the Security Type, Coupon (decimal), Maturity, the Price, Forward Price
            and Carry per 100 of par, the Forward Yield and Repo Rate (decimals), the Forward
            Date and the Price Date.

        Notes:
        - The overnight TGCR stands in for the term repo rate; a security in demand trades
          special, at a lower repo rate, which raises its forward price.
        - TIPS and floating rate notes are left out: their coupons are not fixed in dollars.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        forwards = fixedincome.get_treasury_forward_prices(
            forward_date="2027-02-19", date="2026-11-19", security_type="Note"
        )

        forwards.loc[
            forwards["Maturity"].between("2036-01-01", "2036-12-31"),
            ["Maturity", "Price", "Forward Price", "Carry", "Forward Yield", "Repo Rate"],
        ]
        ```

        Which returns:

        | CUSIP     | Maturity   |   Price |   Forward Price |   Carry |   Forward Yield |   Repo Rate |
        |:----------|:-----------|--------:|----------------:|--------:|----------------:|------------:|
        | 91282CPZ8 | 2036-02-15 | 93.3438 |         93.2471 |  0.0967 |          0.0507 |      0.0388 |
        | 91282CQQ7 | 2036-05-15 | 95.0312 |         94.8621 |  0.1691 |          0.0508 |      0.0388 |
        | 91282CRF0 | 2036-08-15 | 96.8125 |         96.6263 |  0.1862 |          0.0508 |      0.0388 |
        | 91282CRX1 | 2036-11-15 | 99.6562 |         99.3742 |  0.282  |          0.0508 |      0.0388 |
        """
        types = helpers.split_values(security_type) or ["Bill", "Note", "Bond"]
        if unknown := [
            kind for kind in types if kind.lower() not in ("bill", "note", "bond")
        ]:
            raise ValueError(
                f"Forward prices are calculated for bills, notes and bonds, not {', '.join(unknown)}."
            )
        prices, price_date = self._treasury_prices(date, types, bonds)
        end = (
            datetime.strptime(forward_date, "%Y-%m-%d").date()
            if forward_date
            else price_date + relativedelta(months=3)
        )
        if end <= price_date:
            raise ValueError(
                f"The forward date {end} must be after the price date {price_date}."
            )
        repo = self._repo_rate(price_date) if repo_rate is None else float(repo_rate)
        days = (end - price_date).days

        prices = prices[prices["Maturity"].dt.date > end].set_index("CUSIP")
        if prices.empty:
            raise ValueError(f"No security matures after the forward date {end}.")

        rows = {}
        for cusip, security in prices.iterrows():
            maturity = security["Maturity"].date()
            if security["Security Type"] == "Bill":
                forward_price = bond_futures_model.get_bond_forward_price(
                    security["Price"], 0.0, 0.0, repo, days
                )
                forward_yield = bond_data_model.get_bill_analytics(
                    (maturity - end).days, forward_price
                )[0]
            else:
                accrued, forward_accrued, coupons = self._carry_terms(
                    security["Coupon"], price_date, end, maturity
                )
                forward_price = bond_futures_model.get_bond_forward_price(
                    security["Price"], accrued, forward_accrued, repo, days, coupons
                )
                forward_yield = bond_data_model.get_bond_analytics(
                    security["Coupon"],
                    bond_data_model.get_years_to_maturity(end, maturity),
                    forward_price,
                )[0]
            rows[cusip] = {
                "Security Type": security["Security Type"],
                "Coupon": security["Coupon"],
                "Maturity": security["Maturity"].strftime("%Y-%m-%d"),
                "Price": security["Price"],
                "Forward Price": forward_price,
                "Carry": security["Price"] - forward_price,
                "Forward Yield": forward_yield,
                "Repo Rate": repo,
                "Forward Date": str(end),
                "Price Date": str(price_date),
            }

        forwards = pd.DataFrame.from_dict(rows, orient="index").sort_values(
            ["Maturity", "Security Type"], kind="stable"
        )
        forwards.index.name = "CUSIP"

        return apply_rounding(
            forwards, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_treasury_futures_basis(
        self,
        contract: str = "ZN",
        contract_month: str | None = None,
        repo_rate: float | None = None,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the conversion factor, basis and implied repo rate of every Treasury that
        can be delivered into a CME US Treasury futures contract: what delivering each costs
        and earns against the futures price, and which of them is the cheapest to deliver.

        The seller of a Treasury futures contract delivers a Treasury of its choice from a
        basket of eligible ones and receives the futures price times that security's
        conversion factor, plus accrued interest. The factors put securities of different
        coupons and maturities on one scale, but not exactly, so one of them is the cheapest
        to deliver: the one whose purchase today, financed in repo and delivered into the
        futures, earns the most (the highest implied repo rate). The futures price tracks
        that security (Burghardt and Belton, 2005, The Treasury Bond Basis; Hull, 2022,
        Options, Futures and Other Derivatives, ch. 6.2).

        The formula is as follows:

        - Conversion Factor = a * (c / 2 + C + d) - b: the price per 1 of par at a 6% yield on
          the first day of the delivery month, with a = 1 / 1.03^(v / 6), b = (c / 2) *
          (6 - v) / 6, C = 1 / 1.03^(2n) (or 1.03^(2n + 1) from 7 months), d = (c / 0.06) *
          (1 - C), n and z the whole years and months to maturity and v = z below 7 months
          (CME Group, "Calculating U.S. Treasury Futures Conversion Factors")
        - Invoice Price = F * CF + AI_T
        - Gross Basis = P - F * CF
        - Carry = P - Forward Price, the coupon income less the repo cost to delivery
          (see `get_treasury_forward_prices`)
        - Net Basis = Gross Basis - Carry
        - Implied Repo Rate = (F * CF + AI_T + Σ_k c_k - (P + AI_0)) /
          ((P + AI_0) * d / 360 - Σ_k c_k * (d - d_k) / 360)

        where c is the coupon, P the clean price (FedInvest), F the futures price, AI_0 and
        AI_T the accrued interest on the price date and the delivery date, d the days to the
        delivery date and c_k the coupons paid before it. The months to maturity are rounded
        down to whole quarters for the 10-year (ZN), Ultra 10-year (TN), bond (ZB) and Ultra
        bond (UB) contracts and to whole months for the 2-year (ZT) and 5-year (ZF) ones.
        Rates are simple interest on actual/360. The cheapest to deliver has the highest
        implied repo rate and the lowest net basis.

        The conversion factors equal those of FinancePy's BondFuture and rateslib's BondFuture
        (CME conventions) and Hull's examples (1.4623 for a 10% coupon with 20 years and 2
        months).

        See definition: https://www.cmegroup.com/trading/interest-rates/treasury-conversion-factors.html

        Also known as: cheapest to deliver, CTD, Treasury basis, implied repo rate, IRR,
        conversion factor, net basis, gross basis, invoice price, delivery basket.

        Referenced against:
        - rateslib, BondFuture.cfs: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.BondFuture.html#rateslib.instruments.BondFuture.cfs

        Shares similar logic:
        - FinancePy, BondFuture.gross_basis: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_future.py#L209
        - FinancePy, BondFuture.net_basis: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_future.py#L237
        - rateslib, BondFuture.gross_basis: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.BondFuture.html#rateslib.instruments.BondFuture.gross_basis
        - rateslib, BondFuture.net_basis: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.BondFuture.html#rateslib.instruments.BondFuture.net_basis
        - rateslib, BondFuture.implied_repo: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.BondFuture.html#rateslib.instruments.BondFuture.implied_repo

        Args:
            contract (str, optional): The futures contract. Options are: "ZT" (2-year note), "ZF"
                (5-year note), "ZN" (10-year note), "TN" (Ultra 10-year note), "ZB" (bond), "UB"
                (Ultra bond). Defaults to "ZN".
            contract_month (str | None, optional): The delivery month (YYYY-MM), March, June,
                September or December. Defaults to None, the first of those months that starts
                after the price date.
            repo_rate (float | None, optional): The repo rate (decimal, simple interest on
                actual/360) to the delivery date, for the carry and net basis. Defaults to
                None, the tri-party general collateral rate (TGCR) of the price date.
            date (str | None, optional): The day of the FedInvest prices (YYYY-MM-DD); a weekend
                or holiday uses the business day before it. Defaults to None, the latest day
                with prices.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per deliverable, indexed by CUSIP and sorted from the highest
            implied repo rate, with the Coupon (decimal), Maturity, Price, Conversion Factor,
            Futures Price, Invoice Price, Gross Basis, Carry and Net Basis (per 100 of par),
            the Implied Repo Rate and Repo Rate (decimals), whether it is the Cheapest to
            Deliver, the Futures Symbol, Delivery Date and Price Date.

        Notes:
        - The basket follows CME's contract rules, applied to the FedInvest securities: ZT
          (original term up to 5 years 3 months, from 1 year 9 months remaining on the first
          day of the delivery month to 2 years on its last day), ZF (original term up to 5
          years 3 months, at least 4 years 2 months remaining), ZN (notes with 6 years 6 months
          to 8 years remaining), TN (original 10-year notes with 9 years 5 months to 10 years),
          ZB (bonds with 15 to 25 years) and UB (bonds with 25 years or more). The original
          terms come from TreasuryDirect's auction data. CME publishes the basket itself; a
          security issued after the price date is not in it here.
        - The futures price is the settlement price of the contract month from Yahoo Finance
          (such as ZNZ26.CBT) on the price date, or the last day before it. FedInvest prices
          are not taken at the time of the futures settlement, so on a day the market moves a
          lot the basis carries part of that move. Before its end-of-day prices are published
          (the next business day) the latest day uses the bid.
        - Delivery is taken on the last delivery day: the last business day of the delivery
          month for ZN, TN, ZB and UB, the third business day of the next month for ZT and
          ZF. The short's options on what and when to deliver are left out, so the net basis
          of the cheapest to deliver is the value the market puts on them. Business days are
          weekdays, without the holidays.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        basis = fixedincome.get_treasury_futures_basis(contract="ZN", contract_month="2026-12", date="2026-11-19")

        basis[["Maturity", "Price", "Conversion Factor", "Gross Basis", "Net Basis", "Implied Repo Rate"]]
        ```

        Which returns:

        | CUSIP     | Maturity   |    Price |   Conversion Factor |   Gross Basis |   Net Basis |   Implied Repo Rate |
        |:----------|:-----------|---------:|--------------------:|--------------:|------------:|--------------------:|
        | 91282CRJ2 | 2033-08-31 |  97.5    |              0.9202 |        0.0738 |     -0.0024 |              0.039  |
        | 91282CHT1 | 2033-08-15 |  93.9688 |              0.887  |        0.0576 |      0.0453 |              0.0347 |
        | 91282CRC7 | 2033-07-31 |  96.8438 |              0.9136 |        0.1163 |      0.0614 |              0.0334 |
        | 91282CRM5 | 2033-09-30 | 100.344  |              0.9451 |        0.2813 |      0.1617 |              0.0251 |
        | 91282CQW4 | 2033-06-30 |  96.2188 |              0.9069 |        0.2007 |      0.1586 |              0.0249 |
        | 91282CJJ1 | 2033-11-15 |  97.4062 |              0.9177 |        0.2448 |      0.1638 |              0.0244 |
        | 91282CRV5 | 2033-10-31 |  99.625  |              0.9382 |        0.2931 |      0.1796 |              0.0234 |
        | 91282CJZ5 | 2034-02-15 |  94.25   |              0.887  |        0.3389 |      0.3137 |              0.0106 |
        | 91282CKQ3 | 2034-05-15 |  96.375  |              0.9055 |        0.5052 |      0.4341 |              0.0002 |
        | 91282CLF6 | 2034-08-15 |  93      |              0.8732 |        0.5499 |      0.5333 |             -0.0098 |
        | 91282CLW9 | 2034-11-15 |  95.1875 |              0.8927 |        0.6729 |      0.6109 |             -0.0162 |
        """
        key = contract.strip().upper()
        if key not in TREASURY_FUTURES:
            raise ValueError(
                f"The contract {contract} is not a Treasury futures contract. Choose from "
                f"{', '.join(TREASURY_FUTURES)}."
            )
        terms = TREASURY_FUTURES[key]

        prices, price_date = self._treasury_prices(date, terms["security_type"], None)
        if contract_month is None:
            # The first delivery month that starts after the price date.
            first_day = price_date.replace(day=1) + relativedelta(months=1)
            while first_day.month % 3:
                first_day += relativedelta(months=1)
        else:
            first_day = datetime.strptime(contract_month, "%Y-%m").date()
            if first_day.month not in (3, 6, 9, 12):
                raise ValueError(
                    f"Treasury futures deliver in March, June, September and December, not in "
                    f"{contract_month}."
                )
        last_day = first_day + relativedelta(day=31)

        # The last delivery day: the last business day of the month, or for the 2- and
        # 5-year notes the third business day of the next month.
        delivery = last_day
        while delivery.weekday() >= 5:  # noqa: PLR2004
            delivery -= timedelta(days=1)
        if terms["delivery"] == "third business day":
            remaining = 3
            while remaining:
                delivery += timedelta(days=1)
                if delivery.weekday() < 5:  # noqa: PLR2004
                    remaining -= 1
        if delivery <= price_date:
            raise ValueError(
                f"The {key} contract of {first_day:%Y-%m} has delivered by {price_date}."
            )

        maturities = prices["Maturity"].dt.date
        low, high = terms["remaining"]
        eligible = maturities >= first_day + relativedelta(months=low)
        if high is not None:
            high_months, measured_from, bound = high
            limit = (
                last_day if measured_from == "last" else first_day
            ) + relativedelta(months=high_months)
            eligible &= maturities < limit if bound == "below" else maturities <= limit
        if terms["original_terms"]:
            originals = pd.concat(
                [
                    treasurydirect_model.get_auctioned_securities("Note", term)
                    for term in terms["original_terms"]
                ]
            )
            eligible &= prices["CUSIP"].isin(originals["CUSIP"])
        basket = prices[eligible].set_index("CUSIP")
        if basket.empty:
            raise ValueError(
                f"No Treasury on {price_date} is deliverable into the {key} contract of "
                f"{first_day:%Y-%m}."
            )

        symbol = f"{key}{FUTURES_MONTH_CODES[first_day.month]}{first_day:%y}.CBT"
        history, _ = historical_model.get_historical_data(
            [symbol],
            api_key=self._api_key or None,
            enforce_source="YahooFinance",
            start=str(price_date - timedelta(days=10)),
            end=str(price_date),
            fill_nan=False,
            show_ticker_separation=False,
            cache=self._cache,
        )
        closes = (
            history["Close"][symbol].dropna()
            if "Close" in history and symbol in history["Close"]
            else pd.Series(dtype=float)
        )
        closes = closes[closes.index <= pd.Period(price_date, freq="D")]
        if closes.empty:
            raise ValueError(
                f"No futures price of {symbol} is available up to {price_date}."
            )
        futures_price = float(closes.iloc[-1])

        repo = self._repo_rate(price_date) if repo_rate is None else float(repo_rate)
        days = (delivery - price_date).days

        rows = {}
        for cusip, security in basket.iterrows():
            maturity = security["Maturity"].date()
            remaining = relativedelta(maturity, first_day)
            factor = bond_futures_model.get_conversion_factor(
                security["Coupon"],
                remaining.years,
                remaining.months,
                terms["rounding"],
            )
            accrued, delivery_accrued, coupons = self._carry_terms(
                security["Coupon"], price_date, delivery, maturity
            )
            basis = bond_futures_model.get_futures_basis(
                security["Price"],
                accrued,
                futures_price,
                factor,
                delivery_accrued,
                repo,
                days,
                coupons,
            )
            rows[cusip] = {
                "Coupon": security["Coupon"],
                "Maturity": maturity.strftime("%Y-%m-%d"),
                "Price": security["Price"],
                "Conversion Factor": factor,
                "Futures Price": futures_price,
                **basis,
                "Repo Rate": repo,
            }

        table = pd.DataFrame.from_dict(rows, orient="index").sort_values(
            "Implied Repo Rate", ascending=False, kind="stable"
        )
        table["Cheapest to Deliver"] = table.index == table.index[0]
        table["Futures Symbol"] = symbol
        table["Delivery Date"] = str(delivery)
        table["Price Date"] = str(price_date)
        table.index.name = "CUSIP"

        return apply_rounding(
            table, rounding if rounding is not None else self._rounding
        )
