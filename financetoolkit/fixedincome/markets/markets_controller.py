"""Bond Markets Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd
import requests

from financetoolkit import (
    fmp_model as toolkit_fmp_model,
    helpers,
)
from financetoolkit.fixedincome import helpers as fixedincome_helpers
from financetoolkit.fixedincome.curves.curves_controller import CurvesFixedIncome
from financetoolkit.fixedincome.markets import bond_data_model
from financetoolkit.fixedincome.sources import (
    fiscaldata_model,
    treasurydirect_model,
)
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
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
            security_type (list[str] | str | None, optional): The kind of security: "Bill",
                "Cash Management Bill", "Note", "Bond", "TIPS" or "FRN", or several (a list or
                a comma-separated string). Defaults to None, which returns every kind.
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
        bond held by several funds is listed once, with the funds that hold it.

        The formula is as follows:

        - Price = Market Value / Par Amount × 100 (a clean price per 100 of par)
        - Yield to Maturity: the rate y at which the coupons and principal, discounted at
          (1 + y / 2) per half year, are worth the clean price plus accrued interest
          (see `get_yield_to_maturity`)
        - Modified Duration = Macaulay Duration / (1 + y / 2) (see `get_duration`)
        - G-Spread = Yield to Maturity - Treasury Par Yield at the bond's maturity

        Coupons are semi-annual, the US convention for corporate bonds, and the yield is a
        nominal annual rate compounded twice a year (bond-equivalent), like the Treasury par
        yields it is compared with. The coupon periods fall on the bond's actual coupon dates,
        counted back from maturity, and the time in the current period is counted in actual
        days. The Treasury par yield at a maturity is interpolated linearly between the
        maturities of the US Treasury par yield curve on the price date (held at the 1-month
        and 30-year yield beyond them).

        Also known as: corporate bond prices, bond yields, bond screener, credit spreads per
        bond, G-spread.

        Args:
            etfs (list[str] | str | None, optional): The bond ETFs to read the holdings of, as
                a list or a comma-separated string. The holdings must name each bond as
                "ISSUER COUPON% MM/DD/YYYY" with a par amount, as iShares funds do. Defaults
                to None, which uses LQD, IGSB and USHY.
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
        - The yield is to maturity. A callable bond that trades above its call price
          yields less to its call date (the yield to worst), which is common among high
          yield bonds; call schedules are not in the holdings.
        - The fund prints the coupon to two decimals (2.125% shows as 2.13%), which moves the
          yield by well under a basis point a year of maturity.
        - Left out, with a count in the log: lines that are not fixed-coupon bonds (cash,
          futures, swaps), floating rate notes, bank notes marked fixed-to-floating
          ("FXD-FRN", which switch to a floating coupon before maturity), inflation-linked
          bonds, mortgage pools, perpetual bonds, bonds without a US ISIN (the funds of
          other currencies) and prices outside 1 to 300. A fixed-to-floating note the fund
          does not mark as such is kept, and its yield assumes the fixed coupon to maturity.
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
        | US037833EC07 |   0.012  | 2028-02-08 |   95.96 |              0.0454 |              1.2213 |     0.0021 |
        | US037833ET32 |   0.04   | 2028-05-10 |   99.12 |              0.0461 |              1.4224 |     0.0024 |
        | US037833EY27 |   0.04   | 2028-05-12 |   99.09 |              0.0463 |              1.4275 |     0.0025 |
        | US037833EH93 |   0.014  | 2028-08-05 |   94.67 |              0.046  |              1.6896 |     0.0019 |
        | US037833FD70 |   0.0465 | 2028-11-09 |   99.95 |              0.0468 |              1.8564 |     0.0022 |
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

        # Fund families count the par amount differently: iShares and Vanguard in hundreds
        # (market value / amount is the price per 100), Schwab in dollars (the price per 1).
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
            isin = str(holding["ISIN"] or "") or bond_data_model.cusip_to_isin(
                holding["CUSIP"]
            )
            # A bond without a US ISIN is a foreign security, such as the euro bonds of a
            # European fund, whose coupons and currency differ.
            if reason is None and isin and not isin.startswith("US"):
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
                    # A fund that names its bonds without identifiers keeps them apart by name.
                    "ISIN": isin or holding["Name"],
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
                f"The holdings of {', '.join(funds)} hold no US fixed-coupon bonds."
            )

        bonds = pd.DataFrame(records)
        funds_held = bonds.groupby("ISIN", sort=False)["Fund"].agg(
            lambda values: ", ".join(dict.fromkeys(values))
        )
        bonds = bonds.drop_duplicates("ISIN").set_index("ISIN")
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

        bonds["Years"] = [
            bond_data_model.get_years_to_maturity(settlement, maturity)
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

        try:
            curve = self._latest_yield_curve(
                "United States", date=str(max(bonds["Price Date"]))
            )
            bonds["G-Spread"] = bonds[
                "Yield to Maturity"
            ] - bond_data_model.interpolate_curve(curve, bonds["Years"])
        except (ValueError, requests.exceptions.RequestException) as error:
            logger.warning("The G-Spread is left empty: no Treasury curve (%s).", error)
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
        | APPLE INC         | 0.0017 | 0.0029 | 0.0052 |      43 |     1.2473 |   35.7473 |
        | BOEING CO         | 0.0062 | 0.0083 | 0.0093 |      20 |     1.2283 |   37.4779 |
        | INTEL CORPORATION | 0.0081 | 0.0112 | 0.0149 |      35 |     1.2527 |   36.2527 |
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
                    pd.Timestamp(settlement).date(), pd.Timestamp(maturity).date()
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
        API key.

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

        Coupons are semi-annual and fall on the actual coupon dates, counted back from
        maturity; the time in the current period is counted in actual days (actual/actual,
        the Treasury convention). Yields are nominal annual rates compounded twice a year
        (bond-equivalent), as the Treasury quotes them. For TIPS the price is per 100 of
        inflation-adjusted principal, so the yield is the real yield. Floating rate notes
        reset their coupon every week, so they have a price but no yield or duration.

        See definition: https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate

        Also known as: Treasury bond prices, Treasury yields per security, CUSIP yields,
        off-the-run Treasuries, FedInvest prices.

        Args:
            date (str | None, optional): The day (YYYY-MM-DD); a weekend or holiday uses the
                business day before it. Defaults to None, the latest day with prices.
            security_type (list[str] | str | None, optional): The kind of security: "Bill",
                "Note", "Bond", "TIPS" or "FRN", or several (a list or a comma-separated
                string). Defaults to None, which returns every kind.
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
        if prices.empty:
            raise ValueError(
                f"FedInvest lists no security of the type and CUSIPs given on {price_date}."
            )

        def analytics(security: pd.Series) -> tuple[float, float]:
            maturity = security["Maturity"].date()
            if security["Security Type"] == "FRN":
                return np.nan, np.nan
            if security["Security Type"] == "Bill":
                return bond_data_model.get_bill_analytics(
                    (maturity - price_date).days, security["Price"]
                )
            return bond_data_model.get_bond_analytics(
                security["Coupon"],
                bond_data_model.get_years_to_maturity(price_date, maturity),
                security["Price"],
            )

        prices = prices[prices["Maturity"].dt.date > price_date].copy()
        values = prices.apply(analytics, axis=1, result_type="expand")
        prices["Yield to Maturity"] = values[0]
        prices["Modified Duration"] = values[1]
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
