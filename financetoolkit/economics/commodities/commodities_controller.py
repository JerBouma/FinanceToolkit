"""Commodity Curves Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit import yfinance_model
from financetoolkit.economics.helpers import (
    buffered_start_date,
    get_exchange_business_days,
)
from financetoolkit.economics.sources import futures_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import finalize_dataset

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class CommoditiesEconomics:
    """
    Commodity futures curves: the forward curve and the term structure.
    """

    def _get_commodity_curves(
        self,
        commodity: str | list[str] | None,
        contracts: int,
        generic: bool,
        start_date: str,
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Collects the forward curve of each commodity with the contract behind every price:
        the listed contracts as columns, or, with generic, the positions on the curve of
        each day (see futures_model.get_generic_curve).

        Yahoo Finance drops a contract once it expires, so every expired month of the
        commodity's listing cycle (for gold and silver the active cycle, which the generic
        series rolls through) that it does not return is added as a contract without
        prices, with its last trading day from the contract specification: the positions
        after it stay right and its own is NaN. A month that has not expired and is not
        returned is left out: Yahoo Finance carries the other gold and silver months only
        shortly before delivery.

        Returns:
            dict[str, tuple[pd.DataFrame, pd.DataFrame]]: Per commodity the prices and the
            delivery month of the contract behind each of them.
        """
        names = (
            [name.strip() for name in commodity.split(",") if name.strip()]
            if isinstance(commodity, str)
            else list(commodity or yfinance_model.COMMODITY_TICKERS)
        )
        curves: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

        for name in names:
            forward_curve = yfinance_model.get_commodity_forward_curve(
                name, start_date, self._end_date, contracts
            )

            if forward_curve.empty:
                continue

            if not generic:
                labels = pd.DataFrame(
                    [forward_curve.columns.to_list()] * len(forward_curve),
                    index=forward_curve.index,
                    columns=forward_curve.columns,
                )
                curves[name] = (forward_curve, labels)
                continue

            first_month = pd.Period(start_date, freq="M")
            last_month = pd.Period(max(forward_curve.columns), freq="M")
            # Crude oil and natural gas stop trading in the month before delivery.
            business_days = get_exchange_business_days(
                (pd.Timestamp(start_date) + pd.DateOffset(months=-2)).strftime(
                    "%Y-%m-%d"
                ),
                (pd.Timestamp(f"{last_month}-01") + pd.DateOffset(months=2)).strftime(
                    "%Y-%m-%d"
                ),
            )
            rule = futures_model.LAST_TRADING_DAY_RULES[name]
            last_trading_days = pd.Series(
                {
                    str(month): futures_model.get_last_trading_day(
                        rule, month, business_days
                    )
                    for month in pd.period_range(first_month, last_month, freq="M")
                    if month.month in futures_model.CONTRACT_MONTHS[name]
                    or str(month) in forward_curve.columns
                }
            )
            today = pd.Timestamp.today().normalize()
            # Only the months of the active cycle are added back: a placeholder has no first
            # price, so it counts as listed on every date and an off-cycle month would take a
            # position on dates before it traded.
            active_months = futures_model.ACTIVE_MONTHS.get(
                name, futures_model.CONTRACT_MONTHS[name]
            )
            expired = [
                label
                for label, last_trading_day in last_trading_days.items()
                if last_trading_day < today
                and label not in forward_curve.columns
                and pd.Period(label, freq="M").month in active_months
            ]
            forward_curve = forward_curve.reindex(
                columns=sorted([*forward_curve.columns, *expired])
            )
            curves[name] = futures_model.get_generic_curve(
                forward_curve, last_trading_days, contracts
            )

        return curves

    @handle_errors
    def get_commodity_forward_curve(
        self,
        commodity: str | list[str] | None = None,
        contracts: int = 12,
        generic: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the futures curve of a commodity from Yahoo Finance: the daily closing (settlement) price of each of its
        nearest dated contracts, the price the market sets for delivery in each month.

        The contracts run from the current month on, such as Crude Oil's December 2026 and January 2027 contracts,
        rather than a single continuous or spot price.

        This is what a Schwartz-Smith (2000) two-factor commodity price model needs
        to back out the convenience-yield term structure under the risk-neutral (Q)
        measure -- the curve's shape (contango or backwardation) at each point in
        time is exactly what a single spot price series cannot reveal.

        Not every commodity has a listed contract for every calendar month (grains
        in particular only trade specific delivery months), and the contracts of the
        first months may already have expired (for crude oil the current month's
        contract and, late in the month, the next month's). Those months are skipped and
        later months are tried instead, until the requested number of contracts is found.

        By default the columns are the contracts listed today, so a past date shows how
        those contracts traded then, not the curve as it stood that day. With the
        generic curve the columns are the positions on each day's curve instead:
        the front (the nearest contract that has not expired on that day), +1, +2 and so on,
        rolling to the next contract the day after the front's last trading day (the
        generic or nearby series, CL1, CL2, ... in Bloomberg notation). A position is a
        rank on the curve, not a constant maturity: the front's time to expiry shortens
        each day and jumps at the roll, as no price is interpolated. The last trading days
        follow the CME Group contract specifications on the US exchange calendar.

        Yahoo Finance only publishes the contracts that still trade: a contract's history
        disappears once it expires, and no free source publishes it (FinancialModelingPrep
        has the continuous front contract only). On a past date the positions of the
        contracts that have expired since are therefore NaN rather than filled with
        another contract: for crude oil on 2026-09-15 the front (October 2026) and +1
        (November 2026) are NaN and +2 is the December 2026 contract. The further back the
        date, the more positions are NaN, until none is left about as many months back as there are contracts.

        With growth the change is that of each column over the lag in trading days: of a
        contract, or with the generic curve of a position, which compares two contracts
        across a roll.

        Also known as: futures term structure, forward curve, generic futures, nearby
        futures.

        Args:
            commodity (str | list[str] | None, optional): The commodity, or a list of them, to retrieve the curve for.
                Options are: "Crude Oil" (WTI crude oil), "Natural Gas" (Henry Hub natural gas), "Gold" (gold),
                "Silver" (silver), "Copper" (copper), "Corn" (corn), "Wheat" (wheat), "Soybeans" (soybeans). Defaults
                to None, which retrieves every one of them.
            contracts (int, optional): The number of contracts to return, the nearest
                listed ones from the current month on, or with generic the
                number of positions (the front included). Defaults to 12.
            generic (bool, optional): Whether the columns are the positions on
                each day's curve (Front, +1, +2, ...) instead of the contracts listed
                today. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
            growth (bool, optional): Whether to return the change of each column instead
                of the price. Defaults to False.
            lag (int | list[int], optional): The number of trading days to calculate the
                change over. Defaults to 1.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            InvalidArgumentError: If the commodity is not one of the supported names.

        Returns:
            pd.DataFrame: A DataFrame indexed by date, with one column per contract
            labeled by its delivery month (e.g. "2026-12"), containing that
            contract's daily closing price over its trading life, or with
            generic one column per position ("Front", "+1", ...). Columns are NaN
            outside the date range the contract actually traded in. With a list of
            commodities (or none given), the columns are the commodity and the contract.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-10-13', end_date='2026-10-23')

        economics.get_commodity_forward_curve(
            "Crude Oil", contracts=4, generic=True
        )
        ```

        Which returns:

        | Date       |   Front |     +1 |    +2 |    +3 |
        |:-----------|--------:|-------:|------:|------:|
        | 2026-10-13 |  nan    |  93.17 | 92.23 | 91.15 |
        | 2026-10-14 |  nan    |  93.9  | 92.83 | 91.53 |
        | 2026-10-15 |  nan    |  96.5  | 94.89 | 93.02 |
        | 2026-10-16 |  nan    |  99.02 | 97.44 | 95.48 |
        | 2026-10-19 |  nan    |  98.54 | 96.67 | 94.69 |
        | 2026-10-20 |  nan    |  97.82 | 96.02 | 94.05 |
        | 2026-10-21 |   97.15 |  95.46 | 93.71 | 91.99 |
        | 2026-10-22 |   99.47 |  97.52 | 95.35 | 93.08 |
        | 2026-10-23 |  103.27 | 100.94 | 98.36 | 95.68 |
        """
        start_date = (
            buffered_start_date(self._start_date, "daily")
            if growth
            else self._start_date
        )
        curves = self._get_commodity_curves(commodity, contracts, generic, start_date)

        # No commodity has prices (Yahoo Finance unreachable), which the model logged.
        if not curves:
            return pd.DataFrame()

        # Each commodity's changes are taken over its own trading days.
        commodity_forward_curve = {
            name: finalize_dataset(
                dataset=prices,
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                growth=growth,
                lag=lag,
                rounding=rounding,
                axis="rows",
                row_slice=True,
                dropna=generic,
            )
            for name, (prices, _) in curves.items()
        }

        if isinstance(commodity, str) and "," not in commodity:
            return commodity_forward_curve.get(commodity.strip(), pd.DataFrame())

        return pd.concat(commodity_forward_curve, axis=1)

    @handle_errors
    def get_commodity_term_structure(
        self,
        commodity: str | list[str] | None = None,
        generic: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the term structure of a commodity futures curve per day: whether the curve is in backwardation, contango
        or flat, with the 12-month calendar spread, the annualised roll yield and the return of rolling the front into
        the next contract.

        Next to the front contract, the calendar spread compares it with the last contract of the 12-month strip that
        starts with it, in price and in percent.

        The far contract is the furthest one delivering less than 12 months after the
        front: for a December 2026 front of a monthly listed commodity (crude oil, natural
        gas, the metals) the November 2027 contract, 11 months later, and for corn's
        December 2026 front the September 2027 contract, 9 months later.

        The formulas are as follows:

            Spread = Front - Far
            Slope = Far / Front - 1
            Roll Yield = ln(Front / Far) x 12 / Months
            Roll Return = Front / Next - 1

        The spread follows the exchange quote of a calendar spread, nearby minus deferred,
        so it is positive in backwardation, and its widest backwardation is its maximum.
        The slope is the percentage the far contract lies above the front (negative in
        backwardation). The roll yield is the annualised log basis of Fama and French
        (1987) and Szymanowska, De Roon, Nijman and Van den Goorbergh (2014): the
        continuously compounded yearly return a long position earns from the shape of the
        curve alone, positive in backwardation and negative in contango, with Months the
        months between both delivery months. The roll return is the simple return of one
        roll, not annualised (Erb and Harvey, 2006): what a long position earns by selling
        the front and buying the next contract when the curve keeps its shape until the next
        contract becomes the front, positive in backwardation and negative in contango. It
        covers the months between the front and the next contract (one for crude oil, two or
        three for corn and the other grains).

        For more information, see the following papers:

        - Fama, E. F., and French, K. R. (1987). "Commodity Futures Prices: Some Evidence on
          Forecast Power, Premiums, and the Theory of Storage." Journal of Business, 60(1), 55-73.
        - Erb, C. B., and Harvey, C. R. (2006). "The Strategic and Tactical Value of Commodity
          Futures." Financial Analysts Journal, 62(2), 69-97.
        - Szymanowska, M., De Roon, F., Nijman, T., and Van den Goorbergh, R. (2014). "An
          Anatomy of Commodity Futures Risk Premia." Journal of Finance, 69(1), 453-482.

        By default the front of each day is the nearest contract that had not expired on
        that day (see the generic curve of get_commodity_forward_curve), so the
        spread is a time series of the curve as it stood each day. Yahoo Finance drops a
        contract once it expires, so on the days whose front has expired since, Front, the
        Spread, Slope, Roll Yield and Roll Return are NaN while Front Contract names the contract.
        Without the generic curve, today's front and far contract are taken on every day
        (e.g. December 2026 against November 2027 for crude oil): a fixed calendar spread
        with its full history.

        Also known as: calendar spread, roll yield, roll return, carry, basis, backwardation, contango.

        Args:
            commodity (str | list[str] | None, optional): The commodity, or a list of them. Options are: "Crude Oil"
                (WTI crude oil), "Natural Gas" (Henry Hub natural gas), "Gold" (gold), "Silver" (silver), "Copper"
                (copper), "Corn" (corn), "Wheat" (wheat), "Soybeans" (soybeans). Defaults to None, which retrieves
                every one of them.
            generic (bool, optional): Whether the front and far contract are
                those of each day's curve, instead of today's. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            InvalidArgumentError: If the commodity is not one of the supported names.

        Returns:
            pd.DataFrame: Per date the Front Contract, Next Contract and Far Contract
            (delivery months), their prices Front, Next and Far, the Months between the front
            and far contract, the Spread, Slope, Roll Yield and Roll Return (decimals) and the Structure ("Backwardation", "Contango" or "Flat" when
            the front and far contract trade at the same price). With a list
            of commodities (or none given), the columns are the commodity and the field.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-11-23', end_date='2026-11-27')

        economics.get_commodity_term_structure("Crude Oil")
        ```

        Which returns:

        | Date       | Front Contract   |   Front | Next Contract   |   Next | Far Contract   |   Far |   Months |   Spread |   Slope |   Roll Yield |   Roll Return | Structure     |
        |:-----------|:-----------------|--------:|:----------------|-------:|:---------------|------:|---------:|---------:|--------:|-------------:|--------------:|:--------------|
        | 2026-11-23 | 2027-01          |   80.17 | 2027-02         |  79.07 | 2027-12        | 70.47 |       11 |     9.7  | -0.121  |       0.1407 |        0.0139 | Backwardation |
        | 2026-11-24 | 2027-01          |   82.05 | 2027-02         |  81.07 | 2027-12        | 71.71 |       11 |    10.34 | -0.126  |       0.1469 |        0.0121 | Backwardation |
        | 2026-11-25 | 2027-01          |   83.68 | 2027-02         |  82.35 | 2027-12        | 72.06 |       11 |    11.62 | -0.1389 |       0.1631 |        0.0162 | Backwardation |
        | 2026-11-27 | 2027-01          |   83.08 | 2027-02         |  81.96 | 2027-12        | 72.08 |       11 |    11    | -0.1324 |       0.1549 |        0.0137 | Backwardation |
        """
        curves = self._get_commodity_curves(commodity, 12, generic, self._start_date)

        # No commodity has prices (Yahoo Finance unreachable), which the model logged.
        if not curves:
            return pd.DataFrame()

        term_structures = {}

        for name, (prices, labels) in curves.items():
            term_structure = futures_model.get_term_structure(prices, labels)
            term_structures[name] = finalize_dataset(
                dataset=term_structure[
                    term_structure[["Front", "Far"]].notna().any(axis=1)
                ],
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                rounding=rounding,
                axis="rows",
                row_slice=True,
            )

        if isinstance(commodity, str) and "," not in commodity:
            return term_structures.get(commodity.strip(), pd.DataFrame())

        return pd.concat(term_structures, axis=1)
