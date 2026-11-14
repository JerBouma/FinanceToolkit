"""Commodity Curves Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    buffered_start_date,
    get_exchange_business_days,
)
from financetoolkit.economics.sources import futures_model, yfinance_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import (
    track_sources,
)
from financetoolkit.utilities.statistics_model import (
    finalize_dataset,
)

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class CommoditiesEconomics(EconomicsBase):
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
        Get the forward/futures curve for a commodity from Yahoo Finance -- the
        historical daily closing (settlement) price of each of the nearest `contracts`
        dated futures contracts from the current month on (e.g. Crude Oil's December 2026,
        January 2027, ... contracts), rather than a single flat continuous/spot price.

        This is what a Schwartz-Smith (2000) two-factor commodity price model needs
        to back out the convenience-yield term structure under the risk-neutral (Q)
        measure -- the curve's shape (contango or backwardation) at each point in
        time is exactly what a single spot price series cannot reveal.

        Not every commodity has a listed contract for every calendar month (grains
        in particular only trade specific delivery months), and the contracts of the
        first months may already have expired (for crude oil the current month's
        contract and, late in the month, the next month's). Those months are skipped and
        later months are tried instead, until `contracts` contracts are found.

        By default the columns are the contracts listed today, so a past date shows how
        those contracts traded then, not the curve as it stood that day. With
        generic=True the columns are the positions on each day's curve instead:
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
        date, the more positions are NaN, until none is left about `contracts` months back.

        With growth=True the change is that of each column over `lag` trading days: of a
        contract, or with generic of a position, which compares two contracts
        across a roll.

        Also known as: futures term structure, forward curve, generic futures, nearby
        futures.

        Args:
            commodity (str | list[str] | None, optional): The commodity, or a list of
                them, to retrieve the curve for: "Crude Oil", "Natural Gas", "Gold",
                "Silver", "Copper", "Corn", "Wheat" or "Soybeans". Defaults to None, which
                retrieves every one of them.
            contracts (int, optional): The number of contracts to return, the nearest
                listed ones from the current month on, or with generic the
                number of positions (the front included). Defaults to 12.
            generic (bool, optional): Whether the columns are the positions on
                each day's curve (Front, +1, +2, ...) instead of the contracts listed
                today. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the change of each column instead
                of the price. Defaults to False.
            lag (int | list[int], optional): The number of trading days to calculate the
                change over. Defaults to 1.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If `commodity` is not one of the supported names.

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
        Get the term structure of a commodity futures curve per day: the front contract,
        the 12-month calendar spread between the front and the last contract of the
        12-month strip that starts with it (in price and in percent), the annualised roll
        yield and whether the curve is in backwardation, contango or flat.

        The far contract is the furthest one delivering less than 12 months after the
        front: for a December 2026 front of a monthly listed commodity (crude oil, natural
        gas, the metals) the November 2027 contract, 11 months later, and for corn's
        December 2026 front the September 2027 contract, 9 months later.

        The formulas are as follows:

            Spread = Front - Far
            Slope = Far / Front - 1
            Roll Yield = ln(Front / Far) x 12 / Months

        The spread follows the exchange quote of a calendar spread, nearby minus deferred,
        so it is positive in backwardation, and its widest backwardation is its maximum.
        The slope is the percentage the far contract lies above the front (negative in
        backwardation). The roll yield is the annualised log basis of Fama and French
        (1987) and Szymanowska, De Roon, Nijman and Van den Goorbergh (2014): the
        continuously compounded yearly return a long position earns from the shape of the
        curve alone, positive in backwardation and negative in contango, with Months the
        months between both delivery months.

        By default the front of each day is the nearest contract that had not expired on
        that day (see get_commodity_forward_curve with generic=True), so the
        spread is a time series of the curve as it stood each day. Yahoo Finance drops a
        contract once it expires, so on the days whose front has expired since, Front, the
        Spread, Slope and Roll Yield are NaN while Front Contract names the contract.
        generic=False instead takes today's front and far contract on every day
        (e.g. December 2026 against November 2027 for crude oil): a fixed calendar spread
        with its full history.

        Also known as: calendar spread, roll yield, carry, basis, backwardation, contango.

        Args:
            commodity (str | list[str] | None, optional): The commodity, or a list of
                them: "Crude Oil", "Natural Gas", "Gold", "Silver", "Copper", "Corn",
                "Wheat" or "Soybeans". Defaults to None, which retrieves every one of them.
            generic (bool, optional): Whether the front and far contract are
                those of each day's curve, instead of today's. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Raises:
            ValueError: If `commodity` is not one of the supported names.

        Returns:
            pd.DataFrame: Per date the Front Contract and Far Contract (delivery months),
            their prices Front and Far, the Months between them, the Spread, Slope and Roll
            Yield (decimals) and the Structure ("Backwardation", "Contango" or "Flat" when
            the front and far contract trade at the same price). With a list
            of commodities (or none given), the columns are the commodity and the field.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-10-26', end_date='2026-11-06')

        economics.get_commodity_term_structure("Crude Oil")
        ```

        Which returns:

        | Date       | Front Contract   |   Front | Far Contract   |   Far |   Months |   Spread |   Slope |   Roll Yield | Structure     |
        |:-----------|:-----------------|--------:|:---------------|------:|---------:|---------:|--------:|-------------:|:--------------|
        | 2026-10-26 | 2026-12          |  101.76 | 2027-11        | 81.98 |       11 |    19.78 | -0.1944 |       0.2358 | Backwardation |
        | 2026-10-27 | 2026-12          |   99.9  | 2027-11        | 82.07 |       11 |    17.83 | -0.1785 |       0.2145 | Backwardation |
        | 2026-10-28 | 2026-12          |  102.78 | 2027-11        | 83.15 |       11 |    19.63 | -0.191  |       0.2312 | Backwardation |
        | 2026-10-29 | 2026-12          |  102    | 2027-11        | 82.54 |       11 |    19.46 | -0.1908 |       0.2309 | Backwardation |
        | 2026-10-30 | 2026-12          |  104.12 | 2027-11        | 83.01 |       11 |    21.11 | -0.2027 |       0.2472 | Backwardation |
        | 2026-11-02 | 2026-12          |   96.46 | 2027-11        | 80.05 |       11 |    16.41 | -0.1701 |       0.2034 | Backwardation |
        | 2026-11-03 | 2026-12          |   96.09 | 2027-11        | 80.31 |       11 |    15.78 | -0.1642 |       0.1957 | Backwardation |
        | 2026-11-04 | 2026-12          |   88.12 | 2027-11        | 75.32 |       11 |    12.8  | -0.1453 |       0.1712 | Backwardation |
        | 2026-11-05 | 2026-12          |   90.52 | 2027-11        | 75.11 |       11 |    15.41 | -0.1702 |       0.2036 | Backwardation |
        | 2026-11-06 | 2026-12          |   90.83 | 2027-11        | 76.42 |       11 |    14.41 | -0.1586 |       0.1884 | Backwardation |
        """
        curves = self._get_commodity_curves(commodity, 12, generic, self._start_date)

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
