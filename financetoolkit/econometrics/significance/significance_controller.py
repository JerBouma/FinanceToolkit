"""Significance Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.econometrics.base.base_controller import EconometricsBase
from financetoolkit.econometrics.diagnostics import diagnostics_model
from financetoolkit.risk import risk_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR, apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class SignificanceEconometrics(EconometricsBase):
    """
    How unusual a level or a change is: the level and move significance, the change over the same window every year and streaks.
    """

    @staticmethod
    def _get_spreads(
        levels: pd.DataFrame, spread_over: str | None, ratio: bool = False
    ) -> pd.DataFrame:
        """
        Every series of `levels` minus the `spread_over` series (matched without regard to
        case), named "<series> - <spread_over>", or divided by it with `ratio`, named
        "<series> / <spread_over>", or `levels` itself without `spread_over` --
        shared by the methods that measure a level, so a spread of two series (producer over
        consumer price inflation) is measured like any series. A date either series misses is
        NaN, which the per-series methods leave out.
        """
        if spread_over is None:
            return levels

        names = {str(name).lower(): name for name in levels.columns}
        if str(spread_over).lower() not in names:
            raise ValueError(
                f"The spread_over series '{spread_over}' is not one of the series: "
                f"{', '.join(str(name) for name in levels.columns)}."
            )
        base = names[str(spread_over).lower()]
        others = levels.drop(columns=[base])
        if others.empty:
            raise ValueError(
                "A spread needs a second series next to the spread_over series."
            )

        if ratio:
            spreads = others.div(levels[base].replace(0, np.nan), axis=0)
            spreads.columns = [f"{name} / {base}" for name in others.columns]
        else:
            spreads = others.sub(levels[base], axis=0)
            spreads.columns = [f"{name} - {base}" for name in others.columns]
        return spreads

    @handle_errors
    def get_level_significance(
        self,
        period: str | None = None,
        column: str = "Close",
        spread_over: str | None = None,
        ratio: bool = False,
        include_benchmark: bool = False,
        level: float | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how unusual the latest value of every series is: its percentile in the series'
        own history and the last dates the series was at least as high ("the highest since") and at
        least as low ("the lowest since"), next to its change since the first observation and the
        highest and lowest values of the range with their dates and the change from them.

        Where `get_move_significance` (Risk) measures how unusual a move is, this measures how
        unusual a level is, which is the question for rates and macroeconomic series: a 10-year
        yield of 5.13% is "the highest since May 2002" when no month since then was as high, and
        an unemployment rate can be the lowest in decades without any single large move. Through
        the Toolkit it measures the prices of the tickers ("the highest close since"); with
        `Econometrics(data=...)` any series, such as the interest rates, inflation or
        unemployment of `Economics` and the yield curves of `FixedIncome`.

        The formula is as follows:

        - Percentile = (Count(x(s) < x(T)) + 0.5 * Count(x(s) = x(T))) / Count(s), over s < T
        - Rank = 1 + Count(x(s) > x(T)), over s < T: 1 for the highest of the history
        - Highest Since = the last date s < T with x(s) >= x(T)
        - Lowest Since = the last date s < T with x(s) <= x(T)
        - Change Since Start = (x(T) - x(0)) / |x(0)|
        - Change From Highest = (x(T) - max(x)) / |max(x)|
        - Change From Lowest = (x(T) - min(x)) / |min(x)|
        - Multiple Since Start = x(T) / x(0)
        - Annualised Change = (x(T) / x(0)) ^ (1 / years) - 1, years between the two dates
        - Largest Drawdown = min over t of (x(t) - max(x(s), s <= t)) / |max(x(s), s <= t)|
        - Recovery Date = the first date after the trough with x >= the peak before it

        Where x(T) is the latest value of a series, x(0) its first observation and s runs over its
        earlier observations. An
        earlier value equal to the latest one counts as half (the percentile rank), so a policy
        rate that has not changed for years sits in the middle of its own ties. Highest Since is
        NaN when no earlier value was as high, which makes the latest value the highest of the
        history; Lowest Since likewise. The Change Since Start is the change between the first and
        the latest observation in the range, so the start and end date choose the two dates to
        compare (housing starts from January 2021 to their latest month with period="monthly");
        it is relative to the absolute first value, so a rise from a negative value is positive,
        and NaN when the first value is zero. The Highest and Lowest are the peak and trough of
        the range, the latest value included, with the first dates they were reached, so the
        start date works as the anchor of an event window: the rise from the start to the peak,
        the date of the peak and the fall from it to the latest value. The changes from them
        are relative to their absolute values, and NaN when they are zero. With
        `Econometrics(data=..., changes="difference")` every change, the Largest Drawdown
        included, is a difference in the unit of the series instead (percentage points for a
        rate: a 10-year yield from 5.29% to 4.94% is -0.0035), and the Multiple Since Start and
        Annualised Change are NaN.

        Also known as: highest since, lowest since, record high, multi-year high, level
        percentile, historical percentile, how unusual is the latest value, change since,
        change between two dates, cumulative change, event window, peak, trough, drawdown
        from the peak, change from the high, change from the low, multiple, compound annual
        growth rate (CAGR), annualised change, peak to trough, time to recover the peak,
        change in percentage points.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly); the weekly to yearly periods take the last value of each period. Defaults to
                None, which is "daily": the series at the dates they are observed, so a monthly
                series is compared month by month.
            column (str, optional): The historical data column to measure. Defaults to "Close", the
                traded close of the tickers and the series themselves with data=.
            spread_over (str | None, optional): The name of a series to subtract from every
                other series, so the spreads over it are measured instead (named "<series> -
                <spread_over>"), e.g. producer price inflation over consumer price inflation.
                The spread exists on the dates both series have, so with period="daily" a weekly
                series minus a daily one is measured on the weekly dates. Defaults to None, which
                measures the series themselves.
            ratio (bool, optional): Whether to divide every other series by the spread_over
                series instead of subtracting it (named "<series> / <spread_over>"), e.g.
                exports over GDP or one country's yield over another's. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            level (float | None, optional): A level to compare every series with, such as an
                earlier peak, which adds the Level, the First Above Level (the first date the
                series exceeded it) and Above Level Since (the date it has been above it since
                without a break, empty when the latest value is not above it). Defaults to
                None, which leaves them out.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per series the Start Date, Start Value and number of Observations of its
            history, the Latest Date and Latest Value, the Change Since Start (decimal), the
            Multiple Since Start and Annualised Change (decimal, the compound annual rate, NaN
            unless both values are positive), the Percentile (decimal), the Rank (1 for the
            highest of the history: "the 3rd highest since" the Start Date), the Highest Since and
            Lowest Since dates, the Highest and Lowest with their Highest Date and Lowest Date
            and the Change From Highest and Change From Lowest (decimals), and the Largest
            Drawdown (decimal) with its Drawdown Peak Date, Drawdown Trough Date and Recovery
            Date (NaN while it has not recovered). The dates are text: the dates of the observations, e.g.
            2002-05-31 for a monthly series, or the periods, e.g. 2002-05 with period="monthly".

        Notes:
        - The answer depends on the history it is measured against: the history starts at the start
          date of the data, so "the highest since" can at most reach back to that date, and a
          later start date gives a higher percentile for a value near the top. Choose a start date
          far enough back, and read the Start Date column to see where each history begins.
        - Each series uses its own history, so a series that starts later has a shorter one.
        - The latest value is the last observation up to the end date, which can be older than the
          end date for a series published with a lag; Latest Date says which it is.

        As an example:

        ```python
        from financetoolkit import Economics, Econometrics

        economics = Economics(start_date='2000-01-01', end_date='2026-10-31')

        long_term_rates = economics.get_long_term_interest_rate(
            countries=['United States', 'Germany', 'Japan'], period='monthly'
        )

        econometrics = Econometrics(data=long_term_rates)

        econometrics.get_level_significance().T
        ```

        Which returns:

        |                      | United States   | Germany    | Japan      |
        |:---------------------|:----------------|:-----------|:-----------|
        | Start Date           | 2000-01-31      | 2000-01-31 | 2000-01-31 |
        | Start Value          | 0.0666          | 0.0554     | 0.0169     |
        | Observations         | 322             | 322        | 322        |
        | Latest Date          | 2026-10-31      | 2026-10-31 | 2026-10-31 |
        | Latest Value         | 0.0513          | 0.0342     | 0.0276     |
        | Change Since Start   | -0.2297         | -0.3827    | 0.6331     |
        | Multiple Since Start | 0.7703          | 0.6173     | 1.6331     |
        | Annualised Change    | -0.0097         | -0.0179    | 0.0185     |
        | Percentile           | 0.9377          | 0.6854     | 0.9907     |
        | Rank                 | 21              | 102        | 4          |
        | Highest Since        | 2002-05-31      | 2026-09-30 | 2026-09-30 |
        | Lowest Since         | 2026-09-30      | 2026-08-31 | 2026-06-30 |
        | Highest              | 0.0666          | 0.0554     | 0.0305     |
        | Highest Date         | 2000-01-31      | 2000-01-31 | 2026-09-30 |
        | Change From Highest  | -0.2297         | -0.3827    | -0.0951    |
        | Lowest               | 0.0062          | -0.0065    | -0.0028    |
        | Lowest Date          | 2020-07-31      | 2019-08-31 | 2019-08-31 |
        | Change From Lowest   | 7.2742          | 6.2615     | 10.8571    |
        | Largest Drawdown     | -0.9069         | -1.1173    | -1.1429    |
        | Drawdown Peak Date   | 2000-01-31      | 2000-01-31 | 2006-04-30 |
        | Drawdown Trough Date | 2020-07-31      | 2019-08-31 | 2019-08-31 |
        | Recovery Date        | nan             | nan        | 2025-12-31 |
        """
        period = period if period else "daily"
        levels = self._get_spreads(
            self._filter_benchmark(
                self._get_series_column(period, column),
                include_benchmark=include_benchmark,
            ),
            spread_over,
            ratio,
        )

        level_significance = pd.DataFrame(
            {
                series: diagnostics_model.get_level_significance(
                    levels[series],
                    level=level,
                    difference=self._changes == "difference",
                )
                for series in levels.columns
            }
        ).T.infer_objects()

        return apply_rounding(
            level_significance, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_move_significance(
        self,
        period: str | None = None,
        window: int | None = None,
        alpha: float = 0.05,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how unusual each move of every series is: the change in units of its trailing
        volatility and of its trailing Value at Risk, its percentile among the earlier moves and
        the date of the last move at least as large in the same direction ("the largest one-day
        rise since").

        This is `get_move_significance` (Risk) for any series rather than the returns of the
        Toolkit's tickers: with `Econometrics(data=..., changes="difference")` a yield, rate or
        spread moves in points, so a rise of the 2-year Treasury yield from 4.56% to 4.73% is a
        move of 0.0017, measured against the earlier daily moves of the same yield. Where
        `get_level_significance` asks how unusual a level is, this asks how unusual a change is.

        The formula is as follows:

        - Volatility Multiple(t) = Change(t) / σ(Change(t - window), ..., Change(t - 1))
        - VaR Multiple(t) = Change(t) / |Quantile(Change(t - window), ..., Change(t - 1); alpha)|
        - Percentile(t) = Count(|Change(s)| < |Change(t)|, s < t) / Count(s < t)
        - Largest Since(t) = max(s < t : Change(s) <= Change(t)) if Change(t) < 0
        - Largest Since(t) = max(s < t : Change(s) >= Change(t)) if Change(t) > 0

        The Change is the series' change as `changes` defines it: a percentage change by default,
        the difference with `changes="difference"` and the series itself with `changes="none"`.
        The volatility σ uses ddof=1 and the quantile is the historic Value at Risk, interpolated
        linearly. Each measure only uses the changes before the period itself (no look-ahead),
        and Largest Since compares falls with falls and rises with rises, as a headline does.

        Also known as: sigma move, largest move since, biggest daily rise since, standardized
        change, yield move significance.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly
                or yearly). Defaults to None, which is daily.
            window (int | None, optional): The number of earlier changes the volatility is
                measured over, which the percentile also needs at least. Defaults to None, which
                uses a year of observations with a minimum of 20 (252 daily, 52 weekly and 20
                monthly, quarterly or yearly changes).
            alpha (float, optional): The tail probability of the trailing historic Value at Risk
                the VaR Multiple divides by (0.05 for the 95% VaR). Defaults to 0.05.
            include_benchmark (bool, optional): Whether to include the benchmark of a Toolkit.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Return (the change), Volatility Multiple, VaR Multiple, Percentile and
            Largest Since (the date as text) per series, with time as the index and the measures
            and series as the column levels.

        Notes:
        - The history a move is compared with starts at the start of the data: the volatility,
          the VaR and the percentile are NaN until `window` changes are available, and Largest
          Since is NaN when no earlier move was as large, which makes the move the largest since
          the start. Choose an early start date for a long history.
        - Each series uses its own history and only the dates it has a value for, so a market
          holiday is not a move of zero.

        As an example:

        ```python
        from financetoolkit import Econometrics, FixedIncome

        fixedincome = FixedIncome(start_date='2025-01-01', end_date='2026-11-10')

        treasury_rates = fixedincome.get_treasury_rates(maturities=['2Y', '10Y'])

        econometrics = Econometrics(data=treasury_rates, changes='difference')

        econometrics.get_move_significance().loc['2026-11-10'].unstack(level=0)
        ```

        Which returns:

        |         |   Return |   Volatility Multiple |   VaR Multiple |   Percentile | Largest Since   |
        |:--------|---------:|----------------------:|---------------:|-------------:|:----------------|
        | 2 Year  |   0.0017 |                3.5345 |         2.4286 |       0.9935 | 2025-04-09      |
        | 10 Year |   0.001  |                2.2808 |         1.6667 |       0.9546 | 2026-09-23      |
        |         |   Return |   Volatility Multiple |   VaR Multiple |   Percentile | Largest Since   |
        |:--------|---------:|----------------------:|---------------:|-------------:|:----------------|
        | 2 Year  |   0.0017 |                3.5345 |         2.4286 |       0.9935 | 2025-04-09      |
        | 10 Year |   0.001  |                2.2808 |         1.6667 |       0.9546 | 2026-09-23      |
        """
        period = period if period else "daily"

        if window is not None and window < 2:  # noqa: PLR2004
            raise ValueError(
                "The window needs at least 2 changes to measure a volatility."
            )
        if not 0 < alpha < 1:
            raise ValueError(
                "alpha must be between 0 and 1, e.g. 0.05 for the 95% VaR."
            )

        window = window if window else max(PERIODS_PER_YEAR[period], 20)
        changes = self._filter_benchmark(
            self._get_series_column(period, "Return"),
            include_benchmark=include_benchmark,
        )

        move_significance = pd.concat(
            {
                series: risk_model.get_move_significance(
                    changes[series].dropna(), window, alpha
                )
                for series in changes.columns
            },
            axis=1,
        ).swaplevel(1, 0, axis=1)

        # The measures in reading order, each with the series in their own order.
        move_significance = move_significance[
            pd.MultiIndex.from_product(
                [
                    [
                        "Return",
                        "Volatility Multiple",
                        "VaR Multiple",
                        "Percentile",
                        "Largest Since",
                    ],
                    changes.columns,
                ]
            )
        ].dropna(how="all", axis=0)

        return apply_rounding(
            move_significance, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_window_change_by_year(
        self,
        from_month: int = 1,
        to_month: int = 12,
        column: str = "Close",
        spread_over: str | None = None,
        ratio: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the change of every series over the same months of every year, such as
        February to October, with the rank of every year's change and the last earlier year
        with a change at least as large ("the largest since").

        Where `get_level_significance` measures one window, from the start date to the
        latest value, this compares the same window across years: "US consumer prices rose
        3.3% from February to October, the most since 2022" is the Change, Rank and Largest
        Since of the latest year. Through the Toolkit it measures the prices of the tickers
        (a seasonal pattern, such as the return from May to October); with
        `Econometrics(data=...)` any series, such as the consumer price indices of
        `Economics`.

        The formula is as follows:

        - Change (y) = (x(to_month, y) - x(from_month, y)) / |x(from_month, y)|
        - Rank (y) = the position of Change (y) among the changes of every year, 1 for the largest
        - Largest Since (y) = the last year s < y with Change (s) >= Change (y)

        Where x(m, y) is the value of month m in year y, the last value of the month for a
        daily series. A window that ends in an earlier month than it starts (November to
        February) runs into the next year and is labelled by the year it ends; the same
        month for both gives the change over twelve months. A year without the value of
        either month has no change, so the current year is empty until its last month is
        published. Largest Since is NaN when no earlier year had a change as large. With
        `Econometrics(data=..., changes="difference")` the Change is x(to_month, y) -
        x(from_month, y), in percentage points for a rate.

        Also known as: same period each year, seasonal change, year-to-date comparison,
        change from February to October, largest since, fastest since, seasonality.

        Args:
            from_month (int, optional): The month the window starts, 1 (January) to 12
                (December). Defaults to 1.
            to_month (int, optional): The month the window ends, 1 to 12. Defaults to 12,
                which with from_month=1 is the change from January to December.
            column (str, optional): The historical data column to measure. Defaults to
                "Close", the traded close of the tickers and the series themselves with data=.
            spread_over (str | None, optional): The name of a series to subtract from every
                other series, so the spreads over it are measured instead. Defaults to None.
            ratio (bool, optional): Whether to divide every other series by the spread_over
                series instead of subtracting it (named "<series> / <spread_over>"). Defaults
                to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class (4 unless set
                otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per year (rows) and series the Start Value, End Value, Change
            (decimal), Rank and Largest Since (columns, a level per series).

        Notes:
        - The ranks and "largest since" depend on the history: start the data early enough,
          as a year before the start date is not compared.
        - The change uses the levels, so the series should be an index or a price, not a
          rate: the change in an inflation rate is a relative change of a rate.

        As an example:

        ```python
        from financetoolkit import Economics, Econometrics

        economics = Economics(start_date='2019-01-01', end_date='2026-10-31')

        consumer_price_index = economics.get_consumer_price_index(
            countries=['United States', 'Euro Area']
        )

        econometrics = Econometrics(data=consumer_price_index)

        econometrics.get_window_change_by_year(from_month=2, to_month=10)['United States']
        ```

        Which returns:

        |   Year |   Start Value |   End Value |   Change |   Rank |   Largest Since |
        |-------:|--------------:|------------:|---------:|-------:|----------------:|
        |   2019 |       115.923 |     118.019 |   0.0181 |      5 |             nan |
        |   2020 |       118.629 |     119.414 |   0.0066 |      7 |            2019 |
        |   2021 |       120.618 |     126.843 |   0.0516 |      1 |             nan |
        |   2022 |       130.112 |     136.668 |   0.0504 |      2 |            2021 |
        |   2023 |       137.965 |     141.098 |   0.0227 |      4 |            2022 |
        |   2024 |       142.315 |     144.763 |   0.0172 |      6 |            2023 |
        |   2025 |       146.331 |     nan     | nan      |    nan |             nan |
        |   2026 |       149.863 |     154.782 |   0.0328 |      3 |            2022 |
        """
        levels = self._get_spreads(
            self._filter_benchmark(
                self._get_series_column("monthly", column),
                include_benchmark=include_benchmark,
            ),
            spread_over,
            ratio,
        )

        window_change = pd.concat(
            {
                series: diagnostics_model.get_window_change_by_year(
                    levels[series],
                    from_month,
                    to_month,
                    difference=self._changes == "difference",
                )
                for series in levels.columns
            },
            axis=1,
        )

        return apply_rounding(
            window_change, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_streaks(
        self,
        period: str | None = None,
        column: str = "Close",
        spread_over: str | None = None,
        ratio: bool = False,
        level: float | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the streak every series is on: whether its latest value is positive,
        negative or zero, since when it has been without a break and how long that is, the
        run before it, and the longest earlier run of the same sign.

        Where `get_level_significance` measures how unusual the latest level is, this measures
        how long a series has been on one side of zero, which is the question for inflation,
        growth and spreads: producer prices that fell year over year for 41 months in a row, a
        spread between two series that turned positive after three years, or a yield curve
        that has been inverted since a given month. With column="Return" it measures the
        changes instead: a price that rose for eight days in a row, or with
        `Econometrics(data=..., changes="difference")` an unemployment rate that rose for six
        months in a row. Through the Toolkit it measures the prices of the tickers; with
        `Econometrics(data=...)` any series, such as the economic calendar of `Economics`
        with as_series=True, its inflation rates or the yield curves of `FixedIncome`.

        A run is a stretch of consecutive observations with the same sign (positive, negative
        or zero), counted in observations: months for a monthly series. The current run is the
        one the latest observation belongs to, the previous run the one just before it and the
        longest earlier run the longest run of the current sign before the current one (the
        most recent of them when several are as long).

        Also known as: streak, run length, consecutive months, months in a row, how long has
        it been negative, first positive since.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly); the weekly to yearly periods take the last value of each period. Defaults to
                None, which is "daily": the series at the dates they are observed, so a monthly
                series is counted month by month.
            column (str, optional): The historical data column to measure. Defaults to "Close", the
                traded close of the tickers and the series themselves with data=; "Return" measures
                their changes.
            spread_over (str | None, optional): The name of a series to subtract from every
                other series, so the streaks of the spreads over it are measured instead (named
                "<series> - <spread_over>"), e.g. producer price inflation over consumer price
                inflation. The spread exists on the dates both series have, so with
                period="daily" a weekly series minus a daily one is measured on the weekly dates.
                Defaults to None, which measures the series themselves.
            ratio (bool, optional): Whether to divide every other series by the spread_over
                series instead of subtracting it (named "<series> / <spread_over>"). Defaults
                to False.
            level (float | None, optional): The level the runs are measured against, so the
                runs are above or below it ("27 years below 2%") with the signs "Above Level",
                "Below Level" and "At Level". Defaults to None, which is zero: the sign of the
                values.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per series the Start Date and number of Observations of its history,
            the Latest Date and Latest Value, the Sign of the current run with its Run Start and
            Run Length, the Previous Sign, Previous Run Start, Previous Run End and Previous Run
            Length, and the Longest Earlier Run Start, End and Length. The dates are text (the
            dates of the observations, or the periods, e.g. 2022-10 with period="monthly"), the
            lengths are numbers of observations and a run that does not exist is NaN.

        Notes:
        - The runs are counted within the history, which starts at the start date of the data:
          a run that began before it is counted from the first observation, which the Run Start
          or Previous Run Start then equals.
        - A value of exactly zero is a run of its own, so it ends a positive or negative run.
        - Each series uses its own history, and the latest value is the last observation up to
          the end date, which Latest Date shows.

        As an example:

        ```python
        from financetoolkit import Economics, Econometrics

        economics = Economics(api_key="FINANCIAL_MODELING_PREP_KEY")

        prices = economics.get_economic_calendar(
            start_date="2021-06-01",
            end_date="2026-11-09",
            countries="China",
            events="Producer Price Index YoY,Inflation Rate YoY",
            impact="All",
            as_series=True,
        )

        econometrics = Econometrics(
            data=prices["China"][["Producer Price Index YoY", "Inflation Rate YoY"]]
        )

        econometrics.get_streaks(period="monthly").T
        ```

        Which returns:

        |                            | Producer Price Index YoY   | Inflation Rate YoY   |
        |:---------------------------|:---------------------------|:---------------------|
        | Start Date                 | 2021-05                    | 2021-05              |
        | Observations               | 66                         | 66                   |
        | Latest Date                | 2026-10                    | 2026-10              |
        | Latest Value               | 0.057                      | 0.007                |
        | Sign                       | Positive                   | Positive             |
        | Run Start                  | 2026-03                    | 2025-10              |
        | Run Length                 | 8                          | 13                   |
        | Previous Sign              | Negative                   | Negative             |
        | Previous Run Start         | 2022-10                    | 2025-08              |
        | Previous Run End           | 2026-02                    | 2025-09              |
        | Previous Run Length        | 41                         | 2                    |
        | Longest Earlier Run Start  | 2021-05                    | 2021-05              |
        | Longest Earlier Run End    | 2022-09                    | 2023-05              |
        | Longest Earlier Run Length | 17                         | 25                   |
        """
        period = period if period else "daily"
        levels = self._get_spreads(
            self._filter_benchmark(
                self._get_series_column(period, column),
                include_benchmark=include_benchmark,
            ),
            spread_over,
            ratio,
        )

        streaks = pd.DataFrame(
            {
                series: diagnostics_model.get_streaks(levels[series], level=level)
                for series in levels.columns
            }
        ).T.infer_objects()

        return apply_rounding(
            streaks, rounding if rounding is not None else self._rounding
        )
