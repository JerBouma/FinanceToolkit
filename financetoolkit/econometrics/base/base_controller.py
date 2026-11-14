"""Econometrics Base Module"""

__docformat__ = "google"

import itertools
from collections.abc import Callable
from functools import cached_property

import pandas as pd

from financetoolkit.econometrics import data_model
from financetoolkit.helpers import LazyDict
from financetoolkit.risk.helpers import determine_within_historical_data
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501

logger = get_logger()

# The synthetic benchmark series, excluded unless `include_benchmark=True`.
BENCHMARK_TICKER = "Benchmark"


@filter_shown_columns
@track_sources
class EconometricsBase:
    """
    The data every econometrics topic shares: the series, returns, spreads, ticker pairs and the fitted OLS results.
    """

    def __init__(
        self,
        tickers: str | list[str] | Callable[[], list[str]] | None = None,
        historical_data: dict[str, pd.DataFrame] | None = None,
        intraday_period: str | None = None,
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
        data: pd.DataFrame | pd.Series | dict | None = None,
        changes: str | None = "percentage",
    ):
        """
        Initializes the Econometrics Controller Class, either with the historical data of a
        Toolkit (which `toolkit.econometrics` passes) or standalone with any data.

        Standalone, the data can be a DataFrame with a column per series, a Series, or a
        dictionary of them; the dates can be in the rows or, as in the ratios of the Toolkit
        (tickers by year), in the columns. Every series becomes an asset: the series as
        given is the "Adj Close" column (also "Level") that the unit root, cointegration and
        forecasting methods use by default, and its change from one observation to the next
        is the "Return" column (also "Change") that the regressions and diagnostics use. The
        "daily" period, the default, holds the series at the dates they are observed (so a
        monthly series is analyzed monthly); the other periods take the last observation of
        each week, month, quarter or year.

        The Toolkit passes the tickers as a function without arguments and the historical data
        as a LazyDict (financetoolkit.helpers), so a method retrieves the data of the period it
        is calculated over when it is first needed, and nothing else.

        Args:
            tickers (str | list[str] | Callable[[], list[str]] | None, optional): The tickers of the
                Toolkit, or a function without arguments that returns them. Defaults to None, which
                takes the names of the series when data is given.
            historical_data (dict[str, pd.DataFrame] | None, optional): The historical data per period,
                as the Toolkit passes it. Defaults to None, which is treated as an empty dictionary.
            intraday_period (str | None, optional): The intraday period used for within-period calculations.
                Defaults to None.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculations. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculations. Defaults to None.
            data (pd.DataFrame | pd.Series | dict | None, optional): Any data to analyze instead
                of the Toolkit's prices, with a series per column. Defaults to None.
            changes (str | None, optional): How the changes ("Return") of the data are calculated:
                "percentage" (relative changes, as returns), "log", "difference" (first
                differences, for rates and ratios that can be zero or negative) or None
                (the data are changes already). The percentage change divides by the
                absolute previous value, so a negative series that rises has a positive
                change. Defaults to "percentage".

        As an example:

        ```python
        from financetoolkit import Economics, Econometrics

        economics = Economics(start_date='2000-01-01', end_date='2025-12-31')

        unemployment = economics.get_unemployment_rate(
            countries=['United States', 'Germany'], period='monthly'
        )

        econometrics = Econometrics(data=unemployment, changes='difference')

        econometrics.get_augmented_dickey_fuller()
        ```

        Which returns:

        |                       |   United States |    Germany |
        |:----------------------|----------------:|-----------:|
        | ADF Statistic         |      -2.98304   |  -0.930407 |
        | P-Value               |       0.0365087 |   0.777794 |
        | Lags Used             |       0         |   3        |
        | Observations          |     310         | 307        |
        | Critical Value 1%     |      -3.45162   |  -3.45183  |
        | Critical Value 5%     |      -2.87091   |  -2.871    |
        | Critical Value 10%    |      -2.57176   |  -2.57181  |
        | Reject Unit Root (5%) |       1         |   0        |
        """
        if data is not None:
            if historical_data is not None:
                raise ValueError(
                    "Pass either data (any metric) or historical_data (from a Toolkit), not both."
                )
            historical_data, series = data_model.build_historical_data(
                data, changes=changes, start_date=start_date, end_date=end_date
            )
            tickers = series if tickers is None else tickers
        if tickers is None:
            raise ValueError(
                "Pass the data to analyze, e.g. Econometrics(data=dataframe), or use the "
                "Econometrics module of a Toolkit: toolkit.econometrics."
            )

        # Mirrors the Risk controller: the Toolkit always passes the period dict, and an
        # explicit empty dict keeps the lookups raising a plain KeyError.
        self._historical_data = historical_data if historical_data is not None else {}
        self._tickers_source = tickers
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        # With "difference", the changes of the level significance are in points too.
        self._changes = changes
        self._portfolio_weights: dict | None = None
        # Memo for the sliced-and-dropna'd frames handed to the calculations: nearly
        # every method starts from the same (period, column) slice, and re-copying it
        # per call adds up when many metrics are collected in a row (as the MCP server
        # does). The historical data is never mutated once retrieved, and the
        # slices are treated as read-only by every caller, so sharing them is safe.
        self._data_cache: dict[tuple, pd.DataFrame] = {}

        # The returns within each period, built when a method first asks for a period.
        # Missing returns stay NaN (not 0), so a ticker without data in part of a period is
        # not treated as flat there.
        self._within_historical_data = LazyDict(
            lambda period: determine_within_historical_data(
                period, self._historical_data, intraday_period
            )
        )

    @cached_property
    def _tickers(self) -> list[str]:
        """The tickers as a list, retrieved when first needed."""
        tickers = (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

        return [tickers] if isinstance(tickers, str) else list(tickers)

    def _get_series_column(
        self, period: str, column: str, tickers: list[str] | None = None
    ) -> pd.DataFrame:
        """
        The `column` slice of `period`, memoized -- see `_data_cache` in `__init__`. It
        serves the Toolkit's prices and returns and any data given with `data=` alike.

        Without `tickers` only the dates no series has are dropped: per-series and
        pairwise methods drop their own missing values, so each series keeps its own
        history and a series observed at other dates -- a monthly series next to a daily
        one, an asset listed later, another holiday calendar -- does not cut the others
        down to the shared dates. Methods that combine series date by date (regressions,
        VAR, cointegration systems) pass the `tickers` they use and keep the dates all of
        those series have, so a series outside the model never removes dates.
        """
        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        # Deduplicated so a ticker named twice (e.g. as dependent and independent)
        # still selects a single column.
        tickers = list(dict.fromkeys(tickers)) if tickers is not None else None
        key = ("column", period, column, tuple(tickers) if tickers else None)

        if key not in self._data_cache:
            # Historical data supplied to the Toolkit can reach beyond its date range.
            data = self._historical_data[period][column].loc[
                self._start_date : self._end_date
            ]
            self._data_cache[key] = (
                data.dropna(how="all") if tickers is None else data[tickers].dropna()
            )

        return self._data_cache[key]

    def _get_tickers(self, include_benchmark: bool = False) -> list[str]:
        """
        `self._tickers`, optionally with `"Benchmark"` filtered out -- shared by
        every method below that derives its own default ticker set (rather than
        having every relevant ticker passed in explicitly), so `include_benchmark`
        behaves identically everywhere it appears.
        """
        tickers = list(self._tickers)

        if not include_benchmark:
            tickers = [ticker for ticker in tickers if ticker != BENCHMARK_TICKER]

        return tickers

    def _filter_benchmark(
        self, data: pd.DataFrame, include_benchmark: bool
    ) -> pd.DataFrame:
        """
        Drops the `"Benchmark"` column from `data` unless `include_benchmark` is
        True -- shared by the per-asset diagnostic methods below, which loop over
        every column of the historical data directly rather than over
        `self._tickers`.
        """
        if include_benchmark or BENCHMARK_TICKER not in data.columns:
            return data

        return data.drop(columns=[BENCHMARK_TICKER])

    def _resolve_dependent_independent(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
    ) -> tuple[str, list[str]]:
        """
        Resolves a dependent ticker and one-or-more independent ticker(s) from
        `self._tickers` for every simple "one dependent, one-or-more
        independent(s)" regression-style method on this controller. Defaults to
        `self._tickers[0]` as the dependent ticker and every other ticker in
        `self._tickers` as independent, unless overridden.
        """
        tickers = self._get_tickers(include_benchmark=include_benchmark)

        dependent_ticker = (
            dependent_ticker if dependent_ticker is not None else tickers[0]
        )

        if independent_tickers is None:
            independent_tickers = [
                ticker for ticker in tickers if ticker != dependent_ticker
            ]
        else:
            independent_tickers = (
                [independent_tickers]
                if isinstance(independent_tickers, str)
                else list(independent_tickers)
            )

        if not independent_tickers:
            raise ValueError(
                "No independent ticker(s) resolved -- the Toolkit instance must "
                "contain at least 2 tickers, or independent_tickers must be given "
                "explicitly."
            )

        logger.info(
            "Dependent ticker: %s | Independent ticker(s): %s",
            dependent_ticker,
            independent_tickers,
        )

        return dependent_ticker, independent_tickers

    def _all_unordered_pairs(
        self, include_benchmark: bool = False
    ) -> list[tuple[str, str]]:
        """
        Every unordered pair `(ticker_a, ticker_b)` of `self._tickers` -- used by
        the symmetric pairwise methods below (two-sample t-test, RMSE, MAE), where
        swapping the pair does not change the test.
        """
        return list(
            itertools.combinations(
                self._get_tickers(include_benchmark=include_benchmark), 2
            )
        )
