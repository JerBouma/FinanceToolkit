"""Risk Base Module"""

__docformat__ = "google"

from collections.abc import Callable
from functools import cached_property

import pandas as pd

from financetoolkit.helpers import LazyDict, handle_portfolio
from financetoolkit.risk.helpers import determine_within_historical_data
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    convert_annualized_rate_to_period,
)

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class RiskBase:
    """
    The data every risk topic shares: the tickers, returns and prices.
    """

    def __init__(
        self,
        tickers: list[str] | Callable[[], list[str]],
        historical_data: dict[str, pd.DataFrame] | None = None,
        risk_free_rate_data: dict[str, pd.Series] | None = None,
        intraday_period: str | None = None,
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Risk Controller Class.

        The Toolkit passes the tickers as a function without arguments and the period
        dictionaries as a LazyDict (financetoolkit.helpers), so a metric retrieves the data of
        the period it is calculated over when it is first needed, and nothing else.

        Args:
            tickers (list[str] | Callable[[], list[str]]): The tickers to use for the Toolkit instance.
            historical_data (dict[str, pd.DataFrame] | None, optional): The historical data per period.
                Defaults to None, which is treated as an empty dictionary.
            risk_free_rate_data (dict[str, pd.Series] | None, optional): The annualized risk free rate
                per period frequency, used for the Excess Volatility calculations. Defaults to None.
            intraday_period (str | None, optional): The intraday period used for within-period calculations.
                Defaults to None.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculations. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculations. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_value_at_risk(period='yearly')
        ```

        Which returns:

        |           |       0 |
        |:----------|--------:|
        | AAPL      | -0.2109 |
        | MSFT      | -0.2188 |
        | Benchmark | -0.1279 |
        """
        self._historical_data = historical_data if historical_data is not None else {}
        rates = risk_free_rate_data if risk_free_rate_data is not None else {}
        # The risk free rate is quoted as an annualized yield, so it is converted to the matching frequency. Without this, a daily return would have a full year of risk free rate subtracted from it.  # noqa: E501
        self._risk_free_rate_data = LazyDict(
            lambda frequency: convert_annualized_rate_to_period(
                rates[frequency], frequency
            )
        )
        self._tickers_source = tickers
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._portfolio_weights: dict | None = None

        # The returns within each period, built when a metric first asks for a period. Missing values stay missing: filling them with 0 would turn a period without data (e.g. before a listing, or the lone boundary observation at the start of the range) into 0% returns, i.e. a VaR, skewness or drawdown of exactly 0 instead of no value.  # noqa: E501
        self._within_historical_data = LazyDict(
            lambda period: determine_within_historical_data(
                period, self._historical_data, intraday_period
            )
        )

        # Memo for the return slices handed to the calculations, keyed by (period,
        # within_period): nearly every method starts from that slice and the MCP server
        # collects many metrics per session. Neither frame is mutated after this point.
        self._data_cache: dict[tuple[str, bool], pd.DataFrame] = {}

    @cached_property
    def _tickers(self) -> list[str]:
        """The tickers, retrieved when first needed."""
        return (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

    @handle_portfolio
    def _get_returns(self, period: str, within_period: bool) -> pd.DataFrame:
        """
        The Return slice a calculation runs on, memoized per (period, within_period)
        pair (see `_data_cache` in `__init__`).

        The whole-series slice drops the rows without any return: a resampled series
        starts with a NaN return (there is no previous period to compare against).
        Rows where only some tickers miss a return (e.g. a ticker that listed later)
        are kept, so every ticker uses its own full history; the calculations skip the
        missing returns per column. The within-period frames keep their missing values
        as well, so a period without observations for a ticker gives NaN rather than 0.
        """
        key = (period, within_period)

        if key not in self._data_cache:
            self._data_cache[key] = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"].dropna(how="all")
            )

        returns = self._data_cache[key]

        if not within_period:
            # A measure over the whole range compares returns with each other, so a single
            # return (e.g. one yearly return in a range within one year) gives a drawdown of
            # 0 and an undefined volatility, which reads as "no risk" without this note.
            short_history = returns.columns[returns.count() < 2]  # noqa: PLR2004

            if len(short_history):
                logger.warning(
                    "Fewer than two %s returns between %s and %s for %s, so measures over the whole range "  # noqa: E501
                    "are 0 or NaN. Use a higher frequency period, or within_period=True for a value per period.",  # noqa: E501
                    period,
                    self._start_date,
                    self._end_date,
                    ", ".join(map(str, short_history)),
                )

        return returns

    def _get_prices(
        self, column: str, period: str = "daily", within_period: bool = False
    ) -> pd.DataFrame:
        """
        The `column` prices on the dates each ticker traded.

        On a date only another ticker or the benchmark traded, a ticker's last price is
        carried forward as a flat bar without a Return (see
        `historical_model.fill_non_trading_days`). A measure on the prices themselves (a
        daily range, a price change) would count it as a day without a move, so the price
        is NaN there, as its Return is.
        """
        frame = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        )
        has_return = frame["Return"].notna()
        inside = has_return.cummax() & has_return[::-1].cummax()[::-1]

        return frame[column].mask(~has_return & inside)

    def _get_price_column(self, period: str, column: str) -> pd.DataFrame:
        # Reads the plain period-frequency history rather than the "within period" multi-index, so period="daily" simply means daily observations here and (unlike every within-period method above) needs no intraday data.  # noqa: E501
        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        # Only rows without any value are dropped: the pairwise calculations (copulas, tail dependence, CoVaR) align each pair themselves, so a joint dropna would cut every ticker's sample down to the shortest history.  # noqa: E501
        return self._historical_data[period][column].dropna(how="all")
