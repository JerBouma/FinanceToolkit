"""Statistics Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
)

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class StatisticsTechnicals:
    """
    Statistical indicators of the price: rolling measures such as the standard deviation, variance, linear regression and correlation over a window.
    """

    @handle_errors
    def collect_statistics_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects the statistics indicators based on the provided data.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            window (int, optional): The window size for calculating indicators.
                Defaults to 14.
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame: Statistics indicators calculated based on the specified parameters.

        Notes:
        - The method calculates the statistics indicators for each asset in the Toolkit instance.
        - While the topic holds no indicators, the result is an empty DataFrame, which
          `collect_all_indicators` leaves out.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.collect_statistics_indicators()
        ```
        """
        if period not in [
            "intraday",
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "yearly",
        ]:
            raise ValueError(
                "Period must be intraday, daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "intraday" and self._historical_data[period].empty:
            raise ValueError(
                "Please define the 'intraday_period' parameter when initializing the Toolkit."
            )

        statistics_indicators: dict = {}

        if not statistics_indicators:
            return pd.DataFrame()

        statistics_collection = pd.concat(statistics_indicators, axis=1)

        statistics_collection = apply_rounding(
            statistics_collection,
            rounding if rounding is not None else self._rounding,
        ).loc[self._start_date : self._end_date]

        if growth:
            statistics_collection = calculate_growth(
                dataset=statistics_collection,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="index",
            )

        if standardize:
            statistics_collection = calculate_standardization(
                dataset=statistics_collection,
                rounding=rounding if rounding is not None else self._rounding,
                axis="rows",
            )

        if len(self._tickers) == 1:
            return statistics_collection.xs(self._tickers[0], level=1, axis="columns")

        return statistics_collection
