"""Breadth Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import get_non_trading_days, handle_portfolio
from financetoolkit.technicals.breadth import breadth_model
from financetoolkit.technicals.helpers import calculate_per_ticker
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    PERIODS_PER_YEAR,
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


@filter_shown_columns
@track_sources
class BreadthTechnicals:
    """
    Breadth and volume indicators: the McClellan oscillator, advancers and decliners, on-balance volume, the price volume trend, the accumulation/distribution line, the Chaikin oscillator and money flow, the ease of movement, the negative and positive volume index, TRIN, new highs and lows and the market breadth.
    """

    @handle_errors
    def collect_breadth_indicators(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects various breadth indicators based on the provided data.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
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
            pd.Series or pd.DataFrame: Breadth indicators calculated based on the specified parameters.

        Notes:
        - The method calculates various breadth indicators for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.collect_breadth_indicators().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   McClellan Oscillator |   Advancers - Decliners |   On-Balance Volume |   Accumulation/Distribution Line |   Chaikin Oscillator |     TRIN |   New Highs - New Lows |   Chaikin Money Flow |   Ease of Movement |   Negative Volume Index |   Positive Volume Index |
        |:-----------|-----------------------:|------------------------:|--------------------:|---------------------------------:|---------------------:|---------:|-----------------------:|---------------------:|-------------------:|------------------------:|------------------------:|
        | 2025-12-17 |                -0.0981 |                      -1 |         2.26688e+09 |                      3.99772e+09 |         -1.74845e+07 | nan      |                      0 |              -0.0358 |            -4.7239 |                 2735.56 |                 788.833 |
        | 2025-12-18 |                -0.0378 |                       1 |         2.31851e+09 |                      4.02709e+09 |         -1.16785e+07 | nan      |                      0 |               0.0024 |            -7.4207 |                 2735.56 |                 789.848 |
        | 2025-12-19 |                 0.0139 |                       1 |         2.46314e+09 |                      4.11449e+09 |          1.95658e+07 | nan      |                      0 |               0.1361 |            -9.5049 |                 2735.56 |                 794.143 |
        | 2025-12-22 |                -0.0419 |                      -1 |         2.42657e+09 |                      4.0879e+09  |          2.21091e+07 | nan      |                      0 |               0.079  |           -12.8462 |                 2708.57 |                 794.143 |
        | 2025-12-23 |                 0.0106 |                       1 |         2.45621e+09 |                      4.11472e+09 |          2.96729e+07 | nan      |                      0 |               0.0651 |           -14.4874 |                 2722.46 |                 794.143 |
        | 2025-12-24 |                 0.0554 |                       1 |         2.47412e+09 |                      4.11466e+09 |          3.0052e+07  | nan      |                      0 |               0.0846 |            -6.6556 |                 2736.96 |                 794.143 |
        | 2025-12-26 |                -0.0066 |                      -1 |         2.4526e+09  |                      4.1024e+09  |          2.35738e+07 | nan      |                      0 |               0.0857 |            -5.448  |                 2736.96 |                 792.954 |
        | 2025-12-29 |                 0.0405 |                       1 |         2.47632e+09 |                      4.11196e+09 |          2.18213e+07 |   0.4593 |                      0 |               0.0756 |            -4.8029 |                 2736.96 |                 793.999 |
        | 2025-12-30 |                -0.0195 |                      -1 |         2.45418e+09 |                      4.1095e+09  |          1.83379e+07 |   1.5877 |                      0 |               0.0249 |            -5.2917 |                 2730.16 |                 793.999 |
        | 2025-12-31 |                -0.0707 |                      -1 |         2.42688e+09 |                      4.08532e+09 |          7.55142e+06 | nan      |                      0 |              -0.0357 |            -5.2549 |                 2730.16 |                 790.451 |
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

        breadth_indicators: dict = {}

        breadth_indicators["McClellan Oscillator"] = self.get_mcclellan_oscillator(
            period=period, close_column=close_column
        )

        breadth_indicators["Advancers - Decliners"] = self.get_advancers_decliners(
            period=period, close_column=close_column
        )
        breadth_indicators["On-Balance Volume"] = self.get_on_balance_volume(
            period=period, close_column=close_column
        )
        breadth_indicators["Price Volume Trend"] = self.get_price_volume_trend(
            period=period, close_column=close_column
        )

        breadth_indicators["Accumulation/Distribution Line"] = (
            self.get_accumulation_distribution_line(
                period=period, close_column=close_column
            )
        )

        breadth_indicators["Chaikin Oscillator"] = self.get_chaikin_oscillator(
            period=period, close_column=close_column
        )

        breadth_indicators["TRIN"] = self.get_trin(
            period=period, close_column=close_column
        )

        breadth_indicators["New Highs - New Lows"] = self.get_new_highs_new_lows(
            period=period, close_column=close_column
        )

        breadth_indicators["Chaikin Money Flow"] = self.get_chaikin_money_flow(
            period=period, close_column=close_column
        )

        breadth_indicators["Ease of Movement"] = self.get_ease_of_movement(
            period=period, close_column=close_column
        )

        breadth_indicators["Negative Volume Index"] = self.get_negative_volume_index(
            period=period, close_column=close_column
        )

        breadth_indicators["Positive Volume Index"] = self.get_positive_volume_index(
            period=period, close_column=close_column
        )

        self._breadth_indicators = pd.concat(breadth_indicators, axis=1)

        self._breadth_indicators = apply_rounding(
            self._breadth_indicators,
            rounding if rounding is not None else self._rounding,
        ).loc[self._start_date : self._end_date]

        if growth:
            self._breadth_indicators_growth = calculate_growth(
                dataset=self._breadth_indicators,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="index",
            )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                self._breadth_indicators_growth = calculate_standardization(
                    dataset=self._breadth_indicators_growth,
                    rounding=standardize_rounding,
                    axis="rows",
                )
            else:
                self._breadth_indicators = calculate_standardization(
                    dataset=self._breadth_indicators,
                    rounding=standardize_rounding,
                    axis="rows",
                )

        if len(self._tickers) == 1:
            return (
                self._breadth_indicators_growth.xs(
                    self._tickers[0], level=1, axis="columns"
                )
                if growth
                else self._breadth_indicators.xs(
                    self._tickers[0], level=1, axis="columns"
                )
            )

        return self._breadth_indicators_growth if growth else self._breadth_indicators

    @handle_portfolio
    @handle_errors
    def get_mcclellan_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_ema_window: int = 19,
        long_ema_window: int = 39,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the McClellan Oscillator for a given price series.

        The McClellan Oscillator is the difference between a short-term and a long-term
        exponential moving average of the daily advance/decline signal. Traditionally it is
        calculated from the number of advancing minus declining stocks across a market; here
        it is applied to each ticker's own signal (+1 on a day it closed higher, -1 on a day it
        closed lower, 0 otherwise), so it measures how persistently that ticker has been rising
        or falling.

        The formula is as follows:

        - McClellan Oscillator = EMA(Advance/Decline Signal, short_ema_window) — EMA(Advance/Decline
          Signal, long_ema_window)

        Also known as: McClellan oscillator, market breadth.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            short_ema_window (int, optional): The window size for the short-term EMA.
                Defaults to 19.
            long_ema_window (int, optional): The window size for the long-term EMA.
                Defaults to 39.
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
            pd.Series or pd.DataFrame: McClellan Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the McClellan Oscillator for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["EWJ", "EWG", "EWU", "EWQ", "EWZ", "INDA", "FXI", "EWC", "EWA", "EWY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_mcclellan_oscillator().loc[:, ["EWJ", "EWG", "EWZ", "INDA", "FXI", "Benchmark"]].tail(5)
        ```

        Which returns:

        | Date       |     EWJ |    EWG |     EWZ |    INDA |    FXI |   Benchmark |
        |:-----------|--------:|-------:|--------:|--------:|-------:|------------:|
        | 2025-12-24 | -0.0357 | 0.1365 | -0.0693 |  0.0631 | 0.0419 |      0.0718 |
        | 2025-12-26 | -0.0875 | 0.1604 | -0.0203 | -0.002  | 0.0863 |      0.0008 |
        | 2025-12-29 | -0.0314 | 0.0801 | -0.0783 | -0.0576 | 0.0239 |     -0.06   |
        | 2025-12-30 | -0.0832 | 0.1111 | -0.0275 | -0.0049 | 0.0703 |     -0.1116 |
        | 2025-12-31 | -0.1272 | 0.037  | -0.0839 |  0.0402 | 0.0097 |     -0.1552 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        mcclellan_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            breadth_model.get_mcclellan_oscillator,
            short_ema_window,
            long_ema_window,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=mcclellan_oscillator,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_advancers_decliners(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the daily advance/decline signal for a given price series.

        Traditionally advancers versus decliners is a market-wide count of rising against
        falling stocks; applied to a single ticker it becomes a directional signal: +1 on a
        day the ticker closed higher than the day before, -1 on a day it closed lower and 0
        on an unchanged day. It feeds the McClellan Oscillator; get_trin and
        get_new_highs_new_lows measure breadth across all tickers.

        The formula is as follows:

        - Signal = +1 if Close(t) > Close(t-1), -1 if Close(t) < Close(t-1), else 0

        Also known as: advance decline ratio, market breadth.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
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
            pd.Series or pd.DataFrame: Advance/decline signal values (+1, 0 or -1).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the advance/decline signal for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["EWJ", "EWG", "EWU", "EWQ", "EWZ", "INDA", "FXI", "EWC", "EWA", "EWY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_advancers_decliners().loc[:, ["EWJ", "EWG", "EWZ", "INDA", "FXI", "Benchmark"]].tail(5)
        ```

        Which returns:

        | Date       |   EWJ |   EWG |   EWZ |   INDA |   FXI |   Benchmark |
        |:-----------|------:|------:|------:|-------:|------:|------------:|
        | 2025-12-24 |    -1 |     1 |    -1 |     -1 |     1 |           1 |
        | 2025-12-26 |    -1 |     1 |     1 |     -1 |     1 |          -1 |
        | 2025-12-29 |     1 |    -1 |    -1 |     -1 |    -1 |          -1 |
        | 2025-12-30 |    -1 |     1 |     1 |      1 |     1 |          -1 |
        | 2025-12-31 |    -1 |    -1 |    -1 |      1 |    -1 |          -1 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        advancers_decliners = calculate_per_ticker(
            historical_data, [close_column], breadth_model.get_advancers_decliners
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=advancers_decliners,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_on_balance_volume(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the On-Balance Volume (OBV) for a given price series.

        The On-Balance Volume (OBV) is a technical indicator that uses volume flow to predict changes in stock price.
        It accumulates the volume on up days and subtracts the volume on down days. The resulting OBV line provides
        insights into the buying and selling pressure behind price movements.

        The formula is as follows:

        - OBV = Previous OBV + Volume if Close > Previous Close, Previous OBV — Volume if Close < Previous Close,
          and Previous OBV otherwise.

        The level of OBV has no meaning by itself, since it depends on the first date of the data; only its
        direction and its divergences from price matter. A rising OBV while price moves sideways is read as
        accumulation ahead of a breakout, a falling OBV as distribution. A single day of very high volume, such
        as an index rebalance or an earnings day, can move the line for a long time.

        For more information, see the following sources:

        - TA-Lib: OBV, the reference implementation (https://ta-lib.org/functions/obv.html).

        Also known as: OBV, volume momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the OBV.
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
            pd.Series or pd.DataFrame: On-Balance Volume values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates On-Balance Volume
          for each asset in the Toolkit instance.
        - TA-Lib's OBV starts its running total at the first period's volume instead of 0, so it is
          the On-Balance Volume here plus that volume; every change is equal.
        - If `growth` is set to True, the method calculates the growth of the OBV using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_on_balance_volume().tail(5)
        ```

        Which returns:

        | Date       |       BTCUSD |     Benchmark |
        |:-----------|-------------:|--------------:|
        | 2025-12-27 | -1.15162e+12 | nan           |
        | 2025-12-28 | -1.13646e+12 | nan           |
        | 2025-12-29 | -1.18487e+12 |   1.32079e+09 |
        | 2025-12-30 | -1.14929e+12 |   1.27363e+09 |
        | 2025-12-31 | -1.18312e+12 |   1.19948e+09 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._historical_data[period].loc[self._start_date :]

        on_balance_volume = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_on_balance_volume,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=on_balance_volume,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_price_volume_trend(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Price Volume Trend (PVT) for a given price series.

        The Price Volume Trend is a running total of volume weighted by the period's percentage
        change in price. On-Balance Volume adds or subtracts the whole volume depending on the
        direction of the close; the PVT adds only the share of the volume that matches the size of
        the move, so a small change in price moves the line only a little.

        The formula is as follows:

        - PVT = Previous PVT + Volume × (Close — Previous Close) / Previous Close
        - PVT = 0 on the first date, which has no previous close

        The level of the PVT has no meaning by itself, since it starts at 0 on the start date; only
        its direction and its divergences from price matter. A rising PVT confirms an uptrend carried
        by volume, and a PVT that fails to follow price to a new high warns that the move lacks
        participation. The line is in shares, so it cannot be compared across stocks.

        For more information, see the following sources:

        - TA-Lib: PVT, the reference implementation (https://ta-lib.org/functions/pvt.html).

        Also known as: PVT, Price and Volume Trend, Volume Price Trend.

        See `breadth_model.get_price_volume_trend` for the references.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", so that the change in price is the return including dividends.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the PVT.
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
            pd.Series or pd.DataFrame: Price Volume Trend values, in shares.

        Notes:
        - The running total starts at 0 on each ticker's first date with a close and a volume on or
          after the start date of the Toolkit, as in TA-Lib's PVT, which it equals.
        - If `growth` is set to True, the method calculates the growth of the PVT using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_price_volume_trend().tail(5)
        ```

        Which returns:

        | Date       |        AAPL |        MSFT |    Benchmark |
        |:-----------|------------:|------------:|-------------:|
        | 2025-12-24 | 6.67008e+07 | 2.02096e+07 | -9.46523e+06 |
        | 2025-12-26 | 6.66686e+07 | 2.02039e+07 | -9.46945e+06 |
        | 2025-12-29 | 6.66998e+07 | 2.01903e+07 | -9.69239e+06 |
        | 2025-12-30 | 6.66449e+07 | 2.02012e+07 | -9.74998e+06 |
        | 2025-12-31 | 6.65229e+07 | 2.00776e+07 | -1.02993e+07 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._historical_data[period].loc[self._start_date :]

        price_volume_trend = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_price_volume_trend,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=price_volume_trend,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_accumulation_distribution_line(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Accumulation/Distribution Line for a given price series.

        The Accumulation/Distribution Line is a technical indicator that evaluates the flow of money
        into or out of an asset. It takes into account both price and volume information to identify
        whether an asset is being accumulated (bought) or distributed (sold) by investors.

        The formula is as follows:

        - Money Flow Multiplier = ((Close — Low) — (High — Close)) / (High — Low)
        - Money Flow Volume = Money Flow Multiplier * Volume
        - ADL = Previous ADL + Money Flow Volume

        For more information, see the following sources:

        - TA-Lib: AD, the reference implementation (https://ta-lib.org/functions/ad.html).

        Also known as: ADL, Chaikin ADL, volume-price trend.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Accumulation/Distribution Line.
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
            pd.Series or pd.DataFrame: Accumulation/Distribution Line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Accumulation/Distribution Line for each asset in the Toolkit instance.
        - The Accumulation/Distribution Line equals TA-Lib's AD.
        - If `growth` is set to True, the method calculates the growth of the Accumulation/Distribution Line
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_accumulation_distribution_line().tail(5)
        ```

        Which returns:

        | Date       |         GLD |   Benchmark |
        |:-----------|------------:|------------:|
        | 2025-12-24 | 7.95203e+08 | 8.01111e+09 |
        | 2025-12-26 | 7.95996e+08 | 8.00572e+09 |
        | 2025-12-29 | 7.9136e+08  | 8.01431e+09 |
        | 2025-12-30 | 7.82463e+08 | 7.98763e+09 |
        | 2025-12-31 | 7.75502e+08 | 7.919e+09   |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._get_historical_data(period, close_column).loc[
            self._start_date :
        ]

        accumulation_distribution_line = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_accumulation_distribution_line,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=accumulation_distribution_line,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_chaikin_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 3,
        long_window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chaikin Oscillator for a given price series.

        The Chaikin Oscillator is a momentum-based indicator that combines price and volume
        to help identify potential trends and reversals in the market. It is calculated as the
        difference between the 3-day and 10-day Accumulation/Distribution Line.

        The formula is as follows:

        - Chaikin Oscillator = EMA(short-window ADL) — EMA(long-window ADL)

        For more information, see the following sources:

        - TA-Lib: ADOSC, the reference implementation (https://ta-lib.org/functions/adosc.html).

        Also known as: Chaikin oscillator, volume accumulation.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            short_window (int, optional): Number of periods for the short-term moving average.
                Defaults to 3.
            long_window (int, optional): Number of periods for the long-term moving average.
                Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Chaikin Oscillator.
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
            pd.Series or pd.DataFrame: Chaikin Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Chaikin Oscillator for each asset in the Toolkit instance.
        - Both EMAs are seeded with the simple average of their first values, as on StockCharts,
          while TA-Lib's ADOSC starts them from the first Accumulation/Distribution value, so the
          first values differ from TA-Lib's until the two converge (within 0.01% after about 65
          periods with the default windows).
        - If `growth` is set to True, the method calculates the growth of the Chaikin Oscillator
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chaikin_oscillator().tail(5)
        ```

        Which returns:

        | Date       |          QQQ |   Benchmark |
        |:-----------|-------------:|------------:|
        | 2025-12-24 |  1.33735e+07 | 4.48979e+07 |
        | 2025-12-26 |  1.24876e+07 | 4.67752e+07 |
        | 2025-12-29 |  1.15765e+07 | 4.60255e+07 |
        | 2025-12-30 |  1.62538e+06 | 3.30466e+07 |
        | 2025-12-31 | -1.44174e+07 | 2.89497e+06 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        chaikin_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_chaikin_oscillator,
            short_window,
            long_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=chaikin_oscillator,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_chaikin_money_flow(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chaikin Money Flow (CMF) for a given price series.

        The Chaikin Money Flow sums the same Money Flow Volume used by the Accumulation/
        Distribution Line over a rolling window and normalizes it by the window's total
        volume, turning the running (unbounded) Accumulation/Distribution Line into a
        bounded oscillator. Sustained readings above zero indicate buying pressure
        (accumulation) is dominating over the window, while sustained readings below zero
        indicate selling pressure (distribution).

        The formula is as follows:

        - Money Flow Multiplier = ((Close — Low) — (High — Close)) / (High — Low)
        - Money Flow Volume = Money Flow Multiplier * Volume
        - CMF = Sum(Money Flow Volume, window) / Sum(Volume, window)

        For more information, see the following sources:

        - TA-Lib: CMF, the reference implementation (https://ta-lib.org/functions/cmf.html).

        Also known as: CMF, Chaikin Money Flow.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods to sum the Money Flow Volume and
                volume over. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Chaikin Money Flow.
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
            pd.Series or pd.DataFrame: Chaikin Money Flow values, bounded between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Chaikin Money Flow for each asset in the Toolkit instance.
        - There is no formal journal citation for the Chaikin Money Flow; the standard
          textbook treatment is Murphy, J.J. (1999). "Technical Analysis of the Financial
          Markets." New York Institute of Finance.
        - The Chaikin Money Flow equals TA-Lib's CMF.
        - If `growth` is set to True, the method calculates the growth of the Chaikin Money
          Flow using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["ETHUSD", "SOLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chaikin_money_flow().tail(5)
        ```

        Which returns:

        | Date       |   ETHUSD |   SOLUSD |   Benchmark |
        |:-----------|---------:|---------:|------------:|
        | 2025-12-27 |  -0.0938 |  -0.143  |    nan      |
        | 2025-12-28 |  -0.1006 |  -0.0945 |    nan      |
        | 2025-12-29 |  -0.1815 |  -0.1419 |      0.0671 |
        | 2025-12-30 |  -0.1279 |  -0.0749 |      0.0614 |
        | 2025-12-31 |  -0.1646 |  -0.1521 |      0.017  |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        chaikin_money_flow = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_chaikin_money_flow,
            window,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=chaikin_money_flow,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_ease_of_movement(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        volume_divisor: float = 100_000_000,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Ease of Movement (EMV) for a given price series.

        The Ease of Movement indicator relates how far price moved (the change in the
        midpoint of the high-low range from one period to the next) to the volume required
        to move it (via the "Box Ratio", volume scaled down and divided by the period's
        high-low range). High positive readings mean price is moving up easily on relatively
        little volume; high negative readings mean price is moving down easily on relatively
        little volume. The raw daily reading is smoothed with a Simple Moving Average to
        reduce noise.

        The formula is as follows:

        - Distance Moved = (High(t) + Low(t)) / 2 — (High(t-1) + Low(t-1)) / 2
        - Box Ratio = (Volume / volume_divisor) / (High — Low)
        - Raw EMV = Distance Moved / Box Ratio
        - EMV = SMA(Raw EMV, window)

        Also known as: EMV, Ease of Movement.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods used to smooth the raw Ease of
                Movement values with a Simple Moving Average. Defaults to 14.
            volume_divisor (float, optional): Scaling constant applied to volume so that the
                Box Ratio (and therefore EMV) stays in a readable range regardless of an
                asset's typical share volume. Defaults to 100,000,000.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Ease of Movement.
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
            pd.Series or pd.DataFrame: Ease of Movement values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ease of Movement for each asset in the Toolkit instance.
        - Reference: Arms, R.W. (1989). "The Arms Index (TRIN): An Introduction to the
          Volume Analysis of Stock and Bond Markets." Business One Irwin.
        - TA-Lib: EMV, the reference implementation (https://ta-lib.org/functions/emv.html).
        - The Ease of Movement equals TA-Lib's EMV with its `volumedivisor` set to `volume_divisor`
          (TA-Lib's default is 10,000).
        - If `growth` is set to True, the method calculates the growth of the Ease of
          Movement using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_ease_of_movement().tail(5)
        ```

        Which returns:

        | Date       |   ^GSPC |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 |  1.404  |      2.8161 |
        | 2025-12-26 |  0.6053 |      2.227  |
        | 2025-12-29 |  0.6173 |      2.7602 |
        | 2025-12-30 |  0.6371 |      2.7794 |
        | 2025-12-31 | -4.2755 |      0.4354 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        ease_of_movement = calculate_per_ticker(
            historical_data,
            ["High", "Low", "Volume"],
            breadth_model.get_ease_of_movement,
            window,
            volume_divisor,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=ease_of_movement,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_negative_volume_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        start_value: float = 1000.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Negative Volume Index (NVI) for a given price series.

        The Negative Volume Index is a cumulative index that only updates on days where
        volume decreases from the prior period, compounding that day's percentage price
        change onto the running index; on days where volume increases (or stays flat), the
        index is carried forward unchanged. The premise, per Fosback, is that "smart money"
        tends to be active on low-volume (quiet) days, so tracking price behaviour
        specifically on those days isolates informed trading from the noise of high-volume,
        crowd-driven days.

        The formula is as follows:

        - Index(t) = Index(t-1) * (1 + (Close(t) / Close(t-1) — 1)) if Volume(t) < Volume(t-1)
        - Index(t) = Index(t-1) otherwise

        Also known as: NVI, Negative Volume Index.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            start_value (float, optional): The index value to start the series at.
                Defaults to 1000.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Negative Volume Index.
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
            pd.Series or pd.DataFrame: Negative Volume Index values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Negative Volume Index for each asset in the Toolkit instance.
        - Reference: Fosback, N.G. (1976). "Stock Market Logic: A Sophisticated Approach to
          Profits on Wall Street." The Institute for Econometric Research.
        - TA-Lib: NVI, the reference implementation (https://ta-lib.org/functions/nvi.html).
        - With the default `start_value` of 1,000, the Negative Volume Index equals TA-Lib's NVI.
        - If `growth` is set to True, the method calculates the growth of the Negative Volume
          Index using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SPY", "IWM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.technicals.get_negative_volume_index().tail(5)
        ```

        Which returns:

        | Date       |     SPY |     IWM |
        |:-----------|--------:|--------:|
        | 2025-12-24 | 5440.48 | 4501.78 |
        | 2025-12-26 | 5440.48 | 4501.78 |
        | 2025-12-29 | 5440.48 | 4501.78 |
        | 2025-12-30 | 5433.84 | 4468.59 |
        | 2025-12-31 | 5433.84 | 4468.59 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._historical_data[period].loc[self._start_date :]

        negative_volume_index = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_negative_volume_index,
            start_value,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=negative_volume_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_positive_volume_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        start_value: float = 1000.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Positive Volume Index (PVI) for a given price series.

        The Positive Volume Index mirrors the Negative Volume Index: it is a cumulative
        index that only updates on days where volume increases from the prior period,
        compounding that day's percentage price change onto the running index; on days where
        volume decreases (or stays flat), the index is carried forward unchanged. Per
        Fosback, the Positive Volume Index isolates price behaviour on high-volume
        (crowd-driven) days, which is traditionally read as tracking less-informed,
        sentiment-driven trading.

        The formula is as follows:

        - Index(t) = Index(t-1) * (1 + (Close(t) / Close(t-1) — 1)) if Volume(t) > Volume(t-1)
        - Index(t) = Index(t-1) otherwise

        Also known as: PVI, Positive Volume Index.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            start_value (float, optional): The index value to start the series at.
                Defaults to 1000.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Positive Volume Index.
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
            pd.Series or pd.DataFrame: Positive Volume Index values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Positive Volume Index for each asset in the Toolkit instance.
        - Reference: Fosback, N.G. (1976). "Stock Market Logic: A Sophisticated Approach to
          Profits on Wall Street." The Institute for Econometric Research.
        - TA-Lib: PVI, the reference implementation (https://ta-lib.org/functions/pvi.html).
        - With the default `start_value` of 1,000, the Positive Volume Index equals TA-Lib's PVI.
        - If `growth` is set to True, the method calculates the growth of the Positive Volume
          Index using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SPY", "IWM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.technicals.get_positive_volume_index().tail(5)
        ```

        Which returns:

        | Date       |     SPY |     IWM |
        |:-----------|--------:|--------:|
        | 2025-12-24 | 368.611 | 308.515 |
        | 2025-12-26 | 368.574 | 306.942 |
        | 2025-12-29 | 367.26  | 305.061 |
        | 2025-12-30 | 367.26  | 305.061 |
        | 2025-12-31 | 364.539 | 302.758 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._historical_data[period].loc[self._start_date :]

        positive_volume_index = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_positive_volume_index,
            start_value,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=positive_volume_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_trin(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the TRIN (Arms Index) for a given price series.

        TRIN compares the ratio of advancing to declining issues against the ratio of
        volume in advancing issues to volume in declining issues. It is a market-wide
        breadth reading computed across all tickers in the Toolkit instance (excluding
        the synthetic "Portfolio" and "Benchmark" columns), and the resulting single
        reading is broadcast to every ticker column so it lines up with the other
        breadth indicators.

        The formula is as follows:

        - TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)

        A TRIN below 1 means the volume in rising stocks outweighs their number, which is read
        as buying pressure, and above 1 as selling pressure; readings far above 2 tend to come
        with capitulation days. It is meant for a broad universe, such as all stocks of an
        exchange. Calculated across a handful of tickers it jumps between extremes and is
        undefined on days when none of them fell.

        A ticker counts on the dates it traded, with its change since its own previous close:
        on a date only a cryptocurrency in the Toolkit traded, such as a weekend, only the
        cryptocurrency counts, and a stock is never counted as unchanged on it.

        Also known as: Arms Index, TRIN.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
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
            pd.Series or pd.DataFrame: TRIN values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the TRIN across all tickers in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # TRIN is one market-wide reading, repeated in every ticker column
        toolkit.technicals.get_trin().loc[:, ["XLK"]].tail(5)
        ```

        Which returns:

        | Date       |      XLK |
        |:-----------|---------:|
        | 2025-12-24 |   2.485  |
        | 2025-12-26 |   1.6498 |
        | 2025-12-29 |   0.6578 |
        | 2025-12-30 |   0.6376 |
        | 2025-12-31 | nan      |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        constituents = [
            ticker
            for ticker in historical_data[close_column].columns
            if ticker not in ("Portfolio", "Benchmark")
        ]

        # NaN on the dates a constituent did not trade, rather than its carried-forward close, so it is not counted as unchanged on another ticker's trading days.  # noqa: E501
        closes = historical_data[close_column].mask(
            get_non_trading_days(historical_data, close_column)
        )

        trin_series = breadth_model.get_trin(
            closes[constituents],
            historical_data["Volume"][constituents],
        ).loc[self._start_date : self._end_date]

        trin = pd.DataFrame(
            {ticker: trin_series for ticker in historical_data[close_column].columns},
            index=trin_series.index,
        )

        return finalize_dataset(
            dataset=trin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_portfolio
    @handle_errors
    def get_new_highs_new_lows(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 252,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the New Highs - New Lows for a given price series.

        New Highs - New Lows measures the number of tickers reaching a new high over
        the specified window minus the number of tickers reaching a new low over the
        same window. It is a market-wide breadth reading computed across all tickers in
        the Toolkit instance (excluding the synthetic "Portfolio" and "Benchmark"
        columns), and the resulting single reading is broadcast to every ticker column
        so it lines up with the other breadth indicators.

        The formula is as follows:

        - New Highs - New Lows = (Number of tickers at a window-period high) - (Number of tickers at a window-period low)

        The reading counts tickers at a high or low of the lookback window, so it needs a broad
        universe and a history of at least one window before the start of the period of interest:
        in the first periods every ticker is compared with only a few prices, so it is often counted
        at a high or a low, and on the first date at both. Persistently more new lows than new highs
        while an index still rises is a classic sign of narrowing breadth.

        Each ticker's window counts its own closes and the ticker counts on the dates it traded,
        so the weekends of a cryptocurrency in the Toolkit do not shorten a stock's window.

        Also known as: new highs minus new lows, record high percent.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the new high / new low lookback.
                Defaults to 252.
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
            pd.Series or pd.DataFrame: New Highs - New Lows values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the New Highs - New Lows across all tickers in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-10-01",
            end_date="2025-12-31",
        )

        # New Highs - New Lows is one market-wide reading, repeated in every ticker column
        toolkit.technicals.get_new_highs_new_lows(window=20).loc[:, ["XLK"]].tail(5)
        ```

        Which returns:

        | Date       |   XLK |
        |:-----------|------:|
        | 2025-12-24 |     5 |
        | 2025-12-26 |     1 |
        | 2025-12-29 |     0 |
        | 2025-12-30 |     2 |
        | 2025-12-31 |     0 |
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        constituents = [
            ticker
            for ticker in historical_data[close_column].columns
            if ticker not in ("Portfolio", "Benchmark")
        ]

        # NaN on the dates a constituent did not trade, so its window counts its own closes.
        closes = historical_data[close_column].mask(
            get_non_trading_days(historical_data, close_column)
        )

        new_highs_new_lows_series = breadth_model.get_new_highs_new_lows(
            closes[constituents],
            window,
        ).loc[self._start_date : self._end_date]

        new_highs_new_lows = pd.DataFrame(
            {
                ticker: new_highs_new_lows_series
                for ticker in historical_data[close_column].columns
            },
            index=new_highs_new_lows_series.index,
        )

        return finalize_dataset(
            dataset=new_highs_new_lows,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            apply_slice=False,
        )

    @handle_errors
    def get_market_breadth(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 50,
        long_window: int = 200,
        high_low_window: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the breadth of the market the Toolkit's tickers make up: how many of them take
        part in a move. An index can rise on the strength of a few large companies while most
        of its members fall; breadth shows that divergence, which often comes before the index
        itself turns. Breadth is most meaningful across many tickers, such as the constituents
        of an index, which `Discovery.get_index_constituents` lists (e.g. the 500 tickers of the
        S&P 500). Today's members leave out the companies that were removed from the index
        since (often after falling behind), so breadth of past dates computed from them is
        biased towards the survivors, the further back the more; its `date=` gives the members
        on the first date of the window instead.

        Per date it returns:

        - Advancers, Decliners and Unchanged: the number of tickers whose close rose, fell or
          stayed the same, and the Net Advances.
        - Advance-Decline Line: the running total of the net advances. A rising index with a
          falling line is the classic negative divergence.
        - Advance-Decline Ratio: the advancers divided by the decliners.
        - McClellan Oscillator: the ratio-adjusted version, the 19-period minus the 39-period
          exponential moving average (smoothing constants of 0.10 and 0.05) of the net advances
          divided by advancers plus decliners (times 1,000); unchanged issues are left out.
          Each average starts from the simple average of its first 19 or 39 values, so the
          oscillator starts once the 39-period average exists (NaN before), as does its running
          total, the McClellan Summation Index.
        - Percentage Above the short and long moving average: the share of tickers trading
          above their 50 and 200-period average, a gauge of how broad a trend is.
        - New Highs and New Lows: the tickers whose close is above the highest (below the
          lowest) close of the previous `high_low_window` - 1 periods, so the highest or
          lowest of the last `high_low_window` periods; a close that only equals the previous
          high is no new high. NaN until at least one ticker has a full window.

        The Benchmark (and a Portfolio) are left out. A constituent counts on the dates it
        traded, and its changes, moving averages and highs and lows run over its own closes: on
        a date only a cryptocurrency in the Toolkit traded, such as a weekend, only the
        cryptocurrency counts, and the McClellan averages then skip that date (see
        `breadth_model.get_market_breadth`).

        For more information, see the following book:

        - McClellan, S. and McClellan, M. (1970). "Patterns for Profit: The McClellan
          Oscillator and Summation Index." Trade Levels Inc.

        Also known as: market breadth, advance-decline line, McClellan oscillator, percentage
        above 200-day moving average, new highs new lows.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            short_window (int, optional): The short moving average window. Defaults to 50.
            long_window (int, optional): The long moving average window. Defaults to 200.
            high_low_window (int | None, optional): The number of periods, including the current
                one, of the new highs and lows. Defaults to None, which uses a year of periods:
                252 daily, 52 weekly, 12 monthly, 4 quarterly and 1 yearly (too short for a new
                high or low, so those stay NaN), and 252 for intraday data.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The breadth measures per date.

        As an example:

        ```python
        from financetoolkit import Discovery, Toolkit

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        constituents = discovery.get_index_constituents("Dow Jones")

        toolkit = Toolkit(
            constituents.index.tolist(),
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_market_breadth().tail()
        ```

        Which returns:

        | Date       |   Advancers |   Decliners |   Unchanged |   Net Advances |   Advance-Decline Line |   Advance-Decline Ratio |   McClellan Oscillator |   McClellan Summation Index |   Percentage Above 50-Period Average |   Percentage Above 200-Period Average |   New Highs |   New Lows |
        |:-----------|------------:|------------:|------------:|---------------:|-----------------------:|------------------------:|-----------------------:|----------------------------:|-------------------------------------:|--------------------------------------:|------------:|-----------:|
        | 2025-12-24 |          26 |           3 |           1 |             23 |                    978 |                  8.6667 |                65.5048 |                    -814.088 |                               0.7333 |                                   0.8 |           3 |          0 |
        | 2025-12-26 |          14 |          16 |           0 |             -2 |                    976 |                  0.875  |                49.8424 |                    -764.246 |                               0.7    |                                   0.8 |           1 |          0 |
        | 2025-12-29 |          10 |          20 |           0 |            -10 |                    966 |                  0.5    |                22.8684 |                    -741.377 |                               0.7333 |                                   0.8 |           0 |          0 |
        | 2025-12-30 |           9 |          21 |           0 |            -12 |                    954 |                  0.4286 |                -3.642  |                    -745.019 |                               0.7    |                                   0.8 |           0 |          0 |
        | 2025-12-31 |           3 |          27 |           0 |            -24 |                    930 |                  0.1111 |               -46.2902 |                    -791.309 |                               0.5667 |                                   0.8 |           0 |          0 |
        """
        # The Advance-Decline Line and McClellan Summation Index start at the start date.
        # NaN on the dates a constituent did not trade, so a stock is not counted as unchanged on the weekends of a cryptocurrency.  # noqa: E501
        prices = self._close_prices(period, close_column, traded_only=True).loc[
            self._start_date :
        ]
        constituents = [
            ticker
            for ticker in prices.columns
            if ticker not in ("Portfolio", "Benchmark")
        ]
        # A year of periods, so weekly data looks for 52-week highs rather than 252-week ones. Intraday bars have no fixed number per year and keep the daily count.  # noqa: E501
        if high_low_window is None:
            high_low_window = PERIODS_PER_YEAR.get(period, PERIODS_PER_YEAR["daily"])
        breadth = breadth_model.get_market_breadth(
            prices[constituents], short_window, long_window, high_low_window
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            breadth, rounding if rounding is not None else self._rounding
        )
