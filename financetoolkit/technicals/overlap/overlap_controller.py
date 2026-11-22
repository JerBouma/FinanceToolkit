"""Overlap Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.technicals.helpers import calculate_per_ticker
from financetoolkit.technicals.overlap import overlap_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


@filter_shown_columns
@track_sources
class OverlapTechnicals:
    """
    Overlap indicators drawn over the price: the moving averages, TRIX, the Kaufman adaptive moving average, VWAP, the parabolic SAR, pivot points, support and resistance levels and Fibonacci retracement levels.
    """

    @handle_errors
    def collect_overlap_indicators(
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
        Calculates and collects various overlap-based indicators based on the provided data.

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
            pd.Series or pd.DataFrame: Overlap-based indicators calculated based on the specified parameters.

        Notes:
        - The method calculates several overlap-based indicators for each asset in the Toolkit instance.
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

        toolkit.technicals.collect_overlap_indicators().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Simple Moving Average (SMA) |   Exponential Moving Average (EMA) |   Double Exponential Moving Average (DEMA) |   TRIX |   Triangular Moving Average |   Weighted Moving Average (WMA) |   Hull Moving Average (HMA) |   Kaufman Adaptive Moving Average (KAMA) |   Volume Weighted Average Price (VWAP) |   Parabolic SAR |   Pivot Point |   Pivot Point Resistance 1 |   Pivot Point Support 1 |   Fibonacci Retracement 50.0% |   Fibonacci Retracement 61.8% |
        |:-----------|------------------------------:|-----------------------------------:|-------------------------------------------:|-------:|----------------------------:|--------------------------------:|----------------------------:|-----------------------------------------:|---------------------------------------:|----------------:|--------------:|---------------------------:|------------------------:|------------------------------:|------------------------------:|
        | 2025-12-17 |                       277.993 |                            275.523 |                                    275.977 | 0.0016 |                     278.193 |                         276.338 |                     273.323 |                                  274.426 |                                278.186 |         285.71  |       273.223 |                    275.394 |                 271.694 |                       279.37  |                       277.371 |
        | 2025-12-18 |                       277.518 |                            274.98  |                                    274.903 | 0.0014 |                     277.449 |                         275.466 |                     272.107 |                                  274.289 |                                277.532 |         284.525 |       272.472 |                    274.041 |                 269.533 |                       277.031 |                       274.481 |
        | 2025-12-19 |                       276.846 |                            274.706 |                                    274.402 | 0.0012 |                     276.642 |                         274.854 |                     271.353 |                                  274.166 |                                276.211 |         282.695 |       270.188 |                    274.15  |                 267.489 |                       277.031 |                       274.481 |
        | 2025-12-22 |                       275.762 |                            274.11  |                                    273.33  | 0.0009 |                     275.788 |                         273.972 |                     270.604 |                                  273.358 |                                275.272 |         281.048 |       271.983 |                    274.799 |                 270.111 |                       277.031 |                       274.481 |
        | 2025-12-23 |                       274.922 |                            273.778 |                                    272.814 | 0.0007 |                     274.973 |                         273.42  |                     270.19  |                                  273.115 |                                274.464 |         279.566 |       271.049 |                    272.322 |                 268.961 |                       275.091 |                       272.999 |
        | 2025-12-24 |                       274.432 |                            273.683 |                                    272.766 | 0.0006 |                     274.257 |                         273.173 |                     270.366 |                                  273.112 |                                274.006 |         278.232 |       270.736 |                    272.645 |                 269.713 |                       273.301 |                       271.631 |
        | 2025-12-26 |                       274.048 |                            273.547 |                                    272.633 | 0.0004 |                     273.637 |                         272.936 |                     270.87  |                                  273.087 |                                273.626 |         277.031 |       273.07  |                    274.679 |                 271.458 |                       272.807 |                       271.254 |
        | 2025-12-29 |                       273.754 |                            273.476 |                                    272.623 | 0.0003 |                     273.126 |                         272.798 |                     271.532 |                                  273.085 |                                273.368 |         275.951 |       273.133 |                    274.147 |                 271.644 |                       272.807 |                       271.254 |
        | 2025-12-30 |                       273.462 |                            273.324 |                                    272.454 | 0.0002 |                     272.738 |                         272.61  |                     272.005 |                                  273.056 |                                273.119 |         274.978 |       272.748 |                    273.885 |                 271.88  |                       272.807 |                       271.254 |
        | 2025-12-31 |                       272.969 |                            273.031 |                                    272.022 | 0.0001 |                     272.407 |                         272.298 |                     272.09  |                                  272.894 |                                272.801 |         274.103 |       272.405 |                    273.269 |                 271.474 |                       272.807 |                       271.254 |
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

        overlap_indicators: dict = {}

        overlap_indicators["Simple Moving Average (SMA)"] = self.get_moving_average(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators["Exponential Moving Average (EMA)"] = (
            self.get_moving_average(
                period=period,
                close_column=close_column,
                window=window,
                method="exponential",
            )
        )

        overlap_indicators["Double Exponential Moving Average (DEMA)"] = (
            self.get_moving_average(
                period=period,
                close_column=close_column,
                window=window,
                method="double_exponential",
            )
        )

        overlap_indicators["TRIX"] = self.get_trix(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators["Triangular Moving Average"] = self.get_moving_average(
            period=period,
            close_column=close_column,
            window=window,
            method="triangular",
        )

        overlap_indicators["Weighted Moving Average (WMA)"] = self.get_moving_average(
            period=period,
            close_column=close_column,
            window=window,
            method="weighted",
        )

        overlap_indicators["Hull Moving Average (HMA)"] = self.get_moving_average(
            period=period,
            close_column=close_column,
            window=window,
            method="hull",
        )

        overlap_indicators["Kaufman Adaptive Moving Average (KAMA)"] = (
            self.get_kaufman_adaptive_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Volume Weighted Average Price (VWAP)"] = (
            self.get_volume_weighted_average_price(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Parabolic SAR"] = self.get_parabolic_sar(
            period=period, close_column=close_column
        )

        pivot_points = self.get_pivot_points(period=period, close_column=close_column)

        overlap_indicators["Pivot Point"] = pivot_points["Pivot Point"]
        overlap_indicators["Pivot Point Resistance 1"] = pivot_points["Resistance 1"]
        overlap_indicators["Pivot Point Support 1"] = pivot_points["Support 1"]

        fibonacci_retracement_levels = self.get_fibonacci_retracement_levels(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators["Fibonacci Retracement 50.0%"] = (
            fibonacci_retracement_levels["50.0%"]
        )
        overlap_indicators["Fibonacci Retracement 61.8%"] = (
            fibonacci_retracement_levels["61.8%"]
        )

        self._overlap_indicators = pd.concat(overlap_indicators, axis=1)

        self._overlap_indicators = apply_rounding(
            self._overlap_indicators,
            rounding if rounding is not None else self._rounding,
        ).loc[self._start_date : self._end_date]

        if growth:
            self._overlap_indicators_growth = calculate_growth(
                dataset=self._overlap_indicators,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="index",
            )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                self._overlap_indicators_growth = calculate_standardization(
                    dataset=self._overlap_indicators_growth,
                    rounding=standardize_rounding,
                    axis="rows",
                )
            else:
                self._overlap_indicators = calculate_standardization(
                    dataset=self._overlap_indicators,
                    rounding=standardize_rounding,
                    axis="rows",
                )

        if len(self._tickers) == 1:
            return (
                self._overlap_indicators_growth.xs(
                    self._tickers[0], level=1, axis="columns"
                )
                if growth
                else self._overlap_indicators.xs(
                    self._tickers[0], level=1, axis="columns"
                )
            )

        return self._overlap_indicators_growth if growth else self._overlap_indicators

    @handle_portfolio
    @handle_errors
    def get_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        method: str = "simple",
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate a Moving Average (MA) of a given price series: the simple, exponential,
        double exponential, triangular, weighted or Hull moving average.

        A Moving Average smooths out price data by averaging the price over a specified
        number of periods. The variants differ in how they weight the prices within the
        window: equally (simple), with exponentially decaying weights (exponential and double
        exponential), with weights that peak in the middle of the window (triangular), with
        linearly increasing weights (weighted) or with a combination of weighted averages
        that reduces the lag (Hull).

        The formula is as follows, per method:

        - "simple": SMA = (Sum of Prices) / (Number of Prices)
        - "exponential": the first EMA is SMA(Close, Window), on the Window-th period, and
          EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA
        - "double_exponential": DEMA = 2 * EMA(Close, Window) — EMA(EMA(Close, Window), Window)
        - "triangular": TMA = SMA(SMA(Close, Sub-window Length 1), Sub-window Length 2), with
          both sub-windows (Window + 1) / 2 for an odd window and Window / 2 and Window / 2 + 1
          for an even window (TA-Lib's TRIMA convention)
        - "weighted": WMA = (Sum of (Price * Weight)) / (Sum of Weights), with weights 1 to
          Window from the oldest to the latest price
        - "hull": HMA = WMA(2 * WMA(Close, Window / 2) — WMA(Close, Window), sqrt(Window))

        Seeding the EMA with a simple average, as TA-Lib and StockCharts do, leaves the first
        Window - 1 periods of the price history empty. The DEMA, TRIX, MACD, Percentage Price
        Oscillator, Force Index, Elder Ray Index, Keltner Channels and Chaikin and McClellan
        Oscillators use the same EMA and start once their own EMAs are seeded. The Kaufman
        Adaptive Moving Average, which adapts its smoothing to the efficiency of the price
        move, is `get_kaufman_adaptive_moving_average`.

        For more information, see the following sources:

        - TA-Lib: SMA, EMA, DEMA, TRIMA, WMA and HMA, the reference implementations
          (https://ta-lib.org/functions/).

        Also known as: MA, SMA, simple moving average, EMA, exponential moving average, DEMA, double exponential moving average, TMA, TRIMA, triangular moving average, WMA, linearly weighted moving average, HMA, Hull moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the moving average.
                The number of periods (time intervals) over which to calculate the MA. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the MA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): The moving average: "simple", "exponential",
                "double_exponential", "triangular", "weighted" or "hull". Defaults to "simple".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series:
            Moving Average (MA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          MA for each asset in the Toolkit instance.
        - The simple, exponential, double exponential, triangular, weighted and Hull moving averages
          equal TA-Lib's SMA, EMA, DEMA, TRIMA, WMA and HMA.
        - If `growth` is set to True, the method calculates the growth of the MA
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_moving_average(window=50).tail(5)
        ```

        Which returns:

        | Date       |   ^GSPC |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 6784.99 |     669.712 |
        | 2025-12-26 | 6790.16 |     670.25  |
        | 2025-12-29 | 6795.7  |     670.828 |
        | 2025-12-30 | 6800.34 |     671.316 |
        | 2025-12-31 | 6802.55 |     671.566 |
        """
        moving_averages = {
            "simple": overlap_model.get_moving_average,
            "exponential": overlap_model.get_exponential_moving_average,
            "double_exponential": overlap_model.get_double_exponential_moving_average,
            "triangular": overlap_model.get_triangular_moving_average,
            "weighted": overlap_model.get_weighted_moving_average,
            "hull": overlap_model.get_hull_moving_average,
        }
        if method not in moving_averages:
            raise ValueError(
                f"Method must be one of {', '.join(moving_averages)}, not {method}."
            )
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

        moving_average = calculate_per_ticker(
            historical_data, [close_column], moving_averages[method], window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=moving_average,
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
    def get_trix(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 15,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Trix (Triple Exponential Moving Average) for a given price series.

        Trix is a momentum oscillator that calculates the percentage rate of change of a triple
        exponentially smoothed moving average. It helps identify overbought and oversold conditions
        in a market.

        The formula is as follows:

        - EMA1 = EMA(Close, Window)
        - EMA2 = EMA(EMA1, Window)
        - EMA3 = EMA(EMA2, Window)
        - TRIX = (EMA3 — EMA3[—1]) / EMA3[—1]

        Also known as: triple smoothed EMA, rate of change oscillator.

        Reference: Hutson, J. (1983). "TRIX — Triple Exponential Smoothing Oscillator."
        Technical Analysis of Stocks & Commodities, 1(5).

        TA-Lib: TRIX, the reference implementation (https://ta-lib.org/functions/trix.html).

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of each of the three exponential moving
                averages. Defaults to 15, the window of Hutson (1983) and StockCharts.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Trix.
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
            pd.DataFrame or pd.Series:
            Trix values as a decimal (0.001 is 0.1%).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Trix for each asset in the Toolkit instance.
        - TRIX is a decimal and equals TA-Lib's TRIX divided by 100, since TA-Lib's TRIX is in
          percent.
        - If `growth` is set to True, the method calculates the growth of the Trix
          using the specified `lag`.
        - Daily TRIX values are small (a few hundredths of a percent), so four decimals keep
          about one significant digit. Pass a higher `rounding` (or None) for more precision.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^IXIC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_trix(rounding=6).tail(5)
        ```

        Which returns:

        | Date       |    ^IXIC |   Benchmark |
        |:-----------|---------:|------------:|
        | 2025-12-24 | 0.000367 |    0.000541 |
        | 2025-12-26 | 0.000402 |    0.000582 |
        | 2025-12-29 | 0.000432 |    0.00062  |
        | 2025-12-30 | 0.000454 |    0.000651 |
        | 2025-12-31 | 0.000452 |    0.00066  |
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

        trix = calculate_per_ticker(
            historical_data, [close_column], overlap_model.get_trix, window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=trix,
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
    def get_kaufman_adaptive_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        fast_window: int = 2,
        slow_window: int = 30,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Kaufman Adaptive Moving Average (KAMA) for a given price series.

        The Kaufman Adaptive Moving Average adjusts its own responsiveness to price changes
        based on how "efficiently" price is moving. It compares the net directional move
        over the window to the total (sum of absolute) movement over that same window — the
        Efficiency Ratio. When price trends strongly in one direction (an efficient move),
        the Efficiency Ratio is close to 1 and KAMA tracks price closely, behaving like a
        fast EMA. When price whipsaws sideways (an inefficient move), the Efficiency Ratio is
        close to 0 and KAMA flattens out, behaving like a slow EMA — reducing whipsaw signals
        in choppy markets while still reacting quickly during strong trends.

        The formula is as follows:

        - Change = |Close(t) — Close(t - window)|
        - Volatility = Sum(|Close(i) — Close(i - 1)|, window)
        - Efficiency Ratio (ER) = Change / Volatility
        - Fastest SC = 2 / (fast_window + 1), Slowest SC = 2 / (slow_window + 1)
        - Smoothing Constant (SC) = [ER * (Fastest SC — Slowest SC) + Slowest SC]^2
        - KAMA(t) = KAMA(t-1) + SC * (Close(t) — KAMA(t-1))

        Also known as: KAMA, Kaufman's Adaptive Moving Average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods over which the Efficiency Ratio is
                calculated. Defaults to 10.
            fast_window (int, optional): The number of periods that corresponds to the
                fastest EMA constant used when the Efficiency Ratio is at its maximum (1.0).
                Defaults to 2.
            slow_window (int, optional): The number of periods that corresponds to the
                slowest EMA constant used when the Efficiency Ratio is at its minimum (0.0).
                Defaults to 30.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the KAMA.
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
            pd.Series or pd.DataFrame: KAMA values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the KAMA for each asset in the Toolkit instance.
        - Reference: Kaufman, P.J. (1998). "Trading Systems and Methods." 3rd ed. Wiley.
        - TA-Lib: KAMA, the reference implementation (https://ta-lib.org/functions/kama.html).
        - With the default `fast_window` of 2 and `slow_window` of 30, which TA-Lib fixes, the KAMA
          equals TA-Lib's KAMA.
        - If `growth` is set to True, the method calculates the growth of the KAMA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^STOXX50E"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_kaufman_adaptive_moving_average().tail(5)
        ```

        Which returns:

        | Date       |   ^STOXX50E |   Benchmark |
        |:-----------|------------:|------------:|
        | 2025-12-24 |      nan    |     674.145 |
        | 2025-12-26 |      nan    |     674.285 |
        | 2025-12-29 |     5725.07 |     674.644 |
        | 2025-12-30 |     5726.77 |     674.963 |
        | 2025-12-31 |      nan    |     674.999 |
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

        kaufman_adaptive_moving_average = calculate_per_ticker(
            historical_data,
            [close_column],
            overlap_model.get_kaufman_adaptive_moving_average,
            window,
            fast_window,
            slow_window,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=kaufman_adaptive_moving_average,
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
    def get_volume_weighted_average_price(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Volume Weighted Average Price (VWAP) for a given price series.

        The Volume Weighted Average Price (VWAP) weighs the typical price of each period
        by its traded volume over a rolling window, giving a more volume-informed view of
        the average price than a plain moving average.

        The formula is as follows:

        - Typical Price = (High + Low + Close) / 3
        - VWAP = Sum(Typical Price * Volume, Window) / Sum(Volume, Window)

        The VWAP that trading desks use as an execution benchmark is calculated from the trades
        of a single session and resets every day. This method computes a rolling VWAP over several
        periods from the high, low, close and volume of each period, which works as a
        volume-weighted moving average, not as the intraday benchmark against which fills are
        measured.

        For more information, see the following sources:

        - TA-Lib: VWAP, the reference implementation (https://ta-lib.org/functions/vwap.html).

        Also known as: VWAP.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods to consider for the VWAP.
                The number of periods (time intervals) over which to calculate the VWAP. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the VWAP.
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
            pd.DataFrame or pd.Series:
            Volume Weighted Average Price (VWAP) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          VWAP for each asset in the Toolkit instance.
        - The VWAP is taken over a rolling `window`. TA-Lib's VWAP accumulates the typical price
          times volume from the first period on, which is the VWAP over a window that grows with the
          data.
        - If `growth` is set to True, the method calculates the growth of the VWAP
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SPY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.technicals.get_volume_weighted_average_price().tail(5)
        ```

        Which returns:

        | Date       |     SPY |
        |:-----------|--------:|
        | 2025-12-24 | 675.242 |
        | 2025-12-26 | 675.346 |
        | 2025-12-29 | 675.664 |
        | 2025-12-30 | 675.891 |
        | 2025-12-31 | 675.847 |
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

        volume_weighted_average_price = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            overlap_model.get_volume_weighted_average_price,
            window,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=volume_weighted_average_price,
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
    def get_parabolic_sar(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Parabolic Stop and Reverse (SAR) for a given price series.

        The Parabolic SAR is a trend-following indicator that trails price action,
        flipping from below to above price (and vice versa) whenever the trend reverses.
        The acceleration factor increases as the trend extends, causing the SAR to
        converge towards price over time.

        The formula is as follows:

        - Uptrend SAR = Prior SAR + AF * (Extreme Point — Prior SAR)
        - Downtrend SAR = Prior SAR — AF * (Prior SAR — Extreme Point)

        For more information, see the following sources:

        - TA-Lib: SAR, the reference implementation (https://ta-lib.org/functions/sar.html).

        Also known as: Parabolic SAR, stop and reverse, PSAR.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            af_start (float, optional): Initial acceleration factor. Defaults to 0.02.
            af_increment (float, optional): Amount by which the acceleration factor
                increases every time a new extreme point is reached. Defaults to 0.02.
            af_max (float, optional): Maximum value the acceleration factor can reach.
                Defaults to 0.2.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Parabolic SAR.
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
            pd.DataFrame or pd.Series:
            Parabolic SAR values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Parabolic SAR for each asset in the Toolkit instance.
        - The Parabolic SAR equals TA-Lib's SAR, which takes `af_increment` equal to `af_start`, and
          its SAREXT for any `af_increment`.
        - If `growth` is set to True, the method calculates the growth of the Parabolic SAR
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["CLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_parabolic_sar().tail(5)
        ```

        Which returns:

        | Date       |   CLUSD |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-26 | 55.5988 |     666.32  |
        | 2025-12-28 | 55.8613 |     nan     |
        | 2025-12-29 | 56.1028 |     667.918 |
        | 2025-12-30 | 56.325  |     669.389 |
        | 2025-12-31 | 56.5294 |     670.742 |
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

        parabolic_sar = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            overlap_model.get_parabolic_sar,
            af_start,
            af_increment,
            af_max,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=parabolic_sar,
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
    def get_pivot_points(
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
        Calculate the Pivot Points for a given price series.

        Pivot Points are calculated from the previous period's high, low and close
        prices and are used to identify potential support and resistance levels for
        the current period.

        The formula is as follows:

        - Pivot Point = (Previous High + Previous Low + Previous Close) / 3
        - Resistance 1 = (2 * Pivot Point) — Previous Low
        - Support 1 = (2 * Pivot Point) — Previous High
        - Resistance 2 = Pivot Point + (Previous High — Previous Low)
        - Support 2 = Pivot Point — (Previous High — Previous Low)
        - Resistance 3 = Previous High + 2 * (Pivot Point — Previous Low)
        - Support 3 = Previous Low — 2 * (Previous High — Pivot Point)

        Also known as: pivot points, floor trader pivots.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Pivot Points.
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
            pd.Series or pd.DataFrame:
            Pivot Points (pivot, resistance 1-3, support 1-3).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Pivot Points for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Pivot Points
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["EURUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        pivot_points = toolkit.technicals.get_pivot_points().xs("EURUSD", level=1, axis="columns")

        pivot_points.loc[:, ["Pivot Point", "Resistance 1", "Resistance 2", "Support 1", "Support 2"]].tail(5)
        ```

        Which returns:

        | Date       |   Pivot Point |   Resistance 1 |   Resistance 2 |   Support 1 |   Support 2 |
        |:-----------|--------------:|---------------:|---------------:|------------:|------------:|
        | 2025-12-26 |        1.1785 |         1.1793 |         1.1802 |      1.1777 |      1.1768 |
        | 2025-12-28 |        1.1776 |         1.1791 |         1.1812 |      1.1756 |      1.1741 |
        | 2025-12-29 |        1.1772 |         1.178  |         1.1794 |      1.1759 |      1.1751 |
        | 2025-12-30 |        1.177  |         1.1791 |         1.181  |      1.1752 |      1.1731 |
        | 2025-12-31 |        1.1757 |         1.1771 |         1.1793 |      1.1734 |      1.1721 |
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

        pivot_points = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            overlap_model.get_pivot_points,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=pivot_points,
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
    def get_support_resistance_levels(
        self,
        sensitivity: float = 0.05,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Retrieves the support and resistance levels for the specified period and assets.

        The Support and Resistance Levels are price levels where the price tends to stop and reverse.

        - Support Levels: These are the valleys where the price tends to stop going down and may start to go up.
        Think of support levels as "floors" that the price has trouble falling below.
        - Resistance Levels: These are the peaks where the price tends to stop going up and may start to go down.
        Think of resistance levels as "ceilings" that the price has trouble breaking through.

        It does so by:

        - Looking for Peaks and Valleys: The function looks at the stock prices and finds the high points
        (peaks) and low points (valleys) over time.
        - Grouping Similar Peaks and Valleys: Sometimes, prices will stop at similar points multiple times.
        The function groups these similar peaks and valleys together to identify key resistance and
        support levels.

        The formula is as follows:

        - Peak (t) = Close (t) if Close (t) > every Close within `window` periods before and after t (14 by default)
        - Valley (t) = Close (t) if Close (t) < every Close within `window` periods before and after t
        - Level = (Existing Level + Peak or Valley) / 2 if |Peak or Valley - Existing Level| / |Existing Level| <
          Sensitivity (0.05 by default), otherwise a new level at the Peak or Valley
        - Resistance (t) = the most recent level from the peaks published up to t, and Support (t) the same for
          the valleys, with each level published `window` periods after its peak or valley (when it is confirmed)

        Where Close is the `close_column`, the Adj Close by default (the Close for intraday data).

        Also known as: support levels, resistance levels, pivot points.

        Args:
            sensitivity (float, optional): The sensitivity parameter to determine the significance of the peaks
                and valleys. A higher sensitivity value will result in fewer support and resistance levels
                being identified. Defaults to 0.05.
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for calculating support and resistance levels.
                The number of periods (time intervals) over which to calculate the support and resistance levels.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                If None, the rounding value specified during the initialization of the Toolkit instance will be used.
                Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the levels.
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
           pd.Series or pd.DataFrame: The support and resistance levels for each asset.

        Raises:
            ValueError: If the specified `period` is not one of the valid options.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          support and resistance levels for each asset in the Toolkit instance.
        - A level is only identified on the handful of dates where a new local maximum or minimum
          is confirmed. The result is forward-filled so every date shows the most recently
          confirmed level (NaN before the first level is confirmed for that asset).
        - Levels are identified with a centred pivot window, which cannot confirm an extreme
          until `window` further periods have printed without exceeding it. Every level is
          therefore published with a confirmation lag of exactly `window` periods, and the
          series is append-only: a value read at any date is exactly the value that was
          available at that date, so the output is safe to use in a backtest.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_support_resistance_levels().xs("GCUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Resistance |   Support |
        |:-----------|-------------:|----------:|
        | 2025-12-26 |       4359.4 |    3960.5 |
        | 2025-12-28 |       4359.4 |    3960.5 |
        | 2025-12-29 |       4359.4 |    3960.5 |
        | 2025-12-30 |       4359.4 |    3960.5 |
        | 2025-12-31 |       4359.4 |    3960.5 |
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

        support_resistance_levels_df = calculate_per_ticker(
            historical_data,
            [close_column],
            overlap_model.get_support_resistance_levels,
            window=window,
            sensitivity=sensitivity,
        )

        return finalize_dataset(
            dataset=support_resistance_levels_df,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_portfolio
    @handle_errors
    def get_fibonacci_retracement_levels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        levels: list[float] | None = None,
        trend: str = "uptrend",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Fibonacci Retracement Levels for a given price series.

        Fibonacci Retracement Levels are horizontal price levels, derived from ratios found in
        the Fibonacci sequence, that traders watch as potential support (during a pullback
        within an uptrend) or resistance (during a bounce within a downtrend) zones. For every
        date, the swing high and swing low are taken as the rolling maximum high and rolling
        minimum low over the specified `window`, and the retracement levels are derived from
        that high/low pair.

        The formula is as follows:

        - Uptrend (retracing down from the high): Level = High — Ratio * (High — Low)
        - Downtrend (retracing up from the low): Level = Low + Ratio * (High — Low)

        Also known as: Fibonacci retracement, Fib levels, retracement levels.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods over which the rolling swing high
                (maximum) and swing low (minimum) are determined. Defaults to 14.
            levels (list[float] | None, optional): The Fibonacci ratios to calculate levels for.
                Defaults to the standard [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0].
            trend (str, optional): Whether to compute retracement levels for an "uptrend"
                (levels measured down from the high — the conventional direction, used when a
                prior move was up and price is now pulling back) or a "downtrend" (levels
                measured up from the low, used when a prior move was down and price is now
                bouncing). Defaults to "uptrend".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the retracement levels.
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
            pd.Series or pd.DataFrame: Fibonacci Retracement Levels, one column per ratio in
            `levels`, labelled by the ratio expressed as a percentage (e.g. "23.6%").

        Raises:
            ValueError: If the specified `period` is not one of the valid options, or if `trend`
                is not "uptrend" or "downtrend".

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Fibonacci Retracement Levels for each asset in the Toolkit instance.
        - The 50% level is not actually a Fibonacci ratio. It is included purely by long-standing
          market convention, based on the Dow Theory observation that markets often retrace
          about half of a prior move.
        - The 78.6% level is the square root of 0.618, not a ratio drawn directly from the
          Fibonacci sequence itself (unlike 23.6%, 38.2% and 61.8%, which are).
        - The columns sort lexicographically by their label (so "100.0%" sorts right after
          "0.0%", ahead of "23.6%"), matching the sorting convention used by every other
          multi-column indicator in this module (e.g. Pivot Points' Resistance/Support levels).
        - There is no single canonical academic paper behind Fibonacci Retracement Levels — the
          indicator is a practitioner tool derived from the Fibonacci sequence's ratios rather
          than a published financial model. The standard textbook treatment is Murphy, J.J.
          (1999). "Technical Analysis of the Financial Markets." New York Institute of Finance.
        - If `growth` is set to True, the method calculates the growth of the retracement levels
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_fibonacci_retracement_levels().xs("BTCUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |    0.0% |   100.0% |   23.6% |   38.2% |   50.0% |   61.8% |   78.6% |
        |:-----------|--------:|---------:|--------:|--------:|--------:|--------:|--------:|
        | 2025-12-27 | 90541.8 |    84400 | 89092.3 | 88195.6 | 87470.9 | 86746.2 | 85714.3 |
        | 2025-12-28 | 90541.8 |    84400 | 89092.3 | 88195.6 | 87470.9 | 86746.2 | 85714.3 |
        | 2025-12-29 | 90541.8 |    84400 | 89092.3 | 88195.6 | 87470.9 | 86746.2 | 85714.3 |
        | 2025-12-30 | 90541.8 |    84400 | 89092.3 | 88195.6 | 87470.9 | 86746.2 | 85714.3 |
        | 2025-12-31 | 90541.8 |    84400 | 89092.3 | 88195.6 | 87470.9 | 86746.2 | 85714.3 |
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

        fibonacci_retracement_levels = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            lambda high, low: overlap_model.get_fibonacci_retracement_levels(
                high.rolling(window=window).max(),
                low.rolling(window=window).min(),
                levels=levels,
                trend=trend,
            ),
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=fibonacci_retracement_levels,
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
