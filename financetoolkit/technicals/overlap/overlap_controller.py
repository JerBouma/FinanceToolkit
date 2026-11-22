"""Overlap Module"""

__docformat__ = "google"

from typing import Literal

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.technicals.helpers import calculate_per_ticker
from financetoolkit.technicals.overlap import overlap_model
from financetoolkit.technicals.volatility import volatility_model
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
    Overlap indicators drawn over the price: the moving averages (including the Kaufman adaptive moving average, Tillson T3, Arnaud Legoux moving average, volume weighted, elastic volume weighted and volume adjusted moving averages, McGinley Dynamic, variable index dynamic average, fractal adaptive moving average and the Holt-Winters moving average and channel), the price transforms and midpoints, TRIX, VWAP, the parabolic SAR, the Gann HiLo Activator, the Zig Zag, pivot points, support and resistance levels and Fibonacci retracement levels.
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

        | Date       |   Simple Moving Average (SMA) |   Exponential Moving Average (EMA) |   Double Exponential Moving Average (DEMA) |   TRIX |   Triangular Moving Average |   Weighted Moving Average (WMA) |   Hull Moving Average (HMA) |   Triple Exponential Moving Average (TEMA) |   Zero Lag Exponential Moving Average (ZLEMA) |   Wilder's Smoothed Moving Average (RMA) |   Sine Weighted Moving Average (SINWMA) |   Fibonacci Weighted Moving Average (FWMA) |   Pascal's Weighted Moving Average (PWMA) |   Kaufman Adaptive Moving Average (KAMA) |   Tillson T3 Moving Average (T3) |   Arnaud Legoux Moving Average (ALMA) |   Volume Weighted Moving Average (VWMA) |   McGinley Dynamic |   Variable Index Dynamic Average (VIDYA) |   Fractal Adaptive Moving Average (FRAMA) |   Holt-Winters Moving Average (HWMA) |   Holt-Winters Channel Upper Band |   Holt-Winters Channel Lower Band |   Elastic Volume Weighted Moving Average (EVWMA) |   Volume Adjusted Moving Average (VAMA) |   Midpoint |   Midpoint Price |   Average Price |   Median Price |   Typical Price |   Weighted Close Price |   Volume Weighted Average Price (VWAP) |   Parabolic SAR |   Gann HiLo Activator |   Zig Zag |   Pivot Point |   Pivot Point Resistance 1 |   Pivot Point Support 1 |   Fibonacci Retracement 50.0% |   Fibonacci Retracement 61.8% |
        |:-----------|------------------------------:|-----------------------------------:|-------------------------------------------:|-------:|----------------------------:|--------------------------------:|----------------------------:|-------------------------------------------:|----------------------------------------------:|-----------------------------------------:|----------------------------------------:|-------------------------------------------:|------------------------------------------:|-----------------------------------------:|---------------------------------:|--------------------------------------:|----------------------------------------:|-------------------:|-----------------------------------------:|------------------------------------------:|-------------------------------------:|----------------------------------:|----------------------------------:|-------------------------------------------------:|----------------------------------------:|-----------:|-----------------:|----------------:|---------------:|----------------:|-----------------------:|---------------------------------------:|----------------:|----------------------:|----------:|--------------:|---------------------------:|------------------------:|------------------------------:|------------------------------:|
        | 2025-12-17 |                       277.781 |                            275.314 |                                    275.767 | 0.0016 |                     277.982 |                         276.128 |                     273.116 |                                    273.419 |                                       274.067 |                                  273.378 |                                 278.063 |                                    273.419 |                                   277.589 |                                  274.218 |                          279.323 |                               274.905 |                                 277.896 |            271.978 |                                  270.616 |                                   275.481 |                              276.12  |                           280.057 |                           272.184 |                                          273.073 |                                 277.765 |    278.046 |          279.158 |         272.712 |        272.949 |         272.265 |                271.923 |                                277.974 |         285.493 |               280.207 |   270.697 |       273.015 |                    275.185 |                 271.488 |                       279.158 |                       277.161 |
        | 2025-12-18 |                       277.308 |                            274.771 |                                    274.694 | 0.0014 |                     277.238 |                         275.257 |                     271.901 |                                    272.199 |                                       272.815 |                                  273.226 |                                 277.267 |                                    272.587 |                                   277.094 |                                  274.081 |                          279.146 |                               273.813 |                                 277.325 |            271.925 |                                  270.663 |                                   273.495 |                              274.766 |                           278.713 |                           270.819 |                                          272.916 |                                 277.208 |    278.046 |          276.821 |         270.652 |        269.352 |         269.983 |                270.298 |                                277.321 |         284.309 |               279.456 |   266.023 |       272.265 |                    273.833 |                 269.328 |                       276.821 |                       274.272 |
        | 2025-12-19 |                       276.636 |                            274.498 |                                    274.194 | 0.0012 |                     276.433 |                         274.645 |                     271.147 |                                    271.835 |                                       272.223 |                                  273.19  |                                 276.385 |                                    272.635 |                                   276.649 |                                  273.957 |                          278.852 |                               273.059 |                                 276.053 |            271.981 |                                  270.755 |                                   272.783 |                              273.865 |                           277.958 |                           269.773 |                                          272.876 |                                 276.066 |    278.046 |          276.821 |         271.634 |        271.305 |         271.776 |                272.012 |                                276.002 |         282.481 |               278.475 |   266.023 |       269.983 |                    273.942 |                 267.286 |                       276.821 |                       274.272 |
        | 2025-12-22 |                       275.553 |                            273.902 |                                    273.122 | 0.0009 |                     275.578 |                         273.764 |                     270.399 |                                    270.666 |                                       270.96  |                                  272.964 |                                 275.489 |                                    271.634 |                                   275.978 |                                  273.151 |                          278.443 |                               272.238 |                                 275.011 |            271.838 |                                  270.713 |                                   272.504 |                              272.454 |                           276.493 |                           268.414 |                                          272.723 |                                 275.059 |    276.596 |          276.821 |         271.111 |        271.25  |         270.843 |                270.64  |                                275.063 |         280.835 |               277.345 |   266.023 |       271.776 |                    274.59  |                 269.906 |                       276.821 |                       274.272 |
        | 2025-12-23 |                       274.714 |                            273.57  |                                    272.607 | 0.0007 |                     274.764 |                         273.212 |                     269.985 |                                    270.319 |                                       270.788 |                                  272.853 |                                 274.716 |                                    271.545 |                                   275.004 |                                  272.908 |                          277.949 |                               271.772 |                                 274.32  |            271.808 |                                  270.755 |                                   272.185 |                              271.5   |                           275.349 |                           267.651 |                                          272.665 |                                 274.38  |    274.877 |          274.882 |         270.373 |        270.089 |         270.531 |                270.752 |                                274.256 |         279.354 |               276.408 |   266.023 |       270.843 |                    272.115 |                 268.757 |                       274.882 |                       272.792 |
        | 2025-12-24 |                       274.223 |                            273.476 |                                    272.559 | 0.0006 |                     274.049 |                         272.965 |                     270.161 |                                    270.616 |                                       270.958 |                                  272.854 |                                 274.088 |                                    272.044 |                                   273.884 |                                  272.904 |                          277.412 |                               271.704 |                                 273.909 |            271.882 |                                  270.834 |                                   272.353 |                              270.973 |                           274.704 |                           267.242 |                                          272.67  |                                 273.974 |    273.921 |          273.094 |         272.496 |        272.864 |         272.863 |                272.862 |                                273.798 |         278.021 |               275.97  |   266.023 |       270.531 |                    272.438 |                 269.508 |                       273.094 |                       271.425 |
        | 2025-12-26 |                       273.84  |                            273.339 |                                    272.426 | 0.0004 |                     273.429 |                         272.729 |                     270.664 |                                    270.745 |                                       271.364 |                                  272.825 |                                 273.546 |                                    272.197 |                                   272.872 |                                  272.88  |                          276.858 |                               271.784 |                                 273.559 |            271.922 |                                  270.903 |                                   272.475 |                              270.423 |                           273.963 |                           266.884 |                                          272.663 |                                 273.628 |    273.921 |          272.6   |         272.996 |        273.163 |         272.926 |                272.807 |                                273.418 |         276.821 |               275.64  |   266.023 |       272.863 |                    274.47  |                 271.252 |                       272.6   |                       271.048 |
        | 2025-12-29 |                       273.546 |                            273.268 |                                    272.416 | 0.0003 |                     272.918 |                         272.591 |                     271.326 |                                    271.012 |                                       271.765 |                                  272.824 |                                 273.082 |                                    272.429 |                                   272.145 |                                  272.877 |                          276.31  |                               272.011 |                                 273.315 |            271.984 |                                  270.911 |                                   272.462 |                              270.03  |                           273.441 |                           266.62  |                                          272.668 |                                 273.386 |    273.921 |          272.6   |         272.341 |        272.406 |         272.541 |                272.608 |                                273.16  |         275.741 |               275.206 |   266.023 |       272.926 |                    273.939 |                 271.438 |                       272.6   |                       271.048 |
        | 2025-12-30 |                       273.255 |                            273.117 |                                    272.247 | 0.0002 |                     272.531 |                         272.403 |                     271.798 |                                    271.015 |                                       271.736 |                                  272.774 |                                 272.681 |                                    272.315 |                                   271.738 |                                  272.849 |                          275.78  |                               272.173 |                                 273.113 |            271.995 |                                  270.932 |                                   272.442 |                              269.554 |                           272.853 |                           266.256 |                                          272.648 |                                 273.181 |    273.921 |          272.6   |         272.115 |        272.232 |         272.198 |                272.182 |                                272.911 |         274.769 |               274.771 |   266.023 |       272.541 |                    273.677 |                 271.674 |                       272.6   |                       271.048 |
        | 2025-12-31 |                       272.762 |                            272.823 |                                    271.815 | 0.0001 |                     272.2   |                         272.091 |                     271.883 |                                    270.627 |                                       271.745 |                                  272.642 |                                 272.319 |                                    271.777 |                                   271.608 |                                  272.686 |                          275.262 |                               272.102 |                                 272.746 |            271.917 |                                  270.932 |                                   272.383 |                              268.893 |                           272.144 |                           265.643 |                                          272.568 |                                 272.8   |    273.672 |          272.6   |         271.641 |        271.768 |         271.484 |                271.342 |                                272.594 |         273.895 |               274.318 |   266.023 |       272.198 |                    273.062 |                 271.268 |                       272.6   |                       271.048 |
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

        overlap_indicators["Triple Exponential Moving Average (TEMA)"] = (
            self.get_moving_average(
                period=period,
                close_column=close_column,
                window=window,
                method="triple_exponential",
            )
        )

        overlap_indicators["Zero Lag Exponential Moving Average (ZLEMA)"] = (
            self.get_moving_average(
                period=period,
                close_column=close_column,
                window=window,
                method="zero_lag",
            )
        )

        overlap_indicators["Wilder's Smoothed Moving Average (RMA)"] = (
            self.get_moving_average(
                period=period,
                close_column=close_column,
                window=window,
                method="wilder",
            )
        )

        overlap_indicators["Sine Weighted Moving Average (SINWMA)"] = (
            self.get_moving_average(
                period=period,
                close_column=close_column,
                window=window,
                method="sine",
            )
        )

        overlap_indicators["Fibonacci Weighted Moving Average (FWMA)"] = (
            self.get_moving_average(
                period=period,
                close_column=close_column,
                window=window,
                method="fibonacci",
            )
        )

        overlap_indicators["Pascal's Weighted Moving Average (PWMA)"] = (
            self.get_moving_average(
                period=period,
                close_column=close_column,
                window=window,
                method="pascal",
            )
        )

        overlap_indicators["Kaufman Adaptive Moving Average (KAMA)"] = (
            self.get_kaufman_adaptive_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Tillson T3 Moving Average (T3)"] = (
            self.get_t3_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Arnaud Legoux Moving Average (ALMA)"] = (
            self.get_arnaud_legoux_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Volume Weighted Moving Average (VWMA)"] = (
            self.get_volume_weighted_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["McGinley Dynamic"] = self.get_mcginley_dynamic(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators["Variable Index Dynamic Average (VIDYA)"] = (
            self.get_variable_index_dynamic_average(
                period=period, close_column=close_column, window=window
            )
        )

        # FRAMA splits its window in two halves, so an odd window is rounded down to an even one.
        overlap_indicators["Fractal Adaptive Moving Average (FRAMA)"] = (
            self.get_fractal_adaptive_moving_average(
                period=period, close_column=close_column, window=max(window // 2 * 2, 2)
            )
        )

        overlap_indicators["Holt-Winters Moving Average (HWMA)"] = (
            self.get_holt_winters_moving_average(
                period=period, close_column=close_column
            )
        )

        holt_winters_channel = self.get_holt_winters_channel(
            period=period, close_column=close_column
        )

        overlap_indicators["Holt-Winters Channel Upper Band"] = holt_winters_channel[
            "Upper Band"
        ]
        overlap_indicators["Holt-Winters Channel Lower Band"] = holt_winters_channel[
            "Lower Band"
        ]

        overlap_indicators["Elastic Volume Weighted Moving Average (EVWMA)"] = (
            self.get_elastic_volume_weighted_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Volume Adjusted Moving Average (VAMA)"] = (
            self.get_volume_adjusted_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Midpoint"] = self.get_midpoint(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators["Midpoint Price"] = self.get_midpoint(
            period=period, close_column=close_column, window=window, method="high_low"
        )

        for method, name in [
            ("average", "Average Price"),
            ("median", "Median Price"),
            ("typical", "Typical Price"),
            ("weighted_close", "Weighted Close Price"),
        ]:
            overlap_indicators[name] = self.get_price_transform(
                period=period, close_column=close_column, method=method
            )

        overlap_indicators["Volume Weighted Average Price (VWAP)"] = (
            self.get_volume_weighted_average_price(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Parabolic SAR"] = self.get_parabolic_sar(
            period=period, close_column=close_column
        )

        overlap_indicators["Gann HiLo Activator"] = self.get_gann_hilo_activator(
            period=period, close_column=close_column
        )["Gann HiLo Activator"]

        overlap_indicators["Zig Zag"] = self.get_zigzag(
            period=period, close_column=close_column
        )["Zig Zag"]

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
        double exponential, triple exponential, zero lag exponential, Wilder, triangular,
        weighted, Hull, sine weighted, Fibonacci weighted or Pascal's weighted moving average.

        A Moving Average smooths out price data by averaging the price over a specified
        number of periods. The variants differ in how they weight the prices within the
        window: equally (simple), with exponentially decaying weights (exponential, double and
        triple exponential, zero lag exponential and Wilder), with weights that peak in the
        middle of the window (triangular, sine and Pascal's weighted), with linearly increasing
        weights (weighted), with weights that grow with the Fibonacci numbers (Fibonacci
        weighted) or with a combination of weighted averages that reduces the lag (Hull).

        The formula is as follows, per method:

        - "simple": SMA = (Sum of Prices) / (Number of Prices)
        - "exponential": the first EMA is SMA(Close, Window), on the Window-th period, and
          EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA
        - "double_exponential": DEMA = 2 * EMA(Close, Window) — EMA(EMA(Close, Window), Window)
        - "triple_exponential": TEMA = 3 * EMA1 - 3 * EMA2 + EMA3, with EMA1 = EMA(Close, Window),
          EMA2 = EMA(EMA1, Window) and EMA3 = EMA(EMA2, Window)
        - "zero_lag": ZLEMA = EMA(2 * Close - Close(t - Lag), Window), with Lag = (Window - 1) / 2
          rounded down
        - "wilder": the first value is SMA(Close, Window), on the Window-th period, and
          RMA = Previous RMA + (Close - Previous RMA) / Window
        - "triangular": TMA = SMA(SMA(Close, Sub-window Length 1), Sub-window Length 2), with
          both sub-windows (Window + 1) / 2 for an odd window and Window / 2 and Window / 2 + 1
          for an even window (TA-Lib's TRIMA convention)
        - "weighted": WMA = (Sum of (Price * Weight)) / (Sum of Weights), with weights 1 to
          Window from the oldest to the latest price
        - "hull": HMA = WMA(2 * WMA(Close, Window / 2) — WMA(Close, Window), sqrt(Window))
        - "sine": SINWMA = (Sum of (Price * Weight)) / (Sum of Weights), with weights
          sin(i * pi / (Window + 1)) for i = 1 to Window
        - "fibonacci": FWMA = (Sum of (Price * Weight)) / (Sum of Weights), with the Fibonacci
          numbers 1, 1, 2, 3, 5, ... from the oldest to the latest price
        - "pascal": PWMA = (Sum of (Price * Weight)) / (Sum of Weights), with the binomial
          coefficients of row Window - 1 of Pascal's triangle

        Seeding the EMA with a simple average, as TA-Lib and StockCharts do, leaves the first
        Window - 1 periods of the price history empty. The DEMA, TRIX, MACD, Percentage Price
        Oscillator, Force Index, Elder Ray Index, Keltner Channels and Chaikin and McClellan
        Oscillators use the same EMA and start once their own EMAs are seeded; TEMA starts at
        period 3 * (Window - 1) and ZLEMA at period Lag + Window - 1. Wilder's average is the
        smoothing of the RSI, ATR and ADX, an EMA with a smoothing factor of 1 / Window instead
        of 2 / (Window + 1). The triangular moving average is also the symmetric weighted moving
        average, with weights 1, 2, ..., 2, 1. The averages with further parameters are separate
        methods: the Kaufman Adaptive Moving Average (`get_kaufman_adaptive_moving_average`), the
        Tillson T3 (`get_t3_moving_average`), the Arnaud Legoux Moving Average
        (`get_arnaud_legoux_moving_average`), the Volume Weighted Moving Average
        (`get_volume_weighted_moving_average`), the McGinley Dynamic (`get_mcginley_dynamic`),
        the Variable Index Dynamic Average (`get_variable_index_dynamic_average`) and the
        Fractal Adaptive Moving Average (`get_fractal_adaptive_moving_average`).

        For more information, see the following sources:

        - Mulloy, P.G. (1994). "Smoothing Data with Faster Moving Averages." Technical Analysis
          of Stocks & Commodities, 12(1).
        - Mulloy, P.G. (1994). "Smoothing Data with Less Lag." Technical Analysis of Stocks &
          Commodities, 12(2).
        - Ehlers, J.F. & Way, R. (2010). "Zero Lag (Well, Almost)." Technical Analysis of Stocks &
          Commodities, November 2010.
        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - Hull, A. (2005). "Hull Moving Average."
        - Everget (2018). "Sine Weighted Moving Average." TradingView script.
        - TA-Lib: SMA, EMA, DEMA, TEMA, ZLEMA, RMA, TRIMA, WMA and HMA, the reference
          implementations (https://ta-lib.org/functions/).
        - Referenced against pandas-ta-classic's swma, sinwma, fwma and pwma of the
          symmetric, sine, Fibonacci and Pascal's weighted averages
          (https://github.com/xgboosted/pandas-ta-classic). The Fibonacci and Pascal's weighted averages
          are defined in pandas-ta by Kevin Johnson.

        Also known as: MA, SMA, simple moving average, EMA, exponential moving average, DEMA, double exponential moving average, TEMA, triple exponential moving average, ZLEMA, ZLMA, zero lag moving average, RMA, SMMA, Wilder's smoothing, smoothed moving average, running moving average, TMA, TRIMA, triangular moving average, SWMA, symmetric weighted moving average, WMA, linearly weighted moving average, HMA, Hull moving average, SINWMA, sine weighted moving average, FWMA, Fibonacci weighted moving average, PWMA, Pascal's weighted moving average.

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
                "double_exponential", "triple_exponential", "zero_lag", "wilder", "triangular",
                "weighted", "hull", "sine", "fibonacci" or "pascal". Defaults to "simple".
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
        - The simple, exponential, double exponential, triple exponential, zero lag exponential,
          Wilder, triangular, weighted and Hull moving averages agree with TA-Lib's SMA, EMA, DEMA, TEMA,
          ZLEMA, RMA, TRIMA, WMA and HMA, and the triangular, sine, Fibonacci and Pascal's weighted
          averages agree with pandas-ta-classic's swma, sinwma, fwma and pwma.
        - finta's TEMA, SMMA and SSMA start their exponential averages on the first price (with
          pandas' ewm) instead of seeding them with a simple average over the window, so they
          differ during the warm-up and converge afterwards.
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
        moving_averages = volatility_model.MOVING_AVERAGES
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
        signal_window: int | None = None,
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
        in a market. With a `signal_window` it adds the signal line, an EMA of the TRIX whose
        crossings Hutson read as signals, and the histogram, the TRIX minus its signal line.

        The formula is as follows:

        - EMA1 = EMA(Close, Window)
        - EMA2 = EMA(EMA1, Window)
        - EMA3 = EMA(EMA2, Window)
        - TRIX = (EMA3 — EMA3[—1]) / EMA3[—1]
        - Signal Line = EMA(TRIX, Signal Window)
        - Histogram = TRIX — Signal Line

        Also known as: triple smoothed EMA, rate of change oscillator, TRIX signal, TRIXH, TRIX histogram.

        For more information, see the following sources:

        - Hutson, J.K. (1983). "Good TRIX." Technical Analysis of Stocks & Commodities, 1(5), 105-108.
        - Referenced against TA-Lib's TRIX (https://ta-lib.org/functions/trix.html).
        - pandas-ta-classic: trix and trixh, with the signal line and histogram
          (https://github.com/xgboosted/pandas-ta-classic).

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of each of the three exponential moving
                averages. Defaults to 15, the window of Hutson (1983) and StockCharts.
            signal_window (int | None, optional): The number of periods of the signal line, e.g. 9.
                Defaults to None, which returns only the TRIX.
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
            Trix values as a decimal (0.001 is 0.1%), or with a `signal_window` the TRIX, Signal Line
            and Histogram.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Trix for each asset in the Toolkit instance.
        - TRIX is a decimal and agrees with TA-Lib's TRIX divided by 100, since TA-Lib's TRIX is in
          percent.
        - The signal line is an EMA seeded with the simple average of its first values, as every
          EMA of the Toolkit, and agrees with TA-Lib's EMA of the TRIX. pandas-ta-classic's trixh takes
          a simple moving average of the TRIX as its signal line instead.
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
            historical_data,
            [close_column],
            overlap_model.get_trix,
            window,
            signal_window,
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
        - Referenced against TA-Lib's KAMA (https://ta-lib.org/functions/kama.html).
        - With the default `fast_window` of 2 and `slow_window` of 30, which TA-Lib fixes, the KAMA
          agrees with TA-Lib's KAMA.
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
    def get_t3_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 5,
        volume_factor: float = 0.7,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Tillson T3 Moving Average for a given price series.

        The T3 applies Tillson's Generalized DEMA, GD(x) = (1 + a) * EMA(x) - a * EMA(EMA(x)),
        three times in a row. The volume factor a sets how much of DEMA's lag correction is
        used: 0 gives an EMA of an EMA of an EMA and 1 three DEMAs in a row. The result is a
        smooth average with little lag and little overshoot.

        The formula is as follows:

        - E1 = EMA(Close, Window), E2 = EMA(E1, Window), ..., E6 = EMA(E5, Window)
        - c1 = -a^3, c2 = 3a^2 + 3a^3, c3 = -6a^2 - 3a - 3a^3, c4 = 1 + 3a + a^3 + 3a^2
        - T3 = c1 * E6 + c2 * E5 + c3 * E4 + c4 * E3

        For more information, see the following sources:

        - Tillson, T. (1998). "Smoothing Techniques for More Accurate Signals." Technical Analysis
          of Stocks & Commodities, 16(1).
        - Referenced against TA-Lib's T3 (https://ta-lib.org/functions/t3.html).
        - pandas-ta-classic: t3 (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: T3, Tillson T3, T3 moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of each of the six exponential moving
                averages. Defaults to 5, TA-Lib's default.
            volume_factor (float, optional): The volume factor a, between 0 and 1. Defaults to 0.7,
                Tillson's value.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the T3.
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
            pd.DataFrame or pd.Series: Tillson T3 Moving Average values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the T3 for each asset in the Toolkit instance.
        - Each of the six EMAs is seeded with the simple average of its first `window` values,
          so the T3 starts at period 6 * (window - 1). It agrees with TA-Lib's T3 and
          pandas-ta-classic's t3.
        - If `growth` is set to True, the method calculates the growth of the T3 using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^DJI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_t3_moving_average().tail(5)
        ```

        Which returns:

        | Date       |    ^DJI |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 48312.7 |     676.353 |
        | 2025-12-26 | 48425.2 |     679.107 |
        | 2025-12-29 | 48504.8 |     681.236 |
        | 2025-12-30 | 48534.1 |     682.521 |
        | 2025-12-31 | 48490.9 |     682.528 |
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

        t3_moving_average = calculate_per_ticker(
            historical_data,
            [close_column],
            overlap_model.get_t3_moving_average,
            window,
            volume_factor,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=t3_moving_average,
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
    def get_arnaud_legoux_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 9,
        sigma: float = 6.0,
        offset: float = 0.85,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Arnaud Legoux Moving Average (ALMA) for a given price series.

        ALMA weights the prices in the window with a Gaussian curve whose peak sits at `offset`
        of the way from the oldest to the latest price. An offset close to 1 puts the weight on
        recent prices (less lag) and one close to 0 on older prices (more smoothing), while
        `sigma` sets how narrow the curve is.

        The formula is as follows:

        - m = floor(Offset * (Window - 1)), s = Window / Sigma
        - Weight(i) = exp(-(i - m)^2 / (2 * s^2)), for i = 0 (the oldest price) to Window - 1
        - ALMA = Sum(Weight(i) * Price(i)) / Sum(Weight(i))

        For more information, see the following sources:

        - Legoux, A. & Kouzis-Loukas, D. (2009). "ALMA: Arnaud Legoux Moving Average."
          arnaudlegoux.com.
        - Referenced against TA-Lib's ALMA (https://ta-lib.org/functions/alma.html).
        - pandas-ta-classic: alma (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: ALMA, Arnaud Legoux moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods to consider. Defaults to 9.
            sigma (float, optional): The width of the Gaussian curve, as the number of standard
                deviations that fit in the window. Defaults to 6.0.
            offset (float, optional): The position of the peak of the curve in the window, from 0
                (the oldest price) to 1 (the latest price). Defaults to 0.85.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ALMA.
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
            pd.DataFrame or pd.Series: Arnaud Legoux Moving Average (ALMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the ALMA for each asset in the Toolkit instance.
        - The peak m is rounded down to a whole period, and the ALMA agrees with TA-Lib's ALMA.
          pandas-ta-classic's alma measures the offset from the latest price and keeps m
          unrounded: its alma(offset=0.85) equals this ALMA with an offset of 0.15 and an
          unrounded peak, so it puts the weight on the older prices.
        - If `growth` is set to True, the method calculates the growth of the ALMA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_arnaud_legoux_moving_average().tail(5)
        ```

        Which returns:

        | Date       |    NVDA |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 182.824 |     678.086 |
        | 2025-12-26 | 185.894 |     681.239 |
        | 2025-12-29 | 187.703 |     683.046 |
        | 2025-12-30 | 188.3   |     683.49  |
        | 2025-12-31 | 187.983 |     682.463 |
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

        arnaud_legoux_moving_average = calculate_per_ticker(
            historical_data,
            [close_column],
            overlap_model.get_arnaud_legoux_moving_average,
            window,
            sigma,
            offset,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=arnaud_legoux_moving_average,
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
    def get_volume_weighted_moving_average(
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
        Calculate the Volume Weighted Moving Average (VWMA) for a given price series.

        The Volume Weighted Moving Average weights every closing price in the window by the
        volume traded in its period, so prices at which much was traded count more than those
        of quiet periods. The VWMA rises above the simple moving average when the volume is
        heavier on the up days and falls below it when the volume is heavier on the down days.

        The formula is as follows:

        - VWMA = Sum(Close * Volume, Window) / Sum(Volume, Window)

        For more information, see the following sources:

        - Dormeier, B.P. (2011). "Investing with Volume Analysis." FT Press.
        - Referenced against TA-Lib's VWMA (https://ta-lib.org/functions/vwma.html).
        - pandas-ta-classic: vwma (https://github.com/xgboosted/pandas-ta-classic).
        - Tulip Indicators: vwma (https://tulipindicators.org/vwma).

        Also known as: VWMA, volume weighted moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods to consider. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the VWMA.
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
            pd.DataFrame or pd.Series: Volume Weighted Moving Average (VWMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the VWMA for each asset in the Toolkit instance.
        - The VWMA differs from the rolling VWAP (`get_volume_weighted_average_price`) only in
          the price it averages: the close instead of the typical price (High + Low + Close) / 3.
          It agrees with TA-Lib's VWMA and pandas-ta-classic's vwma.
        - A window without any volume has no average (NaN).
        - If `growth` is set to True, the method calculates the growth of the VWMA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.technicals.get_volume_weighted_moving_average().tail(5)
        ```

        Which returns:

        | Date       |     QQQ |
        |:-----------|--------:|
        | 2025-12-24 | 614.136 |
        | 2025-12-26 | 613.814 |
        | 2025-12-29 | 613.577 |
        | 2025-12-30 | 613.29  |
        | 2025-12-31 | 612.312 |
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

        volume_weighted_moving_average = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            overlap_model.get_volume_weighted_moving_average,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=volume_weighted_moving_average,
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
    def get_mcginley_dynamic(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        constant: float = 1.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the McGinley Dynamic for a given price series.

        The McGinley Dynamic is a moving average that adjusts its own speed to the market: the
        step towards the price is divided by (Price / MD)^4, so the average moves slowly while
        the price runs above it and quickly when the price falls below it, following the price
        more closely than an average with a fixed speed and with fewer whipsaws.

        The formula is as follows:

        - First MD = SMA(Close, Window), on the Window-th period
        - MD(t) = MD(t-1) + (Close(t) - MD(t-1)) / (Constant * Window * (Close(t) / MD(t-1))^4)

        For more information, see the following sources:

        - McGinley, J.R. (1997). "The McGinley Dynamic." Journal of Technical Analysis (Market
          Technicians Association).
        - Referenced against TA-Lib's MCGD (https://ta-lib.org/functions/mcgd.html).
        - pandas-ta-classic: mcgd (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: MCGD, McGinley Dynamic Indicator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods, the N of the formula. Defaults to 14,
                TA-Lib's default.
            constant (float, optional): The constant k that multiplies the window. Defaults to 1.0,
                as in TA-Lib and pandas-ta-classic; McGinley suggested 0.6 to follow the price more
                closely.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the McGinley Dynamic.
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
            pd.DataFrame or pd.Series: McGinley Dynamic values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the McGinley Dynamic for each asset in the Toolkit instance.
        - The recursion is seeded with the simple average of the first `window` prices, as the
          Toolkit seeds its exponential moving averages. TA-Lib's MCGD and pandas-ta-classic's
          mcgd start the recursion from the first price instead. The difference fades by about a
          factor 1 - 1 / window per period: with the default window of 14 the two agree to 0.01%
          within about 70 periods.
        - If `growth` is set to True, the method calculates the growth of the McGinley Dynamic using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^FTSE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_mcginley_dynamic().tail(5)
        ```

        Which returns:

        | Date       |   ^FTSE |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 9725.8  |     672.687 |
        | 2025-12-26 |  nan    |     673.502 |
        | 2025-12-29 | 9735.29 |     674.113 |
        | 2025-12-30 | 9748.78 |     674.63  |
        | 2025-12-31 | 9760.89 |     674.772 |
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

        mcginley_dynamic = calculate_per_ticker(
            historical_data,
            [close_column],
            overlap_model.get_mcginley_dynamic,
            window,
            constant,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=mcginley_dynamic,
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
    def get_variable_index_dynamic_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 12,
        cmo_window: int = 9,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Variable Index Dynamic Average (VIDYA) for a given price series.

        VIDYA is an exponential moving average whose smoothing factor is scaled by the strength
        of the trend, measured by the absolute Chande Momentum Oscillator: in a strong trend
        (|CMO| near 1) it moves like an EMA of `window` periods, and in a market that goes
        nowhere (|CMO| near 0) it hardly moves at all.

        The formula is as follows:

        - Gains = Sum(max(Close(t) - Close(t-1), 0), CMO Window)
        - Losses = Sum(max(Close(t-1) - Close(t), 0), CMO Window)
        - |CMO| = |Gains - Losses| / (Gains + Losses), and 0 when prices did not move
        - Alpha = 2 / (Window + 1)
        - VIDYA(t) = Alpha * |CMO(t)| * Close(t) + (1 - Alpha * |CMO(t)|) * VIDYA(t-1)

        For more information, see the following sources:

        - Chande, T.S. (1992). "Adapting Moving Averages to Market Volatility." Technical Analysis
          of Stocks & Commodities, 10(3).
        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.
        - Referenced against TA-Lib's VIDYA (https://ta-lib.org/functions/vidya.html).
        - pandas-ta-classic: vidya (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: VIDYA, variable index dynamic average, Chande's dynamic average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the exponential moving average.
                Defaults to 12, TA-Lib's default.
            cmo_window (int, optional): The number of periods of the Chande Momentum Oscillator.
                Defaults to 9, TA-Lib's default.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the VIDYA.
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
            pd.DataFrame or pd.Series: Variable Index Dynamic Average (VIDYA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the VIDYA for each asset in the Toolkit instance.
        - The volatility index is the absolute Chande Momentum Oscillator over unsmoothed sums
          (Chande & Kroll, 1994), as in TA-Lib and pandas-ta-classic. Chande's 1992 article and
          Tulip Indicators' vidya use the ratio of a short to a long standard deviation instead.
        - The recursion is seeded with the simple average of `window` prices on the period before
          the first oscillator value that follows a full window, as the Toolkit seeds its
          exponential moving averages. With equal windows it agrees with pandas-ta-classic's vidya.
          TA-Lib starts the recursion from the first price, with the oscillator over the changes
          available so far, and shows it from period `cmo_window` on. The difference fades by a
          factor 1 - Alpha * |CMO| per period: with the defaults the two agree to 0.01% within
          about 100 periods.
        - If `growth` is set to True, the method calculates the growth of the VIDYA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_variable_index_dynamic_average().tail(5)
        ```

        Which returns:

        | Date       |     GLD |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 397.569 |     670.332 |
        | 2025-12-26 | 399.971 |     671.09  |
        | 2025-12-29 | 399.957 |     671.58  |
        | 2025-12-30 | 399.946 |     672.076 |
        | 2025-12-31 | 399.908 |     672.374 |
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

        variable_index_dynamic_average = calculate_per_ticker(
            historical_data,
            [close_column],
            overlap_model.get_variable_index_dynamic_average,
            window,
            cmo_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=variable_index_dynamic_average,
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
    def get_fractal_adaptive_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 16,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Fractal Adaptive Moving Average (FRAMA) for a given price series.

        FRAMA is an exponential moving average of the median price whose smoothing factor
        follows the fractal dimension of the prices over the window. The dimension compares the
        ranges of the two halves of the window with the range of the whole: in a trend the
        ranges of the halves add up to the range of the whole (a dimension of 1) and FRAMA
        follows the price closely, while in a range-bound market each half covers most of the
        range (a dimension towards 2) and FRAMA barely moves.

        The formula is as follows:

        - Price = (High + Low) / 2
        - N1 = (Highest High - Lowest Low) / (Window / 2) over the latest Window / 2 periods
        - N2 = (Highest High - Lowest Low) / (Window / 2) over the Window / 2 periods before
        - N3 = (Highest High - Lowest Low) / Window over the whole window
        - D = (log(N1 + N2) - log(N3)) / log(2)
        - Alpha = exp(-4.6 * (D - 1)), limited to between 0.01 and 1
        - FRAMA(t) = Alpha(t) * Price(t) + (1 - Alpha(t)) * FRAMA(t-1)

        For more information, see the following sources:

        - Ehlers, J.F. (2005). "Fractal Adaptive Moving Averages." Technical Analysis of Stocks & Commodities,
          23(10), 81-82.
        - Referenced against TA-Lib's FRAMA (https://ta-lib.org/functions/frama.html).
        - finta: FRAMA (https://github.com/peerchemist/finta).

        Also known as: FRAMA, fractal adaptive moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods over which the fractal dimension is
                measured, an even number. Defaults to 16, Ehlers' window.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the FRAMA.
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
            pd.DataFrame or pd.Series: Fractal Adaptive Moving Average (FRAMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the FRAMA for each asset in the Toolkit instance.
        - With the default `close_column` of "Adj Close" the high and low are adjusted for
          dividends in the same way.
        - The recursion is seeded with the simple average of the first `window` median prices,
          on the window-th period, as the Toolkit seeds its exponential moving averages. Ehlers
          and TA-Lib start from the median price of that period, which TA-Lib does not show, so
          TA-Lib's FRAMA starts one period later. The difference fades by a factor 1 - Alpha per
          period: with the default window of 16 the two agree to 0.01% within about 45 periods.
        - A half without a range still gives a dimension, as in TA-Lib: a flat latest half next
          to a moving earlier one is a dimension of 1, so FRAMA moves to the price. Without any
          range in the window, or in both halves, the last dimension is kept.
        - If `growth` is set to True, the method calculates the growth of the FRAMA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_fractal_adaptive_moving_average().tail(5)
        ```

        Which returns:

        | Date       |    MSFT |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 481.61  |     676.384 |
        | 2025-12-26 | 481.639 |     677.132 |
        | 2025-12-29 | 481.659 |     677.331 |
        | 2025-12-30 | 481.764 |     677.517 |
        | 2025-12-31 | 481.759 |     677.645 |
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

        fractal_adaptive_moving_average = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            overlap_model.get_fractal_adaptive_moving_average,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=fractal_adaptive_moving_average,
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
    def get_price_transform(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        method: Literal["average", "median", "typical", "weighted_close"] = "typical",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate a Price Transform: the average, median, typical or weighted close price of each period.

        A price transform condenses the open, high, low and close of a period into one price.
        Indicators that should reflect the whole bar rather than only the close are built on
        them: the Commodity Channel Index, Money Flow Index and VWAP on the typical price, the
        Awesome Oscillator and FRAMA on the median price. The weighted close counts the close
        twice, as the price the period settled at.

        The formula is as follows, per method:

        - "average": Average Price = (Open + High + Low + Close) / 4
        - "median": Median Price = (High + Low) / 2
        - "typical": Typical Price = (High + Low + Close) / 3
        - "weighted_close": Weighted Close Price = (High + Low + 2 * Close) / 4

        For more information, see the following sources:

        - Lambert, D.R. (1980). "Commodity Channel Index: Tool for Trading Cyclic Trends."
          Commodities, October 1980, the typical price of the CCI.
        - Referenced against TA-Lib's AVGPRICE, MEDPRICE, TYPPRICE and WCLPRICE
          (https://ta-lib.org/functions/).
        - pandas-ta-classic: ohlc4, hl2, hlc3 and wcp (https://github.com/xgboosted/pandas-ta-classic).
        - finta: TP (https://github.com/peerchemist/finta).
        - Tulip Indicators: avgprice, medprice, typprice and wcprice (https://tulipindicators.org).

        Also known as: AVGPRICE, average price, OHLC4, MEDPRICE, median price, HL2, TYPPRICE, typical price,
        HLC3, pivot price, WCLPRICE, weighted close price, HLCC4.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            method (str, optional): The price: "average", "median", "typical" or "weighted_close".
                Defaults to "typical".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the price.
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
            pd.DataFrame or pd.Series: The price of each period.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the price for each asset in the Toolkit instance.
        - The four prices agree with TA-Lib's AVGPRICE, MEDPRICE, TYPPRICE and WCLPRICE.
        - If `growth` is set to True, the method calculates the growth of the price using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_price_transform(method="weighted_close").tail(5)
        ```

        Which returns:

        | Date       |    NVDA |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 187.741 |     684.502 |
        | 2025-12-26 | 189.993 |     685.038 |
        | 2025-12-29 | 187.339 |     682.413 |
        | 2025-12-30 | 187.312 |     681.964 |
        | 2025-12-31 | 187.075 |     677.933 |
        """
        if method not in ("average", "median", "typical", "weighted_close"):
            raise ValueError(
                f"method must be average, median, typical or weighted_close, not {method}."
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

        historical_data = self._get_historical_data(period, close_column)

        price_transform = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            overlap_model.get_price_transform,
            method,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=price_transform,
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
    def get_midpoint(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        method: Literal["close", "high_low"] = "close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Midpoint: the middle of the highest and lowest price over a window.

        The midpoint is the centre of the range the price covered over the window, a moving
        average that only looks at the extremes. With the closes it is the midpoint of the
        highest and lowest close; with the highs and lows it is the midpoint price, the middle
        line of a Donchian Channel that includes the current period.

        The formula is as follows, per method:

        - "close": Midpoint = (Highest Close(Window) + Lowest Close(Window)) / 2
        - "high_low": Midpoint Price = (Highest High(Window) + Lowest Low(Window)) / 2

        For more information, see the following sources:

        - Donchian, R.D. (1960). "Commodities: High Finance in Copper." Financial Analysts Journal, 16(6), 133-142.
        - Referenced against TA-Lib's MIDPOINT and MIDPRICE
          (https://ta-lib.org/functions/midpoint.html, https://ta-lib.org/functions/midprice.html).
        - pandas-ta-classic: midpoint and midprice (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: MIDPOINT, midpoint over period, MIDPRICE, midpoint price, Donchian middle line.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted for
                dividends in the same way.
            window (int, optional): The number of periods, including the current one. Defaults to 14.
            method (str, optional): "close" for the midpoint of the closes (MIDPOINT) or "high_low"
                for that of the highs and lows (MIDPRICE). Defaults to "close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the midpoint.
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
            pd.DataFrame or pd.Series: Midpoint values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the midpoint for each asset in the Toolkit instance.
        - The midpoint agrees with TA-Lib's MIDPOINT and the midpoint price its MIDPRICE. The middle
          line of `get_donchian_channels` ends one period earlier, so it differs whenever the
          current period sets a new high or low.
        - If `growth` is set to True, the method calculates the growth of the midpoint using
          the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_midpoint(method="high_low").tail(5)
        ```

        Which returns:

        | Date       |   AMZN |   Benchmark |
        |:-----------|-------:|------------:|
        | 2025-12-24 | 226.97 |     674.756 |
        | 2025-12-26 | 226.99 |     675.168 |
        | 2025-12-29 | 226.99 |     675.168 |
        | 2025-12-30 | 226.99 |     675.168 |
        | 2025-12-31 | 226.99 |     675.168 |
        """
        if method not in ("close", "high_low"):
            raise ValueError(f"method must be close or high_low, not {method}.")
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
        columns = (
            ["High", "Low"] if method == "high_low" else [close_column, close_column]
        )

        midpoint = calculate_per_ticker(
            historical_data, columns, overlap_model.get_midpoint, window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=midpoint,
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
    def get_holt_winters_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        level_smoothing: float = 0.2,
        velocity_smoothing: float = 0.1,
        acceleration_smoothing: float = 0.1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Holt-Winters Moving Average (HWMA) for a given price series.

        The HWMA applies Holt's exponential smoothing with a trend, extended with an
        acceleration term, to the price: it smooths the level of the price, its velocity (the
        change of the level) and its acceleration (the change of the velocity), and adds the
        expected velocity and acceleration of one period to the level. It follows a trend
        with less lag than an exponential moving average of similar smoothness, at the cost of
        overshooting when the trend turns.

        The formula is as follows:

        - F(t) = (1 - Level Smoothing) * (F(t-1) + V(t-1) + 0.5 * A(t-1)) + Level Smoothing * Close(t)
        - V(t) = (1 - Velocity Smoothing) * (V(t-1) + A(t-1)) + Velocity Smoothing * (F(t) - F(t-1))
        - A(t) = (1 - Acceleration Smoothing) * A(t-1) + Acceleration Smoothing * (V(t) - V(t-1))
        - HWMA(t) = F(t) + V(t) + 0.5 * A(t)

        with F starting at the first close and V and A at 0.

        For more information, see the following sources:

        - Holt, C.C. (1957). "Forecasting Seasonals and Trends by Exponentially Weighted Moving
          Averages." ONR Memorandum 52, Carnegie Institute of Technology (reprinted in the
          International Journal of Forecasting, 20(1), 2004).
        - Winters, P.R. (1960). "Forecasting Sales by Exponentially Weighted Moving Averages."
          Management Science, 6(3), 324-342.
        - MQL5 Code Base: HWMA (https://www.mql5.com/en/code/20856).
        - Referenced against pandas-ta-classic's hwma (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: HWMA, Holt-Winters moving average, Holt-Winter moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            level_smoothing (float, optional): The smoothing factor of the level, between 0 and 1.
                Defaults to 0.2.
            velocity_smoothing (float, optional): The smoothing factor of the velocity, between 0
                and 1. Defaults to 0.1.
            acceleration_smoothing (float, optional): The smoothing factor of the acceleration,
                between 0 and 1. Defaults to 0.1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the HWMA.
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
            pd.DataFrame or pd.Series: Holt-Winters Moving Average (HWMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the HWMA for each asset in the Toolkit instance.
        - The HWMA agrees with pandas-ta-classic's hwma. The recursion starts on each ticker's
          first close, as in the MQL5 indicator, so there is no warm-up: the weight of the
          start fades by a factor 1 - Level Smoothing per period, and the first values of the
          price history lean on that close.
        - If `growth` is set to True, the method calculates the growth of the HWMA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["META"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_holt_winters_moving_average().tail(5)
        ```

        Which returns:

        | Date       |    META |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 692.018 |     682.098 |
        | 2025-12-26 | 693.203 |     683.972 |
        | 2025-12-29 | 692.76  |     685.018 |
        | 2025-12-30 | 693.472 |     685.68  |
        | 2025-12-31 | 692.298 |     685.065 |
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

        holt_winters_moving_average = calculate_per_ticker(
            historical_data,
            [close_column],
            overlap_model.get_holt_winters_moving_average,
            level_smoothing,
            velocity_smoothing,
            acceleration_smoothing,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=holt_winters_moving_average,
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
    def get_holt_winters_channel(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        level_smoothing: float = 0.2,
        velocity_smoothing: float = 0.1,
        acceleration_smoothing: float = 0.1,
        deviation_smoothing: float = 0.1,
        multiplier: float = 1.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Holt-Winters Channel for a given price series.

        The channel draws bands around the Holt-Winters Moving Average at a multiple of an
        exponentially smoothed standard deviation of the close from the average. It widens
        when the price strays from its trend and narrows when it follows it, so a close outside
        the channel marks a move the trend does not explain.

        The formula is as follows:

        - Middle Band = HWMA (see `get_holt_winters_moving_average`)
        - Var(t) = (1 - Deviation Smoothing) * Var(t-1) + Deviation Smoothing * (Close(t-1) - HWMA(t-1))^2,
          starting at 0
        - Upper Band = HWMA(t) + Multiplier * sqrt(Var(t-1))
        - Lower Band = HWMA(t) - Multiplier * sqrt(Var(t-1))

        For more information, see the following sources:

        - Holt, C.C. (1957). "Forecasting Seasonals and Trends by Exponentially Weighted Moving
          Averages." ONR Memorandum 52, Carnegie Institute of Technology.
        - Winters, P.R. (1960). "Forecasting Sales by Exponentially Weighted Moving Averages."
          Management Science, 6(3), 324-342.
        - MQL5 Code Base: HWC (https://www.mql5.com/en/code/20857).
        - Referenced against pandas-ta-classic's hwc (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: HWC, Holt-Winters channel, Holt-Winter channel.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            level_smoothing (float, optional): The smoothing factor of the level. Defaults to 0.2.
            velocity_smoothing (float, optional): The smoothing factor of the velocity. Defaults to 0.1.
            acceleration_smoothing (float, optional): The smoothing factor of the acceleration.
                Defaults to 0.1.
            deviation_smoothing (float, optional): The smoothing factor of the variance. Defaults to 0.1.
            multiplier (float, optional): The number of standard deviations of the bands. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the bands.
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
            pd.DataFrame or pd.Series: The Upper Band, Middle Band and Lower Band.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the channel for each asset in the Toolkit instance.
        - The channel agrees with pandas-ta-classic's hwc. The band of a period uses the variance up
          to the period before, which holds the deviations up to two periods before, as in the
          MQL5 indicator. Since the average starts on the first close, the bands open from the
          fourth close on.
        - If `growth` is set to True, the method calculates the growth of the bands using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GOOGL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_holt_winters_channel().xs("GOOGL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Lower Band |   Middle Band |   Upper Band |
        |:-----------|-------------:|--------------:|-------------:|
        | 2025-12-24 |      295.027 |       307.333 |      319.639 |
        | 2025-12-26 |      294.832 |       306.652 |      318.472 |
        | 2025-12-29 |      294.639 |       306.02  |      317.401 |
        | 2025-12-30 |      294.532 |       305.508 |      316.484 |
        | 2025-12-31 |      294.23  |       304.872 |      315.512 |
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

        holt_winters_channel = calculate_per_ticker(
            historical_data,
            [close_column],
            overlap_model.get_holt_winters_channel,
            level_smoothing,
            velocity_smoothing,
            acceleration_smoothing,
            deviation_smoothing,
            multiplier,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=holt_winters_channel,
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
    def get_elastic_volume_weighted_moving_average(
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
        Calculate the Elastic Volume Weighted Moving Average (eVWMA) for a given price series.

        The eVWMA approximates the average price paid per share over the latest window. Every
        period the shares traded replace the same number of shares held at the previous
        average, out of all shares traded over the window, so the average moves towards the
        close by the share of the window's volume traded in that period: quickly on heavy
        volume and hardly at all on light volume. Unlike a moving average over a fixed number
        of periods, it stretches in time with the trading activity.

        The formula is as follows:

        - Volume Sum = Sum(Volume, Window)
        - eVWMA(t) = ((Volume Sum - Volume(t)) * eVWMA(t-1) + Volume(t) * Close(t)) / Volume Sum

        For more information, see the following sources:

        - Fries, C.P. (2001). "Elastic Moving Averages." Technical Analysis of Stocks & Commodities, 19(6),
          52-56 (https://www.christian-fries.de/evwma/).
        - Referenced against finta's EVWMA (https://github.com/peerchemist/finta).

        Also known as: eVWMA, EVWMA, elastic volume weighted moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the volume sum. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the eVWMA.
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
            pd.DataFrame or pd.Series: Elastic Volume Weighted Moving Average (eVWMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the eVWMA for each asset in the Toolkit instance.
        - Fries divides by the number of shares outstanding (the float); the volume over the
          window stands in for it, as in finta.
        - The recursion is seeded with the simple average of the first `window` closes, as the
          Toolkit seeds its exponential moving averages. finta starts it from 0, so its eVWMA
          climbs from 0 to the price; after that start has faded, about 300 periods with the
          default window, the two are equal.
        - A window without any volume keeps the last average.
        - If `growth` is set to True, the method calculates the growth of the eVWMA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_elastic_volume_weighted_moving_average().tail(5)
        ```

        Which returns:

        | Date       |    TSLA |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 449.617 |     669.122 |
        | 2025-12-26 | 450.621 |     669.555 |
        | 2025-12-29 | 451.012 |     670.084 |
        | 2025-12-30 | 451.144 |     670.443 |
        | 2025-12-31 | 451.098 |     670.743 |
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

        elastic_volume_weighted_moving_average = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            overlap_model.get_elastic_volume_weighted_moving_average,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=elastic_volume_weighted_moving_average,
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
    def get_volume_adjusted_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 8,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Volume Adjusted Moving Average (VAMA) for a given price series.

        The VAMA weights every close in the window by its traded value relative to the
        average volume of the window ending on that period, so a period that trades more
        than usual counts more than a quiet one, and the average leans towards the prices at
        which the market was most active.

        The formula is as follows:

        - Weight(t) = Close(t) * Volume(t) / SMA(Volume, Window)(t)
        - VAMA = Sum(Weight * Close, Window) / Sum(Weight, Window)

        For more information, see the following sources:

        - Arms, R.W. Jr. (1983). "Volume Cycles in the Stock Market: Market Timing Through
          Equivolume Charting." Dow Jones-Irwin, on adjusting prices for the volume traded.
        - Referenced against finta's VAMA (https://github.com/peerchemist/finta), by Peerchemist,
          which defines the indicator; it has no published source.

        Also known as: VAMA, volume adjusted moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods. Defaults to 8.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the VAMA.
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
            pd.DataFrame or pd.Series: Volume Adjusted Moving Average (VAMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the VAMA for each asset in the Toolkit instance.
        - The VAMA agrees with finta's VAMA, which has no published source; its weight multiplies
          the volume ratio by the close as well, so it is the traded value relative to the
          average volume.
        - A window without any volume has no average (NaN).
        - If `growth` is set to True, the method calculates the growth of the VAMA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NFLX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_volume_adjusted_moving_average().tail(5)
        ```

        Which returns:

        | Date       |    NFLX |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 94.0732 |     673.34  |
        | 2025-12-26 | 94.1407 |     674.161 |
        | 2025-12-29 | 94.1058 |     675.793 |
        | 2025-12-30 | 93.9638 |     678.813 |
        | 2025-12-31 | 93.9337 |     680.139 |
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

        volume_adjusted_moving_average = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            overlap_model.get_volume_adjusted_moving_average,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=volume_adjusted_moving_average,
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
    def get_gann_hilo_activator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        high_window: int = 13,
        low_window: int = 21,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Gann HiLo Activator for a given price series.

        The HiLo Activator is a trailing stop drawn from a moving average of the highs and one
        of the lows. A close above the previous average of the highs turns the trend up, after
        which the line follows the average of the lows beneath the price; a close below the
        previous average of the lows turns it down, after which the line follows the average
        of the highs above the price. A close between the two averages keeps the trend, so the
        line only flips side on a clear break.

        The formula is as follows:

        - High Average = SMA(High, High Window), Low Average = SMA(Low, Low Window)
        - Trend(t) = 1 if Close(t) > High Average(t-1), -1 if Close(t) < Low Average(t-1),
          otherwise Trend(t-1)
        - HiLo Activator(t) = Low Average(t) when the trend is 1, High Average(t) when it is -1

        For more information, see the following sources:

        - Krausz, R. (1998). "The New Gann Swing Chartist." Technical Analysis of Stocks &
          Commodities, 16(2).
        - pandas-ta-classic: hilo (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: HiLo, Gann HiLo, Gann HiLo Activator, HiLo Activator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted for
                dividends in the same way.
            high_window (int, optional): The number of periods of the average of the highs.
                Defaults to 13.
            low_window (int, optional): The number of periods of the average of the lows.
                Defaults to 21.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the activator.
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
            pd.DataFrame or pd.Series: The Gann HiLo Activator and the Trend Direction (1 up, -1 down).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the activator for each asset in the Toolkit instance.
        - The line follows the average of the side of the trend every period, as in Krausz's
          formula. pandas-ta-classic's hilo keeps the previous value of the line on a close
          between the two averages instead, so the two are equal on every period with a close
          beyond an average and differ on the others.
        - The defaults are those of pandas-ta-classic and TradingView (13 and 21 periods);
          Krausz used three-period averages of both.
        - If `growth` is set to True, the method calculates the growth of the activator using
          the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AMD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_gann_hilo_activator().xs("AMD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Gann HiLo Activator |   Trend Direction |
        |:-----------|----------------------:|------------------:|
        | 2025-12-24 |               217.501 |                -1 |
        | 2025-12-26 |               216.971 |                -1 |
        | 2025-12-29 |               216.295 |                -1 |
        | 2025-12-30 |               215.85  |                -1 |
        | 2025-12-31 |               215.553 |                -1 |
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

        gann_hilo_activator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            overlap_model.get_gann_hilo_activator,
            high_window,
            low_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=gann_hilo_activator,
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
    def get_zigzag(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        sensitivity: float = 0.05,
        minimum_trend_length: int = 1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Zig Zag for a given price series, as it is known on every period.

        The Zig Zag filters out moves smaller than the sensitivity. It follows the highest high
        of a rising leg until the low falls the sensitivity below it, which confirms that high
        as a swing high and starts a falling leg from that low, and the lowest low of a falling
        leg until the high rises the sensitivity above it. Drawn on a chart, the lines between
        the confirmed swing highs and lows are the zig zag, which shows the waves of the
        market without the noise in between.

        The formula is as follows:

        - In a rising leg: the trend turns down when Low(t) <= Extreme * (1 - Sensitivity),
          otherwise Extreme = High(t) when High(t) >= Extreme
        - In a falling leg: the trend turns up when High(t) >= Extreme * (1 + Sensitivity),
          otherwise Extreme = Low(t) when Low(t) <= Extreme
        - A leg only turns once at least Minimum Trend Length periods have passed since its
          extreme

        For more information, see the following sources:

        - Merrill, A.A. (1977). "Filtered Waves: Basic Theory." Analysis Press.
        - Referenced against TA-Lib's ZIGZAG (https://ta-lib.org/functions/zigzag.html).

        Also known as: ZIGZAG, Zig Zag, swing filter, filtered waves.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted for
                dividends in the same way.
            sensitivity (float, optional): The reversal from the extreme that turns a leg, as a
                decimal (0.05 is 5%). Defaults to 0.05.
            minimum_trend_length (int, optional): The number of periods since the extreme of a
                leg before it can turn. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values.
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
            pd.DataFrame or pd.Series: The Zig Zag (the extreme of the current leg), the Trend
            Direction (1 rising, -1 falling) and the Periods Since Pivot (since that extreme).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Zig Zag for each asset in the Toolkit instance.
        - Every period shows what is known at its close: the extreme of the current leg, its
          direction and how many periods ago it was set. This agrees with TA-Lib's ZIGZAG (with the
          sensitivity in percent there and the pivot as a position instead of the periods since
          it) and a value never changes once its period has passed. The line drawn on a chart
          instead connects the swings, and its last leg is redrawn as long as the leg runs: a
          swing high or low is only confirmed by the reversal that follows it, which here is the
          period the Trend Direction turns.
        - The first leg starts falling from the first low, as in TA-Lib, and the first
          `minimum_trend_length` periods are empty. A price equal to the extreme moves the pivot
          to the later period.
        - If `growth` is set to True, the method calculates the growth of the values using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SIUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_zigzag(sensitivity=0.1).xs("SIUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Periods Since Pivot |   Trend Direction |   Zig Zag |
        |:-----------|----------------------:|------------------:|----------:|
        | 2025-12-26 |                     0 |                 1 |    79.7   |
        | 2025-12-28 |                     0 |                 1 |    82.67  |
        | 2025-12-29 |                     0 |                -1 |    70.215 |
        | 2025-12-30 |                     0 |                 1 |    78.065 |
        | 2025-12-31 |                     0 |                -1 |    69.255 |
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

        zigzag = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            overlap_model.get_zigzag,
            sensitivity,
            minimum_trend_length,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=zigzag,
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
        anchor: str | None = None,
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
        - Anchored VWAP = Cumulative Sum(Typical Price * Volume) / Cumulative Sum(Volume), both
          summed from the anchor on

        The VWAP that trading desks use as an execution benchmark is calculated from the trades
        of a single session and resets every day. By default this method computes a rolling VWAP
        over several periods from the high, low, close and volume of each period, which works as
        a volume-weighted moving average. With an `anchor` the sums start at a fixed point
        instead: the first period ("all"), the start of every day, week, month, quarter or year
        (with intraday data and "daily" the session VWAP), or a date such as an earnings release
        or a market low, the anchored VWAP that shows the average price paid since that event.

        For more information, see the following sources:

        - Berkowitz, S.A., Logue, D.E., & Noser, E.A. Jr. (1988). "The Total Cost of Transactions
          on the NYSE." Journal of Finance, 43(1), 97-112.
        - Shannon, B. (2023). "Maximum Trading Gains With Anchored VWAP." Alphatrends Publishing.
        - Referenced against TA-Lib's VWAP (https://ta-lib.org/functions/vwap.html).
        - pandas-ta-classic: vwap with an anchor (https://github.com/xgboosted/pandas-ta-classic).
        - finta: VWAP (https://github.com/peerchemist/finta).

        Also known as: VWAP, anchored VWAP, AVWAP, session VWAP, cumulative VWAP.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods to consider for the VWAP.
                The number of periods (time intervals) over which to calculate the VWAP, when there is
                no anchor. Defaults to 14.
            anchor (str | None, optional): Where the sums start: "all" for the first period, "daily",
                "weekly", "monthly", "quarterly" or "yearly" for the start of every such period, or a
                date such as "2024-08-05" for the VWAP from that date on (empty before it). Defaults
                to None, which takes the rolling `window`.
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
        - The VWAP is taken over a rolling `window` by default. TA-Lib's VWAP accumulates the
          typical price times volume from the first period on, which is `anchor="all"` here (equal
          to TA-Lib's VWAP), and pandas-ta-classic's vwap starts the sums at every calendar period
          of its anchor, which is `anchor="daily"`, "weekly", "monthly", "quarterly" or "yearly".
        - The anchor "all" starts on the first period of the price history the Toolkit loaded,
          which can be before `start_date`.
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
            anchor,
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
        af_start_short: float | None = None,
        af_increment_short: float | None = None,
        af_max_short: float | None = None,
        start_value: float = 0.0,
        offset_on_reverse: float = 0.0,
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
        - AF starts at the initial acceleration factor and rises by the increment with every new
          extreme point, up to the maximum, each with its own value for long and short trades
        - On a reversal the SAR is the extreme point of the trade that ends, moved away from the
          price by the offset on reverse

        The extended parameters (separate acceleration factors for short trades, a start value and
        an offset on reverse) are those of TA-Lib's SAREXT; the defaults give Wilder's SAR.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - Referenced against TA-Lib's SAR and SAREXT (https://ta-lib.org/functions/sar.html,
          https://ta-lib.org/functions/sarext.html).
        - pandas-ta-classic: psar and sarext (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: Parabolic SAR, stop and reverse, PSAR, SAREXT, extended parabolic SAR.

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
            af_start_short (float | None, optional): Initial acceleration factor of a short trade.
                Defaults to None, which takes `af_start` (so the same factor for both directions).
            af_increment_short (float | None, optional): Increment of the acceleration factor of a
                short trade. Defaults to None, which takes `af_increment`.
            af_max_short (float | None, optional): Maximum acceleration factor of a short trade.
                Defaults to None, which takes `af_max`.
            start_value (float, optional): The SAR of the first trade: a positive value starts a long
                trade at that SAR and a negative value a short trade at its absolute value. Defaults
                to 0, which derives the start from the first two periods.
            offset_on_reverse (float, optional): The share by which the SAR moves away from the price
                on a reversal, as a decimal (0.01 is 1%). Defaults to 0.
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
        - The Parabolic SAR agrees with TA-Lib's SAR, which takes `af_increment` equal to `af_start`, and
          its SAREXT for any acceleration factors, start value and offset on reverse. TA-Lib's
          SAREXT returns the SAR negated during a short trade to mark the direction; here it is
          always the price level.
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
            af_start_short,
            af_increment_short,
            af_max_short,
            start_value,
            offset_on_reverse,
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
        method: Literal[
            "classic",
            "fibonacci",
            "camarilla",
            "woodie",
            "demark",
        ] = "classic",
        central_pivot_range: bool = False,
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
        the current period. The classic (floor trader) pivots are the default; the Fibonacci,
        Camarilla, Woodie and DeMark pivots place the levels differently, and the central pivot
        range adds a zone around the pivot point whose width reflects the previous period's range.

        The formula is as follows, with H, L, C and O the previous period's high, low, close and open:

        - "classic": Pivot Point (P) = (H + L + C) / 3, Resistance 1 = 2 * P — L, Support 1 = 2 * P — H,
          Resistance 2 = P + (H — L), Support 2 = P — (H — L), Resistance 3 = H + 2 * (P — L),
          Support 3 = L — 2 * (H — P)
        - "fibonacci": P = (H + L + C) / 3, Resistance n = P + Ratio * (H — L) and Support n =
          P — Ratio * (H — L), with the ratios 0.382, 0.618 and 1.0
        - "camarilla": P = (H + L + C) / 3, Resistance n = C + (H — L) * 1.1 / Divisor and Support n =
          C — (H — L) * 1.1 / Divisor, with the divisors 12, 6, 4 and 2
        - "woodie": P = (H + L + 2 * C) / 4, Resistance 1 = 2 * P — L, Support 1 = 2 * P — H,
          Resistance 2 = P + (H — L), Support 2 = P — (H — L)
        - "demark": X = H + 2 * L + C when C < O, 2 * H + L + C when C > O and H + L + 2 * C otherwise;
          P = X / 4, Resistance 1 = X / 2 — L, Support 1 = X / 2 — H
        - Central pivot range: Bottom Central Pivot = (H + L) / 2, Top Central Pivot =
          2 * P — Bottom Central Pivot

        For more information, see the following sources:

        - Person, J.L. (2004). "A Complete Guide to Technical Trading Tactics." Wiley.
        - DeMark, T.R. (1994). "The New Science of Technical Analysis." Wiley.
        - Ochoa, F. (2010). "Secrets of a Pivot Boss: Revealing Proven Methods for Profiting in the Market."
          311 Publishing.
        - Referenced against pandas-ta-classic's cpr of the classic, Fibonacci, Camarilla
          and Woodie pivots and the central pivot range (https://github.com/xgboosted/pandas-ta-classic).
        - finta: PIVOT and PIVOT_FIB (https://github.com/peerchemist/finta).

        Also known as: pivot points, floor trader pivots, Fibonacci pivots, Camarilla pivots, Woodie pivots,
        DeMark pivots, central pivot range, CPR.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            method (str, optional): The pivots: "classic", "fibonacci", "camarilla", "woodie" or
                "demark". Defaults to "classic".
            central_pivot_range (bool, optional): Whether to add the Top and Bottom Central Pivot.
                Defaults to False.
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
            Pivot Points: the pivot with resistance and support 1-3 (classic and Fibonacci), 1-4
            (Camarilla), 1-2 (Woodie) or 1 (DeMark), and the central pivots when asked for.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Pivot Points for each asset in the Toolkit instance.
        - Every level comes from the previous period, so it is known at the open of the period.
        - The classic, Fibonacci, Camarilla and Woodie pivots and the central pivot range equal
          pandas-ta-classic's cpr; the classic and Fibonacci pivots also agree with finta's PIVOT and
          PIVOT_FIB, which add a fourth level. The Woodie pivot weights the previous close twice,
          as in pandas-ta-classic; some charting packages take the current open instead.
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
            ["High", "Low", close_column, "Open"],
            overlap_model.get_pivot_points,
            method=method,
            central_pivot_range=central_pivot_range,
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

        For more information, see the following sources:

        - Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York Institute of Finance.

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
