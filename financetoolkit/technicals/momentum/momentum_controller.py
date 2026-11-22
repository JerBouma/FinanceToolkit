"""Momentum Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.technicals.helpers import calculate_per_ticker
from financetoolkit.technicals.momentum import momentum_model
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
class MomentumTechnicals:
    """
    Momentum indicators: oscillators such as the RSI, MACD, stochastic, Williams %R, Aroon, ADX and the directional movement system, CCI, money flow index, Ichimoku Cloud, rate of change, momentum, Know Sure Thing and the Coppock curve.
    """

    @handle_errors
    def collect_momentum_indicators(
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
        Calculates and collects various momentum indicators based on the provided data.

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
            pd.Series or pd.DataFrame: Momentum indicators calculated based on the specified parameters.

        Notes:
        - The method calculates various momentum indicators for each asset in the Toolkit instance.
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

        toolkit.technicals.collect_momentum_indicators().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Money Flow Index |   Williams %R |   Aroon Indicator Up |   Aroon Indicator Down |   Aroon Oscillator |   Commodity Channel Index |   Relative Vigor Index |   Relative Vigor Index Signal Line |       Force Index |   Ultimate Oscillator |   Percentage Price Oscillator |   Percentage Price Oscillator Signal Line |   Percentage Price Oscillator Histogram |   Absolute Price Oscillator |   Percentage Volume Oscillator |   Percentage Volume Oscillator Signal Line |   Percentage Volume Oscillator Histogram |   Detrended Price Oscillator |   Average Directional Index |   Plus Directional Movement |   Minus Directional Movement |   Plus Directional Indicator |   Minus Directional Indicator |   Directional Movement Index |   Average Directional Index Rating |   Chande Momentum Oscillator |   Intraday Momentum Index |   Ichimoku Conversion Line |   Ichimoku Base Line |   Ichimoku Leading Span A |   Ichimoku Leading Span B |   Stochastic %K |   Stochastic %D |   MACD Line |   MACD Signal Line |   MACD Histogram |   Relative Strength Index |   Stochastic RSI %K |   Stochastic RSI %D |   True Strength Index |   True Strength Index Signal Line |   Balance of Power |   Internal Bar Strength |   Qstick |   Awesome Oscillator |   Accelerator Oscillator |   Vortex Indicator VI+ |   Vortex Indicator VI- |   Elder Ray Bull Power |   Elder Ray Bear Power |   Rate of Change |   Momentum |   Choppiness Index |   Efficiency Ratio |   Vertical Horizontal Filter |   Know Sure Thing |   Know Sure Thing Signal Line |   Coppock Curve |
        |:-----------|-------------------:|--------------:|---------------------:|-----------------------:|-------------------:|--------------------------:|-----------------------:|-----------------------------------:|------------------:|----------------------:|------------------------------:|------------------------------------------:|----------------------------------------:|----------------------------:|-------------------------------:|-------------------------------------------:|-----------------------------------------:|-----------------------------:|----------------------------:|----------------------------:|-----------------------------:|-----------------------------:|------------------------------:|-----------------------------:|-----------------------------------:|-----------------------------:|--------------------------:|---------------------------:|---------------------:|--------------------------:|--------------------------:|----------------:|----------------:|------------:|-------------------:|-----------------:|--------------------------:|--------------------:|--------------------:|----------------------:|----------------------------------:|-------------------:|------------------------:|---------:|---------------------:|-------------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------:|-----------:|-------------------:|-------------------:|-----------------------------:|------------------:|------------------------------:|----------------:|
        | 2025-12-17 |            39.7393 |      -98.8223 |              28.5714 |               100      |           -71.4286 |                 -139.905  |                -0.2262 |                            -0.0758 |      -3.27419e+07 |               44.5331 |                        0.0063 |                                    0.0116 |                                 -0.0053 |                      1.7262 |                        -0.0329 |                                    -0.0509 |                                   0.018  |                       0.0306 |                     24.6712 |                     12.185  |                      13.7577 |                      17.868  |                       20.1742 |                       6.0621 |                            27.6361 |                     -20.6213 |                   41.1264 |                    275.43  |              276.008 |                   265.041 |                   250.519 |          1.1777 |          8.6605 |      1.7262 |             3.1596 |          -1.4334 |                   45.1865 |              0      |              1.2037 |               12.3226 |                           19.7487 |            -0.7013 |                  0.0442 |  -0.4015 |               1.2653 |                  -1.836  |                 0.9211 |                 1.0533 |                -0.1329 |                -4.6372 |          -0.0206 |    -5.6902 |            49.8426 |             0.2062 |                       0.5182 |            0.3638 |                        0.5107 |          0.0029 |
        | 2025-12-18 |            34.4002 |      -75.8191 |              21.4286 |               100      |           -78.5714 |                 -162.427  |                -0.2832 |                            -0.1659 |      -2.59748e+07 |               50.9808 |                        0.0047 |                                    0.0102 |                                 -0.0055 |                      1.2731 |                        -0.0132 |                                    -0.0434 |                                   0.0302 |                      -0.3822 |                     24.4323 |                     11.3147 |                      17.4487 |                      16.1683 |                       24.9337 |                      21.3258 |                            28.0927 |                     -24.9063 |                   36.3175 |                    272.6   |              276.008 |                   265.041 |                   250.519 |         24.1809 |         14.0381 |      1.2731 |             2.7823 |          -1.5092 |                   45.9183 |              2.4342 |              2.0152 |                9.3442 |                           17.1476 |            -0.2126 |                  0.7844 |  -0.6157 |               0.01   |                  -2.1487 |                 0.865  |                 1.0802 |                -2.0699 |                -8.7267 |          -0.0239 |    -6.6368 |            42.7376 |             0.2491 |                       0.5366 |            0.311  |                        0.4834 |         -0.014  |
        | 2025-12-19 |            43.3299 |      -68.9894 |              14.2857 |                92.8571 |           -78.5714 |                 -103.24   |                -0.2694 |                            -0.2349 |       5.92997e+06 |               51.5414 |                        0.0037 |                                    0.0089 |                                 -0.0052 |                      1.0212 |                         0.1453 |                                    -0.0056 |                                   0.1509 |                      -0.4186 |                     23.9077 |                     11.4731 |                      16.2023 |                      16.4688 |                       23.2574 |                      17.0883 |                            28.6043 |                     -39.3406 |                   28.2097 |                    272.6   |              276.008 |                   265.041 |                   250.519 |         31.0106 |         18.7897 |      1.0212 |             2.4301 |          -1.4089 |                   49.0174 |             12.7427 |              5.059  |                7.6468 |                           14.7724 |             0.3234 |                  0.8021 |  -0.8698 |              -1.1188 |                  -2.2693 |                 0.903  |                 1.137  |                -0.8133 |                -5.497  |          -0.0333 |    -9.3972 |            41.2483 |             0.3934 |                       0.5987 |            0.2645 |                        0.4504 |         -0.0269 |
        | 2025-12-22 |            36.4312 |      -81.4491 |               7.1429 |                85.7143 |           -78.5714 |                 -104.734  |                -0.2367 |                            -0.2614 |      -7.98092e+06 |               45.7604 |                        0.0022 |                                    0.0076 |                                 -0.0054 |                      0.5976 |                         0.1018 |                                     0.0158 |                                   0.0859 |                       2.2593 |                     23.4206 |                     10.6536 |                      15.045  |                      15.6561 |                       22.1096 |                      17.0883 |                            29.1455 |                     -64.5458 |                   17.6515 |                    272.6   |              276.008 |                   264.954 |                   250.519 |         18.5509 |         24.5808 |      0.5976 |             2.0636 |          -1.466  |                   44.0573 |              0      |              5.059  |                4.6682 |                           12.2463 |            -0.5608 |                  0.1365 |  -1.2314 |              -2.0902 |                  -2.1525 |                 0.8075 |                 1.2147 |                -0.8978 |                -4.2561 |          -0.0532 |   -15.1671 |            40.4149 |             0.6455 |                       0.5589 |            0.2055 |                        0.4111 |         -0.0418 |
        | 2025-12-23 |            37.2984 |      -69.5727 |               0      |                78.5714 |           -78.5714 |                  -97.9344 |                -0.205  |                            -0.2501 |      -1.44226e+06 |               49.1967 |                        0.0014 |                                    0.0063 |                                 -0.005  |                      0.3694 |                         0.0548 |                                     0.0236 |                                   0.0311 |                       2.3511 |                     23.1943 |                      9.8926 |                      14.9171 |                      14.9623 |                       22.5617 |                      20.2521 |                            28.9735 |                     -51.4171 |                   23.8077 |                    272.6   |              276.008 |                   265.083 |                   250.519 |         30.4273 |         26.6629 |      0.3694 |             1.7248 |          -1.3553 |                   47.029  |             15.5286 |              9.4238 |                3.0048 |                            9.9359 |             0.517  |                  0.9524 |  -0.9773 |              -2.6623 |                  -1.7431 |                 0.7732 |                 1.2661 |                -1.9284 |                -4.8582 |          -0.0415 |   -11.7491 |            46.452  |             0.5142 |                       0.4243 |            0.1522 |                        0.3663 |         -0.0501 |
        | 2025-12-24 |            41.6581 |      -51.6564 |               0      |                71.4286 |           -71.4286 |                  -34.6711 |                -0.1841 |                            -0.2228 |       2.20069e+06 |               54.4448 |                        0.0011 |                                    0.0053 |                                 -0.0042 |                      0.3017 |                        -0.0045 |                                     0.018  |                                  -0.0225 |                       3.0907 |                     22.0179 |                     12.1058 |                      13.8516 |                      18.7358 |                       21.4378 |                       6.7257 |                            28.2486 |                     -32.9192 |                   31.7545 |                    272.6   |              276.008 |                   265.063 |                   250.519 |         48.3436 |         32.4406 |      0.3017 |             1.4402 |          -1.1385 |                   50.0121 |             38.5455 |             18.0247 |                2.4525 |                            8.0651 |             0.4551 |                  0.4985 |  -0.6307 |              -2.8188 |                  -1.0828 |                 0.8892 |                 1.1613 |                 1.0804 |                -2.1384 |          -0.0245 |    -6.8661 |            53.1327 |             0.3292 |                       0.3731 |            0.1033 |                        0.3168 |         -0.0518 |
        | 2025-12-26 |            46.9441 |      -51.1363 |               0      |                64.2857 |           -64.2857 |                  -25.6923 |                -0.1697 |                            -0.1974 |  734929           |               51.2159 |                        0.0008 |                                    0.0044 |                                 -0.0036 |                      0.2126 |                        -0.0488 |                                     0.0046 |                                  -0.0534 |                      -0.6819 |                     20.9256 |                     11.2411 |                      12.8622 |                      17.986  |                       20.5798 |                       6.7257 |                            27.2888 |                     -27.7032 |                   33.1185 |                    272.6   |              276.098 |                   264.961 |                   250.519 |         48.8637 |         42.5448 |      0.2126 |             1.1946 |          -0.982  |                   49.169  |             34.6225 |             29.5655 |                1.7619 |                            6.4893 |            -0.3028 |                  0.2151 |  -0.5595 |              -2.1658 |                   0.0054 |                 0.9091 |                 1.1011 |                 1.1553 |                -1.346  |          -0.0193 |    -5.3613 |            55.4901 |             0.277  |                       0.4022 |            0.0564 |                        0.2631 |         -0.0522 |
        | 2025-12-29 |            48.1203 |      -48.4089 |              35.7143 |                57.1429 |           -21.4286 |                  -30.5662 |                -0.1563 |                            -0.1782 |       1.77136e+06 |               57.3149 |                        0.0006 |                                    0.0036 |                                 -0.003  |                      0.169  |                        -0.0817 |                                    -0.0126 |                                  -0.0691 |                       0.1103 |                     20.0593 |                     10.4382 |                      12.4517 |                      17.3859 |                       20.7397 |                       8.7966 |                            26.5056 |                     -21.8632 |                   36.4164 |                    270.612 |              276.183 |                   265.068 |                   250.519 |         51.5911 |         49.5994 |      0.169  |             0.9895 |          -0.8205 |                   49.9666 |             40.0254 |             37.7311 |                1.4003 |                            5.217  |             0.5323 |                  0.7015 |  -0.4662 |              -2.0654 |                   0.2951 |                 0.9316 |                 1.0955 |                 0.213  |                -1.79   |          -0.0149 |    -4.1156 |            54.4744 |             0.2186 |                       0.4134 |            0.0118 |                        0.2104 |         -0.0496 |
        | 2025-12-30 |            43.4593 |      -53.5605 |              28.5714 |                50      |           -21.4286 |                  -35.5265 |                -0.1288 |                            -0.1608 | -465185           |               54.1042 |                        0.0003 |                                    0.003  |                                 -0.0027 |                      0.0788 |                        -0.1122 |                                    -0.0325 |                                  -0.0797 |                      -2.3582 |                     19.2761 |                      9.6926 |                      11.632  |                      16.844  |                       20.2144 |                       9.0949 |                            25.7119 |                     -21.7388 |                   38.6752 |                    270.612 |              276.183 |                   265.068 |                   250.519 |         46.4395 |         48.9647 |      0.0788 |             0.8074 |          -0.7285 |                   48.421  |             29.5564 |             34.7348 |                0.6843 |                            4.0839 |             0.15   |                  0.4444 |  -0.3773 |              -1.9448 |                   0.3866 |                 0.9145 |                 1.1096 |                 0.0857 |                -1.708  |          -0.0148 |    -4.0857 |            53.5706 |             0.2174 |                       0.4141 |           -0.0182 |                        0.1612 |         -0.0465 |
        | 2025-12-31 |            38.2003 |      -62.8032 |              21.4286 |                42.8571 |           -21.4286 |                  -56.914  |                -0.0887 |                            -0.1381 |      -4.8276e+06  |               46.4079 |                       -0.0003 |                                    0.0023 |                                 -0.0026 |                     -0.0897 |                        -0.1258 |                                    -0.0512 |                                  -0.0746 |                      -1.5168 |                     18.7176 |                      9.0002 |                      11.3293 |                      16.2588 |                       20.4662 |                      11.4567 |                            24.6539 |                     -37.4457 |                   34.0262 |                    270.248 |              276.821 |                   265.6   |                   250.867 |         37.1968 |         45.0758 |     -0.0897 |             0.628  |          -0.7176 |                   45.6902 |             12.0097 |             27.1971 |               -0.6702 |                            2.8954 |            -0.6218 |                  0.057  |  -0.536  |              -1.6049 |                   0.5151 |                 0.9113 |                 1.1148 |                -0.0091 |                -1.9324 |          -0.0248 |    -6.896  |            52.5946 |             0.3745 |                       0.3956 |           -0.0452 |                        0.1157 |         -0.0431 |
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

        momentum_indicators: dict = {}

        momentum_indicators["Money Flow Index"] = self.get_money_flow_index(
            period=period, close_column=close_column, window=window
        )

        momentum_indicators["Williams %R"] = self.get_williams_percent_r(
            period=period, close_column=close_column, window=window
        )

        aroon_indicator = self.get_aroon_indicator(period=period, window=window)

        momentum_indicators["Aroon Indicator Up"] = aroon_indicator["Aroon Up"]
        momentum_indicators["Aroon Indicator Down"] = aroon_indicator["Aroon Down"]
        momentum_indicators["Aroon Oscillator"] = aroon_indicator["Aroon Oscillator"]

        momentum_indicators["Commodity Channel Index"] = (
            self.get_commodity_channel_index(
                period=period, close_column=close_column, window=window
            )
        )

        relative_vigor_index = self.get_relative_vigor_index(
            period=period, close_column=close_column
        )

        momentum_indicators["Relative Vigor Index"] = relative_vigor_index[
            "Relative Vigor Index"
        ]
        momentum_indicators["Relative Vigor Index Signal Line"] = relative_vigor_index[
            "Signal Line"
        ]

        momentum_indicators["Force Index"] = self.get_force_index(
            period=period, close_column=close_column, window=window
        )
        momentum_indicators["Ultimate Oscillator"] = self.get_ultimate_oscillator(
            period=period, close_column=close_column
        )
        percentage_price_oscillator = self.get_percentage_price_oscillator(
            period=period, close_column=close_column
        )

        momentum_indicators["Percentage Price Oscillator"] = (
            percentage_price_oscillator["Percentage Price Oscillator"]
        )
        momentum_indicators["Percentage Price Oscillator Signal Line"] = (
            percentage_price_oscillator["Signal Line"]
        )
        momentum_indicators["Percentage Price Oscillator Histogram"] = (
            percentage_price_oscillator["Histogram"]
        )
        momentum_indicators["Absolute Price Oscillator"] = (
            self.get_absolute_price_oscillator(period=period, close_column=close_column)
        )

        percentage_volume_oscillator = self.get_percentage_volume_oscillator(
            period=period
        )

        momentum_indicators["Percentage Volume Oscillator"] = (
            percentage_volume_oscillator["Percentage Volume Oscillator"]
        )
        momentum_indicators["Percentage Volume Oscillator Signal Line"] = (
            percentage_volume_oscillator["Signal Line"]
        )
        momentum_indicators["Percentage Volume Oscillator Histogram"] = (
            percentage_volume_oscillator["Histogram"]
        )

        momentum_indicators["Detrended Price Oscillator"] = (
            self.get_detrended_price_oscillator(
                period=period, close_column=close_column, window=window
            )
        )
        momentum_indicators["Average Directional Index"] = (
            self.get_average_directional_index(
                period=period, close_column=close_column, window=window
            )
        )

        directional_movement_index = self.get_directional_movement_index(
            period=period, close_column=close_column, window=window
        )

        momentum_indicators["Plus Directional Movement"] = directional_movement_index[
            "+DM"
        ]
        momentum_indicators["Minus Directional Movement"] = directional_movement_index[
            "-DM"
        ]
        momentum_indicators["Plus Directional Indicator"] = directional_movement_index[
            "+DI"
        ]
        momentum_indicators["Minus Directional Indicator"] = directional_movement_index[
            "-DI"
        ]
        momentum_indicators["Directional Movement Index"] = directional_movement_index[
            "DX"
        ]
        momentum_indicators["Average Directional Index Rating"] = (
            self.get_average_directional_index_rating(
                period=period, close_column=close_column, window=window
            )
        )
        momentum_indicators["Chande Momentum Oscillator"] = (
            self.get_chande_momentum_oscillator(
                period=period, close_column=close_column, window=window
            )
        )
        momentum_indicators["Intraday Momentum Index"] = (
            self.get_intraday_momentum_index(
                period=period, close_column=close_column, window=window
            )
        )

        ichimoku_cloud = self.get_ichimoku_cloud(period=period)

        momentum_indicators["Ichimoku Conversion Line"] = ichimoku_cloud[
            "Conversion Line"
        ]
        momentum_indicators["Ichimoku Base Line"] = ichimoku_cloud["Base Line"]
        momentum_indicators["Ichimoku Leading Span A"] = ichimoku_cloud[
            "Leading Span A"
        ]
        momentum_indicators["Ichimoku Leading Span B"] = ichimoku_cloud[
            "Leading Span B"
        ]

        stochastic_oscillator = self.get_stochastic_oscillator(
            period=period, close_column=close_column, window=window
        )

        momentum_indicators["Stochastic %K"] = stochastic_oscillator["Stochastic %K"]
        momentum_indicators["Stochastic %D"] = stochastic_oscillator["Stochastic %D"]

        macd = self.get_moving_average_convergence_divergence(
            period=period, close_column=close_column
        )

        momentum_indicators["MACD Line"] = macd["MACD Line"]
        momentum_indicators["MACD Signal Line"] = macd["Signal Line"]
        momentum_indicators["MACD Histogram"] = macd["Histogram"]

        momentum_indicators["Relative Strength Index"] = (
            self.get_relative_strength_index(
                period=period, close_column=close_column, window=window
            )
        )

        stochastic_rsi = self.get_stochastic_relative_strength_index(
            period=period, close_column=close_column, window=window
        )

        momentum_indicators["Stochastic RSI %K"] = stochastic_rsi["Stochastic RSI %K"]
        momentum_indicators["Stochastic RSI %D"] = stochastic_rsi["Stochastic RSI %D"]

        true_strength_index = self.get_true_strength_index(
            period=period, close_column=close_column
        )

        momentum_indicators["True Strength Index"] = true_strength_index[
            "True Strength Index"
        ]
        momentum_indicators["True Strength Index Signal Line"] = true_strength_index[
            "Signal Line"
        ]
        momentum_indicators["Balance of Power"] = self.get_balance_of_power(
            period=period, close_column=close_column
        )
        momentum_indicators["Internal Bar Strength"] = self.get_internal_bar_strength(
            period=period, close_column=close_column
        )
        momentum_indicators["Qstick"] = self.get_qstick(
            period=period, close_column=close_column, window=window
        )

        momentum_indicators["Awesome Oscillator"] = self.get_awesome_oscillator(
            period=period, close_column=close_column
        )
        momentum_indicators["Accelerator Oscillator"] = self.get_accelerator_oscillator(
            period=period, close_column=close_column
        )

        vortex_indicator = self.get_vortex_indicator(
            period=period, close_column=close_column, window=window
        )

        momentum_indicators["Vortex Indicator VI+"] = vortex_indicator["VI+"]
        momentum_indicators["Vortex Indicator VI-"] = vortex_indicator["VI-"]

        elder_ray_index = self.get_elder_ray_index(
            period=period, close_column=close_column
        )

        momentum_indicators["Elder Ray Bull Power"] = elder_ray_index["Bull Power"]
        momentum_indicators["Elder Ray Bear Power"] = elder_ray_index["Bear Power"]

        momentum_indicators["Rate of Change"] = self.get_rate_of_change(
            period=period, close_column=close_column, window=window
        )
        momentum_indicators["Momentum"] = self.get_momentum(
            period=period, close_column=close_column, window=window
        )

        momentum_indicators["Choppiness Index"] = self.get_choppiness_index(
            period=period, close_column=close_column, window=window
        )
        momentum_indicators["Efficiency Ratio"] = self.get_efficiency_ratio(
            period=period, close_column=close_column, window=window
        )
        momentum_indicators["Vertical Horizontal Filter"] = (
            self.get_vertical_horizontal_filter(
                period=period, close_column=close_column, window=window
            )
        )

        know_sure_thing = self.get_know_sure_thing(
            period=period, close_column=close_column
        )

        momentum_indicators["Know Sure Thing"] = know_sure_thing["KST"]
        momentum_indicators["Know Sure Thing Signal Line"] = know_sure_thing[
            "Signal Line"
        ]

        momentum_indicators["Coppock Curve"] = self.get_coppock_curve(
            period=period, close_column=close_column
        )

        self._momentum_indicators = pd.concat(momentum_indicators, axis=1)

        self._momentum_indicators = apply_rounding(
            self._momentum_indicators,
            rounding if rounding is not None else self._rounding,
        ).loc[self._start_date : self._end_date]

        if growth:
            self._momentum_indicators_growth = calculate_growth(
                dataset=self._momentum_indicators,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="index",
            )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                self._momentum_indicators_growth = calculate_standardization(
                    dataset=self._momentum_indicators_growth,
                    rounding=standardize_rounding,
                    axis="rows",
                )
            else:
                self._momentum_indicators = calculate_standardization(
                    dataset=self._momentum_indicators,
                    rounding=standardize_rounding,
                    axis="rows",
                )

        if len(self._tickers) == 1:
            return (
                self._momentum_indicators_growth.xs(
                    self._tickers[0], level=1, axis="columns"
                )
                if growth
                else self._momentum_indicators.xs(
                    self._tickers[0], level=1, axis="columns"
                )
            )

        return self._momentum_indicators_growth if growth else self._momentum_indicators

    @handle_portfolio
    @handle_errors
    def get_money_flow_index(
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
        Calculate the Money Flow Index (MFI) for a given price series.

        The Money Flow Index is a momentum indicator that measures the strength and
        direction of money flowing in and out of a security by considering both price
        and volume.

        The formula is as follows:

        - MFI = 100 — (100 / (1 + (positive_money_flow / negative_money_flow)))

        For more information, see the following sources:

        - TA-Lib: MFI, the reference implementation (https://ta-lib.org/functions/mfi.html).

        Also known as: MFI, volume-weighted RSI.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for calculating the MFI.
                Defaults to 14.
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
            pd.Series or pd.DataFrame: Money Flow Index (MFI) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the MFI values for each asset in the Toolkit instance.
        - The MFI equals TA-Lib's MFI.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^IXIC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_money_flow_index().tail(5)
        ```

        Which returns:

        | Date       |   ^IXIC |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 | 49.2154 |     53.8187 |
        | 2025-12-26 | 47.7867 |     52.3146 |
        | 2025-12-29 | 48.6922 |     51.9588 |
        | 2025-12-30 | 48.8975 |     52.4467 |
        | 2025-12-31 | 42.2704 |     45.3999 |
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

        money_flow_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            momentum_model.get_money_flow_index,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=money_flow_index,
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
    def get_williams_percent_r(
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
        Calculate the Williams Percent R (Williams %R) for a given price series.

        The Williams %R is a momentum indicator that measures the level of the close price
        relative to the high-low range over a certain number of periods.

        The formula is as follows:

        - Williams %R = (Highest High — Close) / (Highest High — Lowest Low) * —100

        For more information, see the following sources:

        - TA-Lib: WILLR, the reference implementation (https://ta-lib.org/functions/willr.html).

        Also known as: Williams percent R, overbought oversold oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for calculating the Williams %R.
                Defaults to 14.
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
            pd.Series or pd.DataFrame: Williams %R values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Williams %R values for each asset in the Toolkit instance.
        - Williams %R equals TA-Lib's WILLR.
        - If `growth` is set to True, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_williams_percent_r().tail(5)
        ```

        Which returns:

        | Date       |   EURUSD |   Benchmark |
        |:-----------|---------:|------------:|
        | 2025-12-26 | -29.4352 |     -6.0166 |
        | 2025-12-28 | -39.9621 |    nan      |
        | 2025-12-29 | -33.5227 |    -16.9802 |
        | 2025-12-30 | -56.8182 |    -20.7239 |
        | 2025-12-31 | -58.8068 |    -43.4088 |
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

        williams_percent_r = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_williams_percent_r,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=williams_percent_r,
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
    def get_aroon_indicator(
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
        Calculate the Aroon Indicator for a given price series.

        The Aroon Indicator is an oscillator that measures the strength of a trend and the
        likelihood of its continuation or reversal.

        The formula is as follows:

        - Aroon Up = ((Number of periods) — (Number of periods since highest high)) / (Number of periods) * 100
        - Aroon Down = ((Number of periods) — (Number of periods since lowest low)) / (Number of periods) * 100
        - Aroon Oscillator = Aroon Up — Aroon Down

        The Aroon Oscillator runs from -100 to 100: above zero the latest high is more recent than
        the latest low, which reads as an uptrend, and a reading near 100 or -100 as a strong trend.

        For more information, see the following sources:

        - Chande, T.S. (1995). "A Time Price Oscillator." Technical Analysis of Stocks & Commodities, 13(9).
        - TA-Lib: AROON, the reference implementation (https://ta-lib.org/functions/aroon.html).
        - TA-Lib: AROONOSC, the reference implementation (https://ta-lib.org/functions/aroonosc.html).

        Also known as: Aroon Up, Aroon Down, Aroon Oscillator, AROONOSC, trend strength.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for calculating the Aroon Indicator.
                Defaults to 14.
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
            pd.Series or pd.DataFrame:
            Aroon Indicator values for the upward and downward trends and the Aroon Oscillator.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Aroon Indicator values for each asset in the Toolkit instance.
        - Aroon Up and Aroon Down equal TA-Lib's AROON, and the Aroon Oscillator TA-Lib's AROONOSC.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_aroon_indicator().xs("GCUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Aroon Down |   Aroon Oscillator |   Aroon Up |
        |:-----------|-------------:|-------------------:|-----------:|
        | 2025-12-26 |       0      |           100      |   100      |
        | 2025-12-28 |       0      |            92.8571 |    92.8571 |
        | 2025-12-29 |       0      |            85.7143 |    85.7143 |
        | 2025-12-30 |      14.2857 |            64.2857 |    78.5714 |
        | 2025-12-31 |     100      |           -28.5714 |    71.4286 |
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

        aroon_indicator = calculate_per_ticker(
            historical_data, ["High", "Low"], momentum_model.get_aroon_indicator, window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=aroon_indicator,
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
    def get_commodity_channel_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        constant: float = 0.015,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Commodity Channel Index (CCI) for a given price series.

        The Commodity Channel Index is an oscillator that measures the current price level
        relative to an average price level over a specified period.

        The formula is as follows:

        - CCI = (Typical Price — SMA(Typical Price)) / (constant * Mean Deviation)

        For more information, see the following sources:

        - TA-Lib: CCI, the reference implementation (https://ta-lib.org/functions/cci.html).

        Also known as: CCI, cyclical trend indicator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for calculating the CCI.
                Defaults to 14.
            constant (float, optional): Constant multiplier used in the CCI calculation.
                Defaults to 0.015.
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
            pd.Series or pd.DataFrame: Commodity Channel Index (CCI) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the CCI values for each asset in the Toolkit instance.
        - With the default `constant` of 0.015, the CCI equals TA-Lib's CCI.
        - If `growth` is set to True, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_commodity_channel_index().tail(5)
        ```

        Which returns:

        | Date       |   CLUSD |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-26 | 15.2689 |    144.032  |
        | 2025-12-28 | -1.512  |    nan      |
        | 2025-12-29 | 51.2665 |     82.3342 |
        | 2025-12-30 | 69.184  |     67.1731 |
        | 2025-12-31 | 38.1997 |     13.3141 |
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

        commodity_channel_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_commodity_channel_index,
            window,
            constant,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=commodity_channel_index,
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
    def get_relative_vigor_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Vigor Index (RVI) for a given price series.

        The Relative Vigor Index measures the conviction behind a move by comparing where each
        period closes relative to its open with the period's full high-low range: in an uptrend
        prices tend to close above their open, in a downtrend below it.

        The formula is as follows:

        - Numerator = (Close — Open) smoothed with the symmetric weights 1, 2, 2, 1 over the
          current and three prior periods, divided by 6
        - Denominator = (High — Low) smoothed with the same weights
        - RVI = SMA(Numerator, window) / SMA(Denominator, window)
        - Signal Line = RVI smoothed with the same symmetric weights

        The RVI usually lies between -1 and 1 and is positive when closes sit above opens.
        Crossovers of the RVI and its signal line are the usual trigger, and a divergence from
        price warns that a move lacks conviction. See `momentum_model.get_relative_vigor_index`
        for the full formula.

        Also known as: RVI, RVGI, vigor index.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close", in which case the open, high and
                low prices are adjusted for dividends in the same way.
            window (int, optional): The number of periods for the averages of the numerator
                and denominator. Defaults to 10, Ehlers' setting.
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
            pd.Series or pd.DataFrame: Relative Vigor Index and Signal Line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the RVI values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^HSI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_relative_vigor_index().xs("^HSI", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Relative Vigor Index |   Signal Line |
        |:-----------|-----------------------:|--------------:|
        | 2025-12-24 |                -0.0322 |       -0.0822 |
        | 2025-12-26 |               nan      |      nan      |
        | 2025-12-29 |                -0.0004 |       -0.0522 |
        | 2025-12-30 |                -0.0074 |       -0.0247 |
        | 2025-12-31 |                -0.03   |       -0.013  |
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

        relative_vigor_index = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            momentum_model.get_relative_vigor_index,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=relative_vigor_index,
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
    def get_force_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 13,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Force Index for a given price series.

        The Force Index is an indicator that measures the strength behind price movements.

        The formula is as follows:

        - Raw Force Index = (Close — Close(1)) * Volume
        - Force Index = EMA(Raw Force Index, window)

        For more information, see the following sources:

        - TA-Lib: EFI, the reference implementation (https://ta-lib.org/functions/efi.html).

        Also known as: FI, Elder's Force Index.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods of the EMA that smooths the raw
                Force Index. Defaults to 13, Elder's setting for the trend (2 is his setting
                for timing entries).
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
            pd.Series or pd.DataFrame: Force Index values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Force Index values for each asset in the Toolkit instance.
        - The Force Index equals TA-Lib's EFI.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["TLT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_force_index().tail(5)
        ```

        Which returns:

        | Date       |               TLT |    Benchmark |
        |:-----------|------------------:|-------------:|
        | 2025-12-24 |  946514           |  8.27837e+07 |
        | 2025-12-26 | -287501           |  7.05413e+07 |
        | 2025-12-29 |       1.09756e+06 |  3.86575e+07 |
        | 2025-12-30 |  223430           |  2.75431e+07 |
        | 2025-12-31 |      -3.35421e+06 | -2.98818e+07 |
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

        force_index = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            momentum_model.get_force_index,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=force_index,
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
    def get_ultimate_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window_1: int = 7,
        window_2: int = 14,
        window_3: int = 28,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Ultimate Oscillator for a given price series.

        The Ultimate Oscillator is a momentum oscillator that combines short-term, mid-term,
        and long-term price momentum into a single value.

        The formula is as follows:

        - Average(i) = Sum(Buying Pressure, window_i) / Sum(True Range, window_i)
        - Ultimate Oscillator = 100 * [(4 * Average_1) + (2 * Average_2) + Average_3] / 7

        For more information, see the following sources:

        - TA-Lib: ULTOSC, the reference implementation (https://ta-lib.org/functions/ultosc.html).

        Also known as: UO, ultimate momentum oscillator. See
        `momentum_model.get_ultimate_oscillator` for the Buying Pressure and True Range
        definitions.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window_1 (int, optional): The number of periods for the first short-term window.
                Defaults to 7.
            window_2 (int, optional): The number of periods for the second mid-term window.
                Defaults to 14.
            window_3 (int, optional): The number of periods for the third long-term window.
                Defaults to 28.
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
            pd.Series or pd.DataFrame: Ultimate Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ultimate Oscillator values for each asset in the Toolkit instance.
        - The Ultimate Oscillator equals TA-Lib's ULTOSC. The first period counts with its High —
          Low range and its close minus its low, so the oscillator starts on period `window_3`, one
          period before TA-Lib's.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^GDAXI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_ultimate_oscillator().tail(5)
        ```

        Which returns:

        | Date       |   ^GDAXI |   Benchmark |
        |:-----------|---------:|------------:|
        | 2025-12-24 | nan      |     57.7084 |
        | 2025-12-26 | nan      |     56.8787 |
        | 2025-12-29 |  56.2772 |     64.508  |
        | 2025-12-30 |  61.9948 |     66.0178 |
        | 2025-12-31 | nan      |     52.8916 |
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

        ultimate_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_ultimate_oscillator,
            window_1,
            window_2,
            window_3,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=ultimate_oscillator,
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
    def get_percentage_price_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        moving_average: str = "exponential",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Percentage Price Oscillator (PPO) for a given price series.

        The Percentage Price Oscillator (PPO) is a momentum oscillator that measures the
        difference between two moving averages as a percentage of the longer moving average.

        The formula is as follows:

        - PPO = (Short-term EMA — Long-term EMA) / Long-term EMA
        - Signal Line = EMA(PPO, signal_window)
        - Histogram = PPO — Signal Line

        The PPO is the MACD line as a percentage of the long EMA, so unlike the MACD its level
        can be compared across stocks and over long periods. As for the MACD, the PPO crossing its
        signal line is the usual trigger and the histogram shows the distance between the two.
        `moving_average` replaces the EMAs of the PPO itself with another moving average, as
        TA-Lib's `matype` does; the signal line stays an EMA.

        For more information, see the following sources:

        - Appel, G. (2005). "Technical Analysis: Power Tools for Active Investors." FT Press.
        - TA-Lib: PPO, the reference implementation (https://ta-lib.org/functions/ppo.html).
        - pandas-ta-classic: ppo, with `mamode="ema"` (https://github.com/xgboosted/pandas-ta-classic).
        - ta: PercentagePriceOscillator (https://github.com/bukosabino/ta).
        - finta: PPO (https://github.com/peerchemist/finta).

        Also known as: PPO, price oscillator, PPO signal line, PPO histogram.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            short_window (int, optional): The number of periods for the short-term moving average.
                Defaults to 12, as for MACD.
            long_window (int, optional): The number of periods for the long-term moving average.
                Defaults to 26, as for MACD.
            signal_window (int, optional): The number of periods for the signal line EMA.
                Defaults to 9, as for MACD.
            moving_average (str, optional): The moving average of the short and long term:
                "simple", "exponential", "double_exponential", "triangular", "weighted", "hull" or
                "kaufman_adaptive". Defaults to "exponential".
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
            pd.Series or pd.DataFrame: The Percentage Price Oscillator, Signal Line and Histogram
            as decimals (0.02 is 2%).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the PPO values for each asset in the Toolkit instance.
        - The PPO is a decimal and equals TA-Lib's PPO with the same moving average (`matype`) divided
          by 100, since TA-Lib's PPO is in percent. TA-Lib has no signal line or histogram; with the
          default EMAs all three equal pandas-ta-classic's ppo with `mamode="ema"` divided by 100. ta and
          finta seed their EMAs with the first price instead of a simple average, so their first values
          differ until the two converge.
        - Every EMA is seeded with the simple average of its first values, so the PPO starts on the
          `long_window`-th period and the signal line and histogram `signal_window` - 1 periods later.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^STOXX50E", "^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_percentage_price_oscillator().xs("^GSPC", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Histogram |   Percentage Price Oscillator |   Signal Line |
        |:-----------|------------:|------------------------------:|--------------:|
        | 2025-12-24 |      0.001  |                        0.0045 |        0.0035 |
        | 2025-12-26 |      0.0013 |                        0.0051 |        0.0038 |
        | 2025-12-29 |      0.0011 |                        0.0052 |        0.0041 |
        | 2025-12-30 |      0.0009 |                        0.0051 |        0.0043 |
        | 2025-12-31 |      0.0001 |                        0.0044 |        0.0043 |
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

        percentage_price_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_percentage_price_oscillator,
            short_window,
            long_window,
            signal_window,
            moving_average,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=percentage_price_oscillator,
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
    def get_absolute_price_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 12,
        long_window: int = 26,
        moving_average: str = "exponential",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Absolute Price Oscillator (APO) for a given price series.

        The Absolute Price Oscillator is the difference between a short-term and a long-term moving
        average of the price, in price units. With exponential moving averages it is the MACD line; the
        APO lets the moving average be chosen, such as simple moving averages for a slower oscillator.

        The formula is as follows:

        - APO = MA(Close, short_window) — MA(Close, long_window)

        Above zero the short average is above the long one, which reads as upward momentum, and a cross
        of zero is a crossover of the two averages. As a price difference its level scales with the share
        price; the Percentage Price Oscillator divides it by the long average to compare stocks.

        For more information, see the following sources:

        - Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method." Signalert Corp.
        - Achelis, S.B. (2000). "Technical Analysis from A to Z." 2nd ed. McGraw-Hill.
        - TA-Lib: APO, the reference implementation (https://ta-lib.org/functions/apo.html).

        Also known as: APO, absolute price oscillator, price oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): The number of periods for the short-term moving average.
                Defaults to 12.
            long_window (int, optional): The number of periods for the long-term moving average.
                Defaults to 26.
            moving_average (str, optional): The moving average: "simple", "exponential",
                "double_exponential", "triangular", "weighted", "hull" or "kaufman_adaptive". Defaults to
                "exponential".
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
            pd.Series or pd.DataFrame: Absolute Price Oscillator (APO) values, in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the APO for each asset in the Toolkit instance.
        - The APO equals TA-Lib's APO with the same moving average (`matype`: SMA 0, EMA 1, WMA 2,
          DEMA 3, TRIMA 5, KAMA 6) from the first value, and with the default EMAs the MACD line of
          `get_moving_average_convergence_divergence`.
        - If `growth` is set to True, the method calculates the growth of the APO
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

        toolkit.technicals.get_absolute_price_oscillator(moving_average="simple").tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | -1.1002 | -0.4991 |      4.8402 |
        | 2025-12-26 | -1.6425 | -0.6586 |      4.3798 |
        | 2025-12-29 | -2.2587 |  0.0133 |      3.5335 |
        | 2025-12-30 | -2.9316 | -0.0005 |      2.1336 |
        | 2025-12-31 | -3.4789 | -0.0185 |      1.3652 |
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

        absolute_price_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_absolute_price_oscillator,
            short_window,
            long_window,
            moving_average,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=absolute_price_oscillator,
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
    def get_percentage_volume_oscillator(
        self,
        period: str = "daily",
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Percentage Volume Oscillator (PVO) for a given volume series.

        The Percentage Volume Oscillator is the Percentage Price Oscillator applied to volume: the
        difference between a short-term and a long-term EMA of volume as a share of the long-term
        EMA, with a signal line and a histogram as for the MACD.

        The formula is as follows:

        - PVO = (EMA(Volume, short_window) — EMA(Volume, long_window)) /
          EMA(Volume, long_window)
        - Signal Line = EMA(PVO, signal_window)
        - Histogram = PVO — Signal Line

        A positive PVO means recent volume runs above its longer-term average, which gives weight to
        a breakout or a trend in price; a PVO below zero in a rising market says the move happens on
        thinning volume. Because it is a share of the average volume, the PVO can be compared across
        stocks whatever the number of shares they trade. It says nothing about the direction of
        price.

        For more information, see the following sources:

        - TA-Lib: PVO, the reference implementation (https://ta-lib.org/functions/pvo.html).

        Also known as: PVO, volume oscillator. See `momentum_model.get_percentage_volume_oscillator`
        for the references.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            short_window (int, optional): The number of periods for the short-term moving average.
                Defaults to 12, as for the PPO and MACD.
            long_window (int, optional): The number of periods for the long-term moving average.
                Defaults to 26, as for the PPO and MACD.
            signal_window (int, optional): The number of periods for the signal line.
                Defaults to 9.
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
            pd.Series or pd.DataFrame: The Percentage Volume Oscillator, Signal Line and Histogram
            as decimals (0.02 is 2%), as the Percentage Price Oscillator.

        Notes:
        - The method retrieves the volumes of the specified `period` and calculates the PVO for each
          asset in the Toolkit instance. Every EMA is seeded with the simple average of its first
          values, so the PVO starts on the `long_window`-th period and the signal line and histogram
          `signal_window` - 1 periods later.
        - A ticker without volume, such as an index or a currency pair, has no PVO.
        - The PVO is a decimal, like the Percentage Price Oscillator, and NaN for a ticker without
          volume. It equals TA-Lib's PVO (with its default EMA) divided by 100: TA-Lib's PVO is in
          percent, returns 0 without volume and has no signal line or histogram.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_percentage_volume_oscillator().xs("NVDA", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Histogram |   Percentage Volume Oscillator |   Signal Line |
        |:-----------|------------:|-------------------------------:|--------------:|
        | 2025-12-24 |     -0.0392 |                        -0.0687 |       -0.0295 |
        | 2025-12-26 |     -0.0393 |                        -0.0786 |       -0.0393 |
        | 2025-12-29 |     -0.0451 |                        -0.0957 |       -0.0506 |
        | 2025-12-30 |     -0.0559 |                        -0.1205 |       -0.0646 |
        | 2025-12-31 |     -0.0509 |                        -0.1282 |       -0.0773 |
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

        historical_data = self._historical_data[period]

        percentage_volume_oscillator = calculate_per_ticker(
            historical_data,
            ["Volume"],
            momentum_model.get_percentage_volume_oscillator,
            short_window,
            long_window,
            signal_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=percentage_volume_oscillator,
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
    def get_detrended_price_oscillator(
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
        Calculate the Detrended Price Oscillator (DPO) for a given price series.

        The Detrended Price Oscillator (DPO) is an indicator that helps identify short-term cycles
        by removing longer-term trends from prices.

        The formula is as follows:

        - Displacement = floor(Number of Periods / 2) + 1
        - DPO = Close(t — Displacement) — SMA(Close, Number of Periods)(t)

        For more information, see the following sources:

        - TA-Lib: DPO, the reference implementation (https://ta-lib.org/functions/dpo.html).

        Also known as: DPO, detrended price oscillator. Note the moving average itself is not
        shifted — only the close price used for the comparison is looked up further back in
        time; see `momentum_model.get_detrended_price_oscillator` for the full explanation.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods to consider for the DPO calculation.
                Defaults to 14.
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
            pd.Series or pd.DataFrame: Detrended Price Oscillator (DPO) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the DPO values for each asset in the Toolkit instance.
        - The DPO equals TA-Lib's DPO.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NGUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_detrended_price_oscillator().tail(5)
        ```

        Which returns:

        | Date       |   NGUSD |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-26 | -0.0656 |     -3.2958 |
        | 2025-12-28 | -0.1575 |    nan      |
        | 2025-12-29 | -0.0724 |     -5.5778 |
        | 2025-12-30 | -0.001  |    -13.3921 |
        | 2025-12-31 | -0.0537 |     -8.1194 |
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

        detrended_price_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_detrended_price_oscillator,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=detrended_price_oscillator,
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
    def get_average_directional_index(
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
        Calculate the Average Directional Index (ADX) for a given price series.

        The Average Directional Index (ADX) is an indicator that measures the strength of a trend,
        whether it's an uptrend or a downtrend.

        The formula is as follows:

        - ADX = Wilder's Smoothed Moving Average of DX, where DX = 100 * |+DI — -DI| / (+DI + -DI)

        The ADX measures trend strength, not direction: a reading above about 25 is usually taken as a
        trending market and below 20 as a range, whether prices are rising or falling. Traders combine it
        with the +DI and -DI lines it is built from (`get_directional_movement_index`), or with the price
        itself, to know the direction. With Wilder's smoothing it reacts slowly and needs a few times the
        window of history before its values settle.

        For more information, see the following sources:

        - TA-Lib: ADX, the reference implementation (https://ta-lib.org/functions/adx.html).

        Also known as: ADX, trend strength indicator. See `momentum_model.get_average_directional_index`
        for the full formula, which uses Wilder's smoothing (not a plain SMA) throughout.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods to consider for the ADX calculation.
                Defaults to 14.
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
            pd.DataFrame or pd.Series: Average Directional Index (ADX) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the ADX values for each asset in the Toolkit instance.
        - Wilder's smoothing is seeded with the average of the first `window` values, while TA-Lib's
          ADX seeds its running sums with `window` - 1 values, so the first values differ from
          TA-Lib's until the two converge (within 0.01% after about 120 periods with the default
          window).
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["USDJPY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_average_directional_index().tail(5)
        ```

        Which returns:

        | Date       |   USDJPY |   Benchmark |
        |:-----------|---------:|------------:|
        | 2025-12-26 |  12.9436 |     14.8316 |
        | 2025-12-28 |  12.0867 |    nan      |
        | 2025-12-29 |  11.4583 |     14.7623 |
        | 2025-12-30 |  11.0975 |     14.698  |
        | 2025-12-31 |  10.4608 |     13.7116 |
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

        average_directional_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_average_directional_index,
            window,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=average_directional_index,
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
    def get_directional_movement_index(
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
        Calculate Wilder's Directional Movement System: the smoothed Plus and Minus Directional
        Movement (+DM, -DM), the Directional Indicators (+DI, -DI) and the Directional Movement
        Index (DX).

        The Directional Movement is the part of a period's range outside the previous period's range:
        upward when the high rises above the previous high by more than the low falls below the previous
        low, downward the other way around. Wilder sums each over the window with his smoothing and
        divides it by the smoothed True Range, which gives the Directional Indicators: the share of the
        recent range that moved up (+DI) and down (-DI). The DX is their difference as a share of their
        sum, high in a trend in either direction and low in a range; its Wilder average is the ADX.

        The formula is as follows:

        - +DM(1) = High — Previous High, where positive and larger than Previous Low — Low, else 0
        - -DM(1) = Previous Low — Low, where positive and larger than High — Previous High, else 0
        - +DM = Wilder's sum of +DM(1) over window, Sum(t) = Sum(t-1) — Sum(t-1) / window + +DM(1)(t),
          and -DM analogously
        - +DI = 100 * +DM / Wilder's sum of the True Range, -DI = 100 * -DM / Wilder's sum of the True Range
        - DX = 100 * |+DI — -DI| / (+DI + -DI)

        +DI crossing above -DI is Wilder's buy signal and crossing below it his sell signal, which he only
        took when the ADX showed a trend. The DX itself is jumpy; the ADX (`get_average_directional_index`)
        and the ADXR (`get_average_directional_index_rating`) smooth it.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - TA-Lib: PLUS_DM, the reference implementation (https://ta-lib.org/functions/plus_dm.html).
        - TA-Lib: MINUS_DM, the reference implementation (https://ta-lib.org/functions/minus_dm.html).
        - TA-Lib: PLUS_DI, the reference implementation (https://ta-lib.org/functions/plus_di.html).
        - TA-Lib: MINUS_DI, the reference implementation (https://ta-lib.org/functions/minus_di.html).
        - TA-Lib: DX, the reference implementation (https://ta-lib.org/functions/dx.html).

        Also known as: DMI, Directional Movement System, +DM, -DM, +DI, -DI, DX, PLUS_DM, MINUS_DM,
        PLUS_DI, MINUS_DI, directional indicators.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of Wilder's smoothing. Defaults to 14.
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
            pd.Series or pd.DataFrame: +DM and -DM (Wilder's smoothed sums, in price units), +DI and -DI and the DX
            (between 0 and 100).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Directional Movement Index for each asset in the Toolkit instance.
        - The columns are the inputs of `get_average_directional_index`, whose ADX is Wilder's
          average of this DX, so the two are consistent. Wilder's smoothing is seeded with the sum of
          the first `window` values (from the second period, the first with a previous one), so every
          column starts `window` periods after the first price and the ADX 2 * `window` — 1 periods
          after it.
        - TA-Lib seeds its running sums with `window` - 1 values and starts PLUS_DM and MINUS_DM one
          period earlier, so the first values differ from TA-Lib's PLUS_DM, MINUS_DM, PLUS_DI, MINUS_DI
          and DX and converge as the seed decays by a factor 1 — 1 / `window` per period: with the default
          window within 0.01% after about 100 periods for the +DM, -DM, +DI and -DI and 150 for the DX,
          and to 1e-8 after about 260.
        - A window without any directional movement (+DI and -DI both 0) has no DX.
        - If `growth` is set to True, the method calculates the growth of the Directional Movement Index
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_directional_movement_index().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |     +DI |     +DM |     -DI |     -DM |      DX |
        |:-----------|--------:|--------:|--------:|--------:|--------:|
        | 2025-12-24 | 18.7359 | 12.1058 | 21.4378 | 13.8516 |  6.7255 |
        | 2025-12-26 | 17.9861 | 11.2411 | 20.5798 | 12.8622 |  6.7255 |
        | 2025-12-29 | 17.386  | 10.4382 | 20.7397 | 12.4517 |  8.7965 |
        | 2025-12-30 | 16.844  |  9.6926 | 20.2144 | 11.6321 |  9.0948 |
        | 2025-12-31 | 16.2588 |  9.0003 | 20.4662 | 11.3294 | 11.4565 |
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

        directional_movement_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_directional_movement_index,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=directional_movement_index,
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
    def get_average_directional_index_rating(
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
        Calculate the Average Directional Movement Index Rating (ADXR).

        The ADXR is Wilder's smoothed ADX: the average of today's ADX and the ADX of window — 1 periods
        ago. Wilder rated markets with it on how strongly they trend, to choose the ones to trade with
        his trend-following systems; it moves more slowly than the ADX and lags it.

        The formula is as follows:

        - ADXR = (ADX(t) + ADX(t — (window — 1))) / 2

        with the ADX of `get_average_directional_index`. As for the ADX, a reading above about 25 is
        taken as a trending market and below 20 as a range, whatever the direction.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - TA-Lib: ADXR, the reference implementation (https://ta-lib.org/functions/adxr.html).

        Also known as: ADXR, Average Directional Movement Rating.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the ADX and of the lag of the rating. Defaults to 14.
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
            pd.Series or pd.DataFrame: Average Directional Movement Index Rating (ADXR) values, between 0 and
            100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the ADXR for each asset in the Toolkit instance.
        - The ADXR starts 3 * `window` — 2 periods after the first price, as in TA-Lib. It is built on the ADX of
          `get_average_directional_index`, whose Wilder smoothing is seeded with `window` values where
          TA-Lib seeds with `window` - 1, so the first values differ from TA-Lib's ADXR until the two
          converge (within 0.01% after about 100 to 125 periods with the default window).
        - If `growth` is set to True, the method calculates the growth of the ADXR
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

        toolkit.technicals.get_average_directional_index_rating().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 28.2487 | 23.3351 |     15.1933 |
        | 2025-12-26 | 27.2889 | 22.1898 |     15.5265 |
        | 2025-12-29 | 26.5057 | 21.2637 |     15.4615 |
        | 2025-12-30 | 25.712  | 20.8605 |     15.6433 |
        | 2025-12-31 | 24.654  | 20.5502 |     15.3675 |
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

        average_directional_index_rating = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_average_directional_index_rating,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=average_directional_index_rating,
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
    def get_chande_momentum_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        smoothed: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chande Momentum Oscillator (CMO) for a given price series.

        The Chande Momentum Oscillator is an indicator that measures the momentum of a price
        series and identifies overbought and oversold conditions.

        The formula is as follows:

        - CMO = 100 * ((Sum of Upward Change) — (Sum of Downward Change)) / ((Sum of Upward Change) +
          (Sum of Downward Change)), bounded between -100 and 100.
        - Smoothed CMO = 100 * (Wilder's MA(Upward Change) — Wilder's MA(Downward Change)) /
          (Wilder's MA(Upward Change) + Wilder's MA(Downward Change))

        Chande sums the changes over the window, so every change in the window counts equally and the
        CMO jumps when a large change leaves the window. With `smoothed=True` the changes are averaged
        with Wilder's smoothing, as in the RSI, which gives TA-Lib's CMO: a smoother line that equals
        2 * RSI — 100.

        For more information, see the following sources:

        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.
        - TA-Lib: CMOU, the reference implementation (https://ta-lib.org/functions/cmou.html).
        - TA-Lib: CMO, the reference implementation (https://ta-lib.org/functions/cmo.html).

        Also known as: CMO, CMOU, Chande momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods to consider for the CMO calculation.
                Defaults to 14.
            smoothed (bool, optional): Whether to smooth the gains and losses with Wilder's method
                (TA-Lib's CMO) instead of summing them (Chande's CMO, TA-Lib's CMOU). Defaults to False.
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
            pd.Series or pd.DataFrame: Chande Momentum Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Chande Momentum Oscillator values for each asset in the Toolkit instance.
        - By default the gains and losses are summed over the window, as Chande defines the CMO and as
          on StockCharts, which equals TA-Lib's CMOU. TA-Lib's CMO smooths them with Wilder's method
          instead, as in its RSI, so the two differ throughout (by up to about 55 points on daily data);
          `smoothed=True` gives TA-Lib's CMO, from the same period on.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SIUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chande_momentum_oscillator().tail(5)
        ```

        Which returns:

        | Date       |   SIUSD |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-26 | 59.7384 |     13.98   |
        | 2025-12-28 | 57.585  |    nan      |
        | 2025-12-29 | 26.3917 |     13.0139 |
        | 2025-12-30 | 38.9776 |     12.4231 |
        | 2025-12-31 | 15.5101 |     -7.4365 |
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

        chande_momentum_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_chande_momentum_oscillator,
            window,
            smoothed,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=chande_momentum_oscillator,
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
    def get_intraday_momentum_index(
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
        Calculate the Intraday Momentum Index (IMI) for a given price series.

        The Intraday Momentum Index is a Relative Strength Index of the moves within each period: it
        compares the bodies of the periods that closed above their open with all bodies over the window.
        Tushar Chande designed it for traders who care about the move from open to close rather than
        from close to close.

        The formula is as follows:

        - Gain = Close — Open where Close > Open, else 0
        - Loss = Open — Close where Close < Open, else 0
        - IMI = 100 * Sum(Gain, window) / (Sum(Gain, window) + Sum(Loss, window))

        As for the RSI, values above 70 are read as overbought and below 30 as oversold.

        For more information, see the following sources:

        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.
        - TA-Lib: IMI, the reference implementation (https://ta-lib.org/functions/imi.html).

        Also known as: IMI, intraday momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods to sum the gains and losses over. Defaults to 14.
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
            pd.Series or pd.DataFrame: Intraday Momentum Index (IMI) values, between 0 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the IMI for each asset in the Toolkit instance.
        - The IMI equals TA-Lib's IMI, except on a window in which every period closed at its open,
          which has no IMI (NaN) where TA-Lib returns 50.
        - If `growth` is set to True, the method calculates the growth of the IMI
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

        toolkit.technicals.get_intraday_momentum_index().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 31.7545 | 66.7169 |     45.7711 |
        | 2025-12-26 | 33.1186 | 66.9727 |     45.1725 |
        | 2025-12-29 | 36.4164 | 63.9814 |     48.7126 |
        | 2025-12-30 | 38.6752 | 62.7946 |     48.3366 |
        | 2025-12-31 | 34.0263 | 64.7409 |     36.4712 |
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

        intraday_momentum_index = calculate_per_ticker(
            historical_data,
            ["Open", close_column],
            momentum_model.get_intraday_momentum_index,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=intraday_momentum_index,
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
    def get_ichimoku_cloud(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        conversion_window: int = 9,
        base_window: int = 26,
        lead_span_b_window: int = 52,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Ichimoku Cloud indicator for a given price series.

        The Ichimoku Cloud, also known as the Ichimoku Kinko Hyo, is a versatile indicator that
        defines support and resistance, identifies trend direction, gauges momentum, and provides
        trading signals.

        The formula is as follows:

        - Conversion Line = (Highest High + Lowest Low) / 2, over conversion_window periods
        - Base Line = (Highest High + Lowest Low) / 2, over base_window periods
        - Leading Span A = ((Conversion Line + Base Line) / 2), shifted forward base_window periods
        - Leading Span B = (Highest High + Lowest Low) / 2 over lead_span_b_window periods,
          shifted forward base_window periods

        Price above the cloud is read as an uptrend and below it as a downtrend, with the cloud itself
        as support or resistance; a thick cloud is stronger support than a thin one. The leading spans
        are shifted forward by the base window, so the cloud value shown on a date was calculated
        base_window periods (26 by default) earlier from the data then available. The method returns the
        Conversion and Base Lines and the two Leading Spans but not the Lagging Span (the close shifted
        back), which would report a future close on each date.

        For more information, see the following sources:

        - TA-Lib: ICHIMOKU, the reference implementation (https://ta-lib.org/functions/ichimoku.html).

        Also known as: Ichimoku Kinko Hyo, cloud indicator. The default windows (9, 26, 52)
        are Goichi Hosoda's original values, both leading spans are conventionally displaced
        forward by the base (Kijun-sen) period.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            conversion_window (int, optional): The number of periods to consider for the
                Conversion Line (Tenkan-sen) calculation. Defaults to 9.
            base_window (int, optional): The number of periods to consider for the Base Line
                (Kijun-sen) calculation, also used as the forward displacement for both
                Leading Spans. Defaults to 26.
            lead_span_b_window (int, optional): The number of periods to consider for the
                Lead Span B (Senkou Span B) calculation. Defaults to 52.
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
            pd.Series or pd.DataFrame:
            Conversion Line, Base Line, Lead Span A, and Lead Span B values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ichimoku Cloud values for each asset in the Toolkit instance.
        - The Conversion Line and Base Line equal TA-Lib's ICHIMOKU, and so do the leading spans
          `base_window` periods earlier: TA-Lib places them on the period they are calculated on
          instead of displacing them forward, and starts every line on the `lead_span_b_window`-th
          period.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        - The leading spans are shifted forward by exactly `base_window` periods, as StockCharts
          defines the cloud. Platforms that count the current period as the first of the 26
          shift by `base_window` - 1 and show the same values one period earlier.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^N225"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_ichimoku_cloud().xs("^N225", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Base Line |   Conversion Line |   Leading Span A |   Leading Span B |
        |:-----------|------------:|------------------:|-----------------:|-----------------:|
        | 2025-12-25 |     49681.5 |           49640.4 |          49838.9 |          47250   |
        | 2025-12-26 |     49808.9 |           49792.8 |          49732.3 |          47250   |
        | 2025-12-29 |     49808.9 |           49792.8 |          49824   |          47351.7 |
        | 2025-12-30 |     49819.8 |           49792.8 |          49969.9 |          47710.2 |
        | 2025-12-31 |       nan   |             nan   |          49926.3 |          47990.2 |
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

        ichimoku_cloud = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            momentum_model.get_ichimoku_cloud,
            conversion_window,
            base_window,
            lead_span_b_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=ichimoku_cloud,
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
    def get_stochastic_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        smooth_window: int = 3,
        slow: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Stochastic Oscillator indicator for a given price series.

        The Stochastic Oscillator is a momentum indicator that shows the location of the close
        relative to the high-low range over a set number of periods. It consists of the %K line
        (fast) and the %D line (slow).

        The formula is as follows:

        - %K = 100 * ((Close — Lowest Low) / (Highest High — Lowest Low))
        - %D = SMA(%K, smooth_window)
        - Slow %K = SMA(%K, smooth_window), Slow %D = SMA(Slow %K, smooth_window)

        The fast stochastic (the default) reacts to every close and gives many crossovers. The slow
        stochastic (`slow=True`) smooths %K once more, so its %K is the fast %D; it is the stochastic
        George Lane charted and the one most platforms show by default (14, 3, 3).

        For more information, see the following sources:

        - Lane, G.C. (1984). "Lane's Stochastics." Technical Analysis of Stocks & Commodities, 2(3).
        - TA-Lib: STOCHF, the reference implementation (https://ta-lib.org/functions/stochf.html).
        - TA-Lib: STOCH, the reference implementation (https://ta-lib.org/functions/stoch.html).

        Also known as: stochastic oscillator, percent K, percent D, fast stochastic, slow stochastic.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods to consider for the %K line calculation.
                Defaults to 14.
            smooth_window (int, optional): The number of periods used to smooth the %K line
                into the %D signal line, and with `slow` the raw %K into the slow %K. Defaults to 3.
            slow (bool, optional): Whether to return the slow stochastic instead of the fast one.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the %K and %D values.
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
            pd.Series or pd.DataFrame: Stochastic Oscillator (%K and %D) values.

        Raises:
            ValueError: If the specified `period` is not one of the valid options.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Stochastic Oscillator values for each asset in the Toolkit instance.
        - %K and %D equal TA-Lib's STOCHF (the fast stochastic) with `fastd_period` set to
          `smooth_window` and a simple moving average (`fastd_matype=0`), which TA-Lib's STOCH gives
          with a `slowk_period` of 1. TA-Lib starts %K on the first %D.
        - With `slow=True`, %K and %D equal TA-Lib's STOCH with `slowk_period` and `slowd_period` set to
          `smooth_window` and simple moving averages (`slowk_matype=0`, `slowd_matype=0`), and
          pandas-ta-classic's stoch. TA-Lib starts %K on the first %D.
        - If `growth` is set to True, the method calculates the growth of the %K and %D values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GBPUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_stochastic_oscillator().xs("GBPUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Stochastic %D |   Stochastic %K |
        |:-----------|----------------:|----------------:|
        | 2025-12-26 |         86.4285 |         82.9586 |
        | 2025-12-28 |         85.136  |         82.9137 |
        | 2025-12-29 |         84.997  |         89.1187 |
        | 2025-12-30 |         80.5606 |         69.6493 |
        | 2025-12-31 |         74.4904 |         64.7032 |
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

        stochastic_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_stochastic_oscillator,
            window,
            smooth_window,
            slow,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=stochastic_oscillator,
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
    def get_moving_average_convergence_divergence(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        moving_average: str = "exponential",
        signal_moving_average: str = "exponential",
        fixed: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Moving Average Convergence Divergence (MACD) indicator for a given price series.

        The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator
        that shows the relationship between two moving averages of a security's price. It consists
        of the MACD line and its signal line; their difference is the MACD histogram.

        The formula is as follows:

        - MACD Line = EMA(Close, short_window) — EMA(Close, long_window)
        - Signal Line = EMA(MACD Line, signal_window)
        - Histogram = MACD Line — Signal Line

        The MACD line crossing its signal line is the usual trigger, and the MACD crossing zero means
        the short average has crossed the long one. The histogram is the distance between the two lines:
        it crosses zero with every crossover and shrinks before one, which is why it is read as an early
        warning. Because the MACD is a price difference, its level scales with the share price and cannot
        be compared across stocks; the Percentage Price Oscillator divides by the long average for that.
        Both lines lag price, so crossovers come late in fast reversals.

        `moving_average` and `signal_moving_average` replace the EMAs of the MACD line and of the signal
        line with another moving average (TA-Lib's MACDEXT), and `fixed=True` smooths the 12- and
        26-period EMAs with the fixed constants 0.15 and 0.075 instead of 2 / (window + 1) (TA-Lib's
        MACDFIX).

        For more information, see the following sources:

        - Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method." Signalert Corp.
        - Aspray, T. (1986). "Fine-tuning the MACD." Technical Analysis of Stocks & Commodities, 4(4).
        - TA-Lib: MACD, the reference implementation (https://ta-lib.org/functions/macd.html).
        - TA-Lib: MACDEXT, the reference implementation (https://ta-lib.org/functions/macdext.html).
        - TA-Lib: MACDFIX, the reference implementation (https://ta-lib.org/functions/macdfix.html).

        Also known as: MACD, MACD histogram, MACDEXT, MACDFIX, momentum indicator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): The number of periods for the shorter moving average.
                Defaults to 12.
            long_window (int, optional): The number of periods for the longer moving average.
                Defaults to 26.
            signal_window (int, optional): The number of periods for the signal line.
                Defaults to 9.
            moving_average (str, optional): The moving average of the short and long term: "simple",
                "exponential", "double_exponential", "triangular", "weighted", "hull" or
                "kaufman_adaptive". Defaults to "exponential".
            signal_moving_average (str, optional): The moving average of the signal line, one of the
                same. Defaults to "exponential".
            fixed (bool, optional): Whether to smooth the 12- and 26-period EMAs with the fixed
                constants 0.15 and 0.075 (TA-Lib's MACDFIX). Needs the exponential moving average and
                the default windows. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the MACD and signal values.
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
            MACD line, signal line and histogram values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the MACD and signal line values for each asset in the Toolkit instance.
        - TA-Lib's MACD starts the short EMA on the same period as the long EMA and shows the MACD
          line from the first signal value, so the first values differ from TA-Lib's until the two
          converge (within 0.01% after about 85 periods with the default windows). The histogram
          equals TA-Lib's `macdhist` and ta's `macd_diff` from then on.
        - TA-Lib's MACDEXT and MACDFIX start their averages in the same way. With simple, weighted or
          triangular moving averages, which have no recursion, the lines equal MACDEXT from the
          first value; with exponential ones (also MACDFIX) they converge within 0.01% after about 90
          periods, with double exponential ones after about 130 and with Kaufman adaptive ones, whose
          smoothing can be very slow, after about 210.
        - If `growth` is set to True, the method calculates the growth of the MACD and signal values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-10-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_moving_average_convergence_divergence().xs("MSFT", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Histogram |   MACD Line |   Signal Line |
        |:-----------|------------:|------------:|--------------:|
        | 2025-12-24 |      1.7933 |     -3.0142 |       -4.8074 |
        | 2025-12-26 |      1.8455 |     -2.5005 |       -4.3461 |
        | 2025-12-29 |      1.7825 |     -2.1179 |       -3.9004 |
        | 2025-12-30 |      1.7092 |     -1.7639 |       -3.4731 |
        | 2025-12-31 |      1.361  |     -1.7718 |       -3.1329 |
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

        macd = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_moving_average_convergence_divergence,
            short_window,
            long_window,
            signal_window,
            moving_average,
            signal_moving_average,
            fixed,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=macd,
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
    def get_relative_strength_index(
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
        Calculate the Relative Strength Index (RSI) indicator for a given price series.

        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and
        change of price movements. It ranges from 0 to 100 and is used to identify overbought or
        oversold conditions in an asset's price.

        The formula is as follows:

        - RSI = 100 — (100 / (1 + RS))

        Readings above 70 are conventionally read as overbought and below 30 as oversold, but in a
        strong trend the RSI can stay above 70 or below 30 for weeks, so a crossing is not a signal on
        its own. Practitioners look more at divergences, a new price high with a lower RSI high, and at
        the 50 line as a trend filter. The value depends on the window: a shorter window swings more and
        reaches the extremes more often.

        For more information, see the following sources:

        - TA-Lib: RSI, the reference implementation (https://ta-lib.org/functions/rsi.html).

        Also known as: RSI, momentum oscillator, overbought, oversold.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for RSI calculation. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the RSI.
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
            Relative Strength Index (RSI) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          RSI for each asset in the Toolkit instance.
        - The RSI equals TA-Lib's RSI.
        - If `growth` is set to True, the method calculates the growth of the RSI
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["BTCUSD", "ETHUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_relative_strength_index().tail(5)
        ```

        Which returns:

        | Date       |   BTCUSD |   ETHUSD |   Benchmark |
        |:-----------|---------:|---------:|------------:|
        | 2025-12-27 |  44.6846 |  45.0659 |    nan      |
        | 2025-12-28 |  44.9649 |  45.241  |    nan      |
        | 2025-12-29 |  42.4043 |  44.276  |     58.6691 |
        | 2025-12-30 |  47.7803 |  47.2561 |     57.6091 |
        | 2025-12-31 |  44.643  |  46.9887 |     51.5337 |
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

        relative_strength_index = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_relative_strength_index,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=relative_strength_index,
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
    def get_stochastic_relative_strength_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        stochastic_window: int = 14,
        smooth_window: int = 3,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Stochastic RSI for a given price series.

        The Stochastic RSI applies the Stochastic Oscillator to the Relative Strength Index
        instead of to price: it shows where the current RSI sits within its own high-low range
        over the last `stochastic_window` periods. Chande and Kroll (1994, "The New Technical
        Trader") designed it because the RSI can spend long stretches between 30 and 70 without
        reaching either extreme; the Stochastic RSI reaches 0 and 100 far more often.

        The formula is as follows:

        - RSI = Relative Strength Index(Close, window), with Wilder's smoothing
        - %K = 100 * (RSI — Lowest RSI) / (Highest RSI — Lowest RSI), over the last stochastic_window RSI values
        - %D = SMA(%K, smooth_window)

        The defaults follow Chande and Kroll's definition, a 14-period stochastic of a 14-period
        RSI, which StockCharts and TradingView use as well. StockCharts shows this %K on a scale of
        0 to 1; TradingView smooths it over 3 periods, so its %K is the %D here. TA-Lib's STOCHRSI
        defaults to a 5-period stochastic instead (timeperiod 14, fastk_period 5 and fastd_period 3
        with a simple average): pass stochastic_window=5 for its %K and %D. A %K of 100 means the
        RSI is at its highest of the window, not that the price is overbought, so the readings
        above 80 and below 20 come and go quickly and are mostly read together with the trend.

        For more information, see the following sources:

        - TA-Lib: STOCHRSI, the reference implementation (https://ta-lib.org/functions/stochrsi.html).

        Also known as: StochRSI, Stoch RSI.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the RSI. Defaults to 14.
            stochastic_window (int, optional): The number of periods of RSI values whose high-low
                range gives %K. Defaults to 14.
            smooth_window (int, optional): The number of periods used to smooth %K into the %D
                signal line. Defaults to 3.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the %K and %D values.
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
            pd.Series or pd.DataFrame: Stochastic RSI %K and %D values, between 0 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Stochastic RSI for each asset in the Toolkit instance.
        - The RSI is seeded with a simple average of its first `window` changes, as Wilder and
          TA-Lib do, so %K starts `window` + `stochastic_window` - 1 periods after the first
          price and %D `smooth_window` - 1 periods later. TA-Lib starts both on the first %D,
          with the same values from there.
        - An RSI that did not move over the stochastic window has no range, which leaves %K
          missing (TA-Lib sets it to 0).
        - With `fastk_period` set to `stochastic_window`, `fastd_period` to `smooth_window` and a
          simple moving average (`fastd_matype=0`), %K and %D equal TA-Lib's STOCHRSI.
        - If `growth` is set to True, the method calculates the growth of the %K and %D values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_stochastic_relative_strength_index().xs("NVDA", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Stochastic RSI %D |   Stochastic RSI %K |
        |:-----------|--------------------:|--------------------:|
        | 2025-12-24 |             98.8179 |             96.4537 |
        | 2025-12-26 |             98.8179 |            100      |
        | 2025-12-29 |             94.1715 |             86.0606 |
        | 2025-12-30 |             89.3317 |             81.9345 |
        | 2025-12-31 |             81.1377 |             75.418  |
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

        stochastic_rsi = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_stochastic_relative_strength_index,
            window,
            stochastic_window,
            smooth_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=stochastic_rsi,
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
    def get_true_strength_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        long_window: int = 25,
        short_window: int = 13,
        signal_window: int = 7,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the True Strength Index (TSI) for a given price series.

        The True Strength Index of Blau (1995, "Momentum, Direction, and Divergence") smooths
        the one-period price change twice with Exponential Moving Averages and divides it by
        the same double smoothing of the absolute change. The double smoothing filters out
        noise while lagging less than a single long average, and the division bounds the
        result between -100 (every change down) and +100 (every change up).

        The formula is as follows:

        - Momentum = Close(t) — Close(t - 1)
        - TSI = 100 * EMA(EMA(Momentum, long_window), short_window) / EMA(EMA(|Momentum|, long_window), short_window)
        - Signal Line = EMA(TSI, signal_window)

        Blau smooths over 25 and then 13 periods; the 7-period signal line is the one StockCharts
        uses. The TSI crossing zero means the smoothed momentum has turned, crossing its signal
        line is the faster trigger, and readings beyond about +25 or -25 are read as overbought
        or oversold. Like the RSI it is on a scale of -100 to 100 rather than a decimal.

        For more information, see the following sources:

        - TA-Lib: TSI, the reference implementation (https://ta-lib.org/functions/tsi.html).

        Also known as: TSI, double smoothed momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            long_window (int, optional): The number of periods of the first EMA of the price
                change. Defaults to 25.
            short_window (int, optional): The number of periods of the second EMA. Defaults to 13.
            signal_window (int, optional): The number of periods of the EMA of the TSI that gives
                the signal line. Defaults to 7.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the TSI and signal line.
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
            pd.Series or pd.DataFrame: True Strength Index and Signal Line values, between -100
            and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the True Strength Index for each asset in the Toolkit instance.
        - Every EMA is seeded with the simple average of its first values and is missing
          before it, as in TA-Lib and StockCharts, so the TSI starts `long_window` +
          `short_window` - 1 periods after the first price and the signal line
          `signal_window` - 1 periods later.
        - A stretch without any price change leaves the TSI missing.
        - The TSI equals TA-Lib's TSI, which has no signal line.
        - If `growth` is set to True, the method calculates the growth of the TSI and signal
          line using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_true_strength_index().xs("MSFT", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Signal Line |   True Strength Index |
        |:-----------|--------------:|----------------------:|
        | 2025-12-24 |      -12.4871 |               -8.9924 |
        | 2025-12-26 |      -11.3039 |               -7.7544 |
        | 2025-12-29 |      -10.1871 |               -6.8369 |
        | 2025-12-30 |       -9.1285 |               -5.9526 |
        | 2025-12-31 |       -8.3772 |               -6.1233 |
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

        true_strength_index = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_true_strength_index,
            long_window,
            short_window,
            signal_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=true_strength_index,
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
    def get_balance_of_power(
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
        Calculate the Balance of Power (BOP) indicator for a given price series.

        The Balance of Power (BOP) indicator measures the strength of buyers versus sellers
        in the market. It relates the price change to the change in the asset's trading range.

        The formula is as follows:

        - BOP = (Close — Open) / (High — Low)

        For more information, see the following sources:

        - TA-Lib: BOP, the reference implementation (https://ta-lib.org/functions/bop.html).

        Also known as: BOP, bull bear power.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the BOP.
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
            Balance of Power (BOP) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          BOP for each asset in the Toolkit instance.
        - The BOP equals TA-Lib's BOP.
        - If `growth` is set to True, the method calculates the growth of the BOP
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^FTSE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_balance_of_power().tail(5)
        ```

        Which returns:

        | Date       |    ^FTSE |   Benchmark |
        |:-----------|---------:|------------:|
        | 2025-12-24 |  -0.5953 |      0.802  |
        | 2025-12-26 | nan      |     -0.1381 |
        | 2025-12-29 |  -0.1212 |      0.099  |
        | 2025-12-30 |   0.7846 |     -0.2222 |
        | 2025-12-31 |  -0.279  |     -0.9239 |
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

        balance_of_power = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            momentum_model.get_balance_of_power,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=balance_of_power,
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
    def get_internal_bar_strength(
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
        Calculate the Internal Bar Strength (IBS) for a given price series.

        The Internal Bar Strength is where the close sits within the period's own range: 0 for a close
        on the low, 1 for a close on the high. It is a mean-reversion measure: in equity indices and ETFs
        a close near the low of the day (an IBS below about 0.2) has tended to be followed by higher
        returns the next day, and a close near the high (above about 0.8) by lower returns.

        The formula is as follows:

        - IBS = (Close — Low) / (High — Low)

        For more information, see the following sources:

        - Pagonidis, A.S. (2013). "The IBS Effect: Mean Reversion in Equity ETFs." NAAIM Wagner Award paper.
        - TA-Lib: IBS, the reference implementation (https://ta-lib.org/functions/ibs.html).

        Also known as: IBS, internal bar strength.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
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
            pd.Series or pd.DataFrame: Internal Bar Strength (IBS) values as a decimal between 0 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the IBS for each asset in the Toolkit instance.
        - The IBS equals TA-Lib's IBS, except on a period without a range (High equal to Low), which
          has no IBS (NaN) where TA-Lib returns 0.5.
        - If `growth` is set to True, the method calculates the growth of the IBS
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SPY", "QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_internal_bar_strength().tail(5)
        ```

        Which returns:

        | Date       |    QQQ |   Benchmark |
        |:-----------|-------:|------------:|
        | 2025-12-24 | 0.8633 |      0.8515 |
        | 2025-12-26 | 0.3151 |      0.4351 |
        | 2025-12-29 | 0.5284 |      0.5687 |
        | 2025-12-30 | 0.0709 |      0.2172 |
        | 2025-12-31 | 0.044  |      0.0372 |
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

        internal_bar_strength = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_internal_bar_strength,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=internal_bar_strength,
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
    def get_qstick(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Qstick indicator for a given price series.

        Qstick is the moving average of the candlestick bodies, Close — Open, over the window. It turns
        the colour of the recent candles into a number: above zero the periods closed above their open on
        average (white candles, buying pressure), below zero below their open (black candles, selling
        pressure). A cross of zero is read as a change of the pressure.

        The formula is as follows:

        - Qstick = SMA(Close — Open, window)

        For more information, see the following sources:

        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.
        - TA-Lib: QSTICK, the reference implementation (https://ta-lib.org/functions/qstick.html).

        Also known as: Qstick, QSTICK, Q-stick.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the moving average. Defaults to 10.
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
            pd.Series or pd.DataFrame: Qstick values, in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Qstick for each asset in the Toolkit instance.
        - Qstick equals TA-Lib's QSTICK and pandas-ta-classic's qstick.
        - If `growth` is set to True, the method calculates the growth of the Qstick
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

        toolkit.technicals.get_qstick().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -0.6886 | 1.1227 |     -0.5913 |
        | 2025-12-26 | -0.6577 | 0.5435 |     -1.0227 |
        | 2025-12-29 | -0.5889 | 0.8936 |     -0.3578 |
        | 2025-12-30 |  0.0399 | 1.571  |      0.0942 |
        | 2025-12-31 | -0.2581 | 0.7081 |     -0.3881 |
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

        qstick = calculate_per_ticker(
            historical_data,
            ["Open", close_column],
            momentum_model.get_qstick,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=qstick,
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
    def get_awesome_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 5,
        long_window: int = 34,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Awesome Oscillator (AO) for a given price series.

        The Awesome Oscillator measures market momentum by comparing a short-term and a
        long-term Simple Moving Average of the median price (the midpoint of each period's high
        and low, rather than the closing price). It was developed by Bill Williams as part of
        his broader "Trading Chaos" collection of momentum indicators.

        The formula is as follows:

        - Median Price = (High + Low) / 2
        - AO = SMA(Median Price, short_window) — SMA(Median Price, long_window)

        For more information, see the following sources:

        - TA-Lib: AO, the reference implementation (https://ta-lib.org/functions/ao.html).

        Also known as: AO, Bill Williams Awesome Oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): The number of periods for the short-term SMA of the
                median price. Defaults to 5.
            long_window (int, optional): The number of periods for the long-term SMA of the
                median price. Defaults to 34.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the AO.
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
            pd.Series or pd.DataFrame: Awesome Oscillator (AO) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the AO for each asset in the Toolkit instance.
        - There is no academic journal citation for the Awesome Oscillator. Like most of Bill
          Williams' indicators, it is a practitioner-developed tool rather than one derived from
          a published financial paper. The standard textbook source is Williams, B. (1995).
          "Trading Chaos: Applying Expert Techniques to Maximize Your Profit." Wiley.
        - A cross of the AO above zero occurs exactly when the short-window SMA of the median
          price crosses above the long-window SMA, and vice versa for a cross below zero.
        - The AO equals TA-Lib's AO.
        - If `growth` is set to True, the method calculates the growth of the AO
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SOLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_awesome_oscillator().tail(5)
        ```

        Which returns:

        | Date       |   SOLUSD |   Benchmark |
        |:-----------|---------:|------------:|
        | 2025-12-27 |  -8.9048 |    nan      |
        | 2025-12-28 |  -8.6621 |    nan      |
        | 2025-12-29 |  -7.5992 |     10.4586 |
        | 2025-12-30 |  -6.6618 |     11.0505 |
        | 2025-12-31 |  -5.546  |     10.5955 |
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

        awesome_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            momentum_model.get_awesome_oscillator,
            short_window,
            long_window,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=awesome_oscillator,
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
    def get_accelerator_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 5,
        long_window: int = 34,
        signal_window: int = 5,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Accelerator/Decelerator Oscillator (AC) for a given price series.

        The Accelerator Oscillator is the Awesome Oscillator minus its own 5-period Simple Moving Average.
        Where the Awesome Oscillator measures the momentum of the median price, the Accelerator Oscillator
        measures how that momentum changes, on Bill Williams' premise that momentum changes direction
        before price does.

        The formula is as follows:

        - Median Price = (High + Low) / 2
        - AO = SMA(Median Price, short_window) — SMA(Median Price, long_window)
        - AC = AO — SMA(AO, signal_window)

        Williams reads the AC by the colour of its bars: two rising bars in a row above zero (three below
        zero) are a buy signal, two falling bars below zero (three above zero) a sell signal.

        For more information, see the following sources:

        - Williams, B. (1998). "New Trading Dimensions: How to Profit from Chaos in Stocks, Bonds, and
          Commodities." Wiley.
        - TA-Lib: AC, the reference implementation (https://ta-lib.org/functions/ac.html).

        Also known as: AC, Acceleration/Deceleration Oscillator, AC oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            short_window (int, optional): The number of periods for the short-term SMA of the
                median price. Defaults to 5.
            long_window (int, optional): The number of periods for the long-term SMA of the
                median price. Defaults to 34.
            signal_window (int, optional): The number of periods for the SMA of the Awesome
                Oscillator. Defaults to 5.
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
            pd.Series or pd.DataFrame: Accelerator Oscillator (AC) values, in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the AC for each asset in the Toolkit instance.
        - The AC equals TA-Lib's AC and starts `long_window` + `signal_window` — 2 periods after the
          first price.
        - If `growth` is set to True, the method calculates the growth of the AC
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

        toolkit.technicals.get_accelerator_oscillator().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -1.0828 | 5.4289 |      3.1912 |
        | 2025-12-26 |  0.0054 | 4.1326 |      4.2287 |
        | 2025-12-29 |  0.2951 | 2.6361 |      3.6645 |
        | 2025-12-30 |  0.3866 | 1.8419 |      2.5872 |
        | 2025-12-31 |  0.5151 | 1.2335 |      0.8882 |
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

        accelerator_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            momentum_model.get_accelerator_oscillator,
            short_window,
            long_window,
            signal_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=accelerator_oscillator,
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
    def get_vortex_indicator(
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
        Calculate the Vortex Indicator for a given price series.

        The Vortex Indicator quantifies the presence and strength of a directional trend by
        comparing each period's price movement away from the prior period's range to the
        period's overall volatility (True Range). It consists of two lines, VI+ and VI-, whose
        crossovers signal potential trend changes: VI+ above VI- suggests an uptrend is in
        control, VI- above VI+ suggests a downtrend is in control.

        The formula is as follows:

        - VM+ = |High(t) — Low(t-1)|
        - VM- = |Low(t) — High(t-1)|
        - VI+ = Sum(VM+, window) / Sum(True Range, window)
        - VI- = Sum(VM-, window) / Sum(True Range, window)

        Also known as: VI, Vortex Indicator +/-, trend direction indicator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods to sum the directional movement and
                true range over. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the VI+ and VI- values.
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
            VI+ and VI- values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Vortex Indicator values for each asset in the Toolkit instance.
        - Reference: Botes, E., & Siepman, D. (2010). "The Vortex Indicator." Technical Analysis
          of Stocks & Commodities, 28(1), 20-25.
        - TA-Lib: VORTEX, the reference implementation (https://ta-lib.org/functions/vortex.html).
        - VI+ and VI- equal TA-Lib's VORTEX.
        - If `growth` is set to True, the method calculates the growth of the VI+ and VI-
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AUDUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_vortex_indicator().xs("AUDUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |    VI+ |    VI- |
        |:-----------|-------:|-------:|
        | 2025-12-26 | 1.1422 | 0.9189 |
        | 2025-12-28 | 1.1661 | 0.8887 |
        | 2025-12-29 | 1.1108 | 0.8889 |
        | 2025-12-30 | 1.1099 | 0.8697 |
        | 2025-12-31 | 1.0576 | 0.9069 |
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

        vortex_indicator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_vortex_indicator,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=vortex_indicator,
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
    def get_elder_ray_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 13,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Elder Ray Index (Bull Power and Bear Power) for a given price series.

        The Elder Ray Index measures buying and selling pressure in the market relative to a
        trend baseline (an Exponential Moving Average of the closing price). Bull Power captures
        how far the high extends above the EMA (buying pressure), while Bear Power captures how
        far the low extends below the EMA (selling pressure).

        The formula is as follows:

        - Bull Power = High — EMA(Close, window)
        - Bear Power = Low — EMA(Close, window)

        Also known as: Elder Ray, Bull Power, Bear Power.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for the EMA used as the trend baseline.
                Defaults to 13, as originally proposed by Elder.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Bull and Bear Power.
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
            Bull Power and Bear Power values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Elder Ray Index values for each asset in the Toolkit instance.
        - When the close is above the EMA (uptrend), Bull Power tends to stay positive and Bear
          Power moves toward zero from below; when the close is below the EMA (downtrend), both
          tend to be negative.
        - Reference: Elder, A. (1993). "Trading for a Living." Wiley.
        - TA-Lib: ERI, the reference implementation (https://ta-lib.org/functions/eri.html).
        - Bull Power and Bear Power equal TA-Lib's ERI.
        - If `growth` is set to True, the method calculates the growth of the Bull and Bear Power
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["HGUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_elder_ray_index().xs("HGUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Bear Power |   Bull Power |
        |:-----------|-------------:|-------------:|
        | 2025-12-26 |       0.1171 |       0.3356 |
        | 2025-12-28 |       0.1638 |       0.3223 |
        | 2025-12-29 |      -0.0388 |       0.3272 |
        | 2025-12-30 |      -0.0877 |       0.1728 |
        | 2025-12-31 |       0.0063 |       0.1473 |
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

        elder_ray_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_elder_ray_index,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=elder_ray_index,
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
    def get_rate_of_change(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 12,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Rate of Change (ROC) for a given price series.

        The Rate of Change is a pure momentum oscillator that measures the percentage
        change in price between the current period and the price a fixed number of periods
        ago. It oscillates around zero: positive values indicate price is higher than
        `window` periods ago (upward momentum), while negative values indicate price is
        lower (downward momentum).

        The formula is as follows:

        - ROC = Close(t) / Close(t - window) — 1

        Also known as: ROC, Price Rate of Change, momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to look back for the rate of change
                calculation. Defaults to 12.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Rate of Change.
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
            pd.Series or pd.DataFrame: Rate of Change values as a decimal (0.05 is 5%).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Rate of Change for each asset in the Toolkit instance.
        - Reference: Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New
          York Institute of Finance.
        - TA-Lib: ROCP, the reference implementation (https://ta-lib.org/functions/rocp.html).
        - The Rate of Change is a decimal and equals TA-Lib's ROCP; TA-Lib's ROC is the same in
          percent.
        - If `growth` is set to True, the method calculates the growth of the Rate of Change
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GCUSD", "SIUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_rate_of_change().tail(5)
        ```

        Which returns:

        | Date       |   GCUSD |   SIUSD |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-26 |  0.0518 |  0.245  |      0.0136 |
        | 2025-12-28 |  0.0411 |  0.2629 |    nan      |
        | 2025-12-29 |  0.0019 |  0.1081 |      0.0034 |
        | 2025-12-30 |  0.0125 |  0.2305 |     -0.0002 |
        | 2025-12-31 | -0.0075 |  0.0553 |      0.0032 |
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

        rate_of_change = calculate_per_ticker(
            historical_data, [close_column], momentum_model.get_rate_of_change, window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=rate_of_change,
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
    def get_momentum(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Momentum (MOM) for a given price series.

        Momentum is the change in price over a fixed number of periods, in price units: the simplest
        measure of the speed of a move. Above zero the price is higher than window periods ago, below
        zero lower; momentum that falls while the price still rises says the move slows down.

        The formula is as follows:

        - Momentum = Close(t) — Close(t — window)

        The Rate of Change (`get_rate_of_change`) is the same change as a share of the earlier price,
        which can be compared across stocks; the momentum is in the currency of the price.

        For more information, see the following sources:

        - Pring, M.J. (1991). "Technical Analysis Explained." 3rd ed. McGraw-Hill.
        - TA-Lib: MOM, the reference implementation (https://ta-lib.org/functions/mom.html).

        Also known as: MOM, price momentum, momentum oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods to look back. Defaults to 10.
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
            pd.Series or pd.DataFrame: Momentum values, the change in price in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Momentum for each asset in the Toolkit instance.
        - The Momentum equals TA-Lib's MOM, pandas-ta-classic's mom and finta's MOM.
        - If `growth` is set to True, the method calculates the growth of the Momentum
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

        toolkit.technicals.get_momentum().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | -4.9527 |  9.3822 |      4.7986 |
        | 2025-12-26 | -4.6139 |  4.2051 |      3.1462 |
        | 2025-12-29 | -4.5043 |  8.4996 |      8.0362 |
        | 2025-12-30 | -1.0264 | 12.5559 |      8.2217 |
        | 2025-12-31 | -2.7405 |  7.1706 |      5.0113 |
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

        momentum = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_momentum,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=momentum,
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
    def get_choppiness_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        range_method: str = "high_low",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Choppiness Index (CHOP) for a given price series.

        The Choppiness Index quantifies whether the market is trending or moving sideways
        ("choppy") by comparing the sum of True Range over the window (a measure of the
        total price path travelled) to the net range the price actually covered over that
        same window (the distance between the highest high and the lowest low). When price
        travels a long, winding path but ends up covering little net ground, the index is
        high (near 100), signalling a choppy, range-bound market. When price travels
        efficiently in one direction, the index is low (near 0), signalling a trending
        market.

        The formula is as follows:

        - CHOP = 100 * log10( Sum(True Range, window) / (Max(High, window) — Min(Low, window)) ) / log10(window)

        With `range_method="true_range"` the net range is that of the true highs and lows,
        Max(High, Previous Close) and Min(Low, Previous Close), the range the summed True Ranges span,
        so a gap between periods counts in both the path and the net range (TA-Lib's CHOPTR).

        For more information, see the following sources:

        - Kaufman, P.J. (2013). "Trading Systems and Methods." 5th ed. Wiley.
        - TA-Lib: CHOP, the reference implementation (https://ta-lib.org/functions/chop.html).
        - TA-Lib: CHOPTR, the reference implementation (https://ta-lib.org/functions/choptr.html).

        Also known as: CHOP, CHOPTR, Choppiness Index.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods to consider for the Choppiness Index
                calculation. Defaults to 14.
            range_method (str, optional): The net range the summed True Range is divided by:
                "high_low", the highest high minus the lowest low, or "true_range", the highest true
                high minus the lowest true low (TA-Lib's CHOPTR). Defaults to "high_low".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Choppiness Index.
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
            pd.Series or pd.DataFrame: Choppiness Index values, bounded between 0 and 100.
            Values above 61.8 are commonly read as signalling a choppy (range-bound) market,
            while values below 38.2 are commonly read as signalling a trending market.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Choppiness Index for each asset in the Toolkit instance.
        - Developed by Australian commodities trader Bill Dreiss; there is no formal journal
          citation. The standard textbook treatment is Kaufman, P.J. (2013). "Trading Systems
          and Methods." 5th ed. Wiley.
        - The Choppiness Index equals TA-Lib's CHOP, and with `range_method="true_range"` TA-Lib's
          CHOPTR. The first period counts with its High — Low range, so the index starts on period
          `window`, one period before TA-Lib's.
        - If `growth` is set to True, the method calculates the growth of the Choppiness
          Index using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["EURUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_choppiness_index().tail(5)
        ```

        Which returns:

        | Date       |   EURUSD |   Benchmark |
        |:-----------|---------:|------------:|
        | 2025-12-26 |  58.6157 |     50.3552 |
        | 2025-12-28 |  61.2072 |    nan      |
        | 2025-12-29 |  61.879  |     49.989  |
        | 2025-12-30 |  63.269  |     49.6207 |
        | 2025-12-31 |  63.0658 |     48.7037 |
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

        choppiness_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_choppiness_index,
            window,
            range_method,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=choppiness_index,
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
    def get_efficiency_ratio(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Kaufman's Efficiency Ratio (ER) for a given price series.

        The Efficiency Ratio compares the net change in price over the window with the path the price
        travelled to get there, the sum of the absolute changes of each period. A price that moves in a
        straight line has a ratio of 1; a price that ends where it started, however much it moved in
        between, has a ratio of 0. Perry Kaufman uses it to adapt the speed of his Adaptive Moving
        Average (`get_kaufman_adaptive_moving_average`) to the noise in the market.

        The formula is as follows:

        - Change = |Close(t) — Close(t — window)|
        - Volatility = Sum(|Close(i) — Close(i — 1)|, window)
        - Efficiency Ratio = Change / Volatility

        For more information, see the following sources:

        - Kaufman, P.J. (1995). "Smarter Trading: Improving Performance in Changing Markets." McGraw-Hill.
        - TA-Lib: ER, the reference implementation (https://ta-lib.org/functions/er.html).

        Also known as: ER, Kaufman Efficiency Ratio, fractal efficiency.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the change and the path. Defaults to 10.
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
            pd.Series or pd.DataFrame: Efficiency Ratio values as a decimal between 0 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Efficiency Ratio for each asset in the Toolkit instance.
        - The Efficiency Ratio equals TA-Lib's ER, pandas-ta-classic's er and finta's ER, except on a
          window without any change (flat prices), which has no ratio (NaN) where TA-Lib returns 1.
        - If `growth` is set to True, the method calculates the growth of the Efficiency Ratio
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

        toolkit.technicals.get_efficiency_ratio().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.3144 | 0.3229 |      0.1201 |
        | 2025-12-26 | 0.2993 | 0.1717 |      0.0818 |
        | 2025-12-29 | 0.2901 | 0.4207 |      0.2395 |
        | 2025-12-30 | 0.0852 | 0.743  |      0.2464 |
        | 2025-12-31 | 0.2147 | 0.374  |      0.137  |
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

        efficiency_ratio = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_efficiency_ratio,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=efficiency_ratio,
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
    def get_vertical_horizontal_filter(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 28,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Vertical Horizontal Filter (VHF) for a given price series.

        The Vertical Horizontal Filter tells a trending market from a congested one by dividing the range
        of the closes over the window (the vertical move) by the sum of the absolute changes from period
        to period (the path, horizontal in a range). A high or rising VHF says the price moves in one
        direction, which suits trend-following indicators; a low or falling VHF says it moves back and
        forth, which suits oscillators.

        The formula is as follows:

        - VHF = (Max(Close, window) — Min(Close, window)) / Sum(|Close(i) — Close(i — 1)|, window)

        For more information, see the following sources:

        - White, A. (1991). "Trending or Trading? The Vertical Horizontal Filter." Futures Magazine.
        - TA-Lib: VHF, the reference implementation (https://ta-lib.org/functions/vhf.html).

        Also known as: VHF, vertical horizontal filter.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the range and the path. Defaults to 28.
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
            pd.Series or pd.DataFrame: Vertical Horizontal Filter (VHF) values, a ratio between 0 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the VHF for each asset in the Toolkit instance.
        - The VHF equals TA-Lib's VHF and pandas-ta-classic's vhf and starts `window` periods after the
          first price, except on a window without any change (flat prices), which has no VHF (NaN) where TA-Lib
          returns 0.
        - If `growth` is set to True, the method calculates the growth of the VHF
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

        toolkit.technicals.get_vertical_horizontal_filter().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.3606 | 0.2618 |      0.3575 |
        | 2025-12-26 | 0.3614 | 0.2547 |      0.3576 |
        | 2025-12-29 | 0.3942 | 0.1559 |      0.3702 |
        | 2025-12-30 | 0.3891 | 0.1661 |      0.3873 |
        | 2025-12-31 | 0.3884 | 0.1701 |      0.3779 |
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

        vertical_horizontal_filter = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_vertical_horizontal_filter,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=vertical_horizontal_filter,
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
    def get_know_sure_thing(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        roc_windows: list[int] | None = None,
        sma_windows: list[int] | None = None,
        weights: list[int] | None = None,
        signal_window: int = 9,
        moving_average: str = "simple",
        signal_moving_average: str = "simple",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Know Sure Thing (KST) for a given price series.

        The Know Sure Thing is a momentum oscillator developed by Martin Pring that combines
        four smoothed Rate of Change series, each calculated over a progressively longer
        lookback period, into a single weighted sum. Smoothing each Rate of Change with a
        Simple Moving Average before combining them reduces noise, while the increasing
        weights on the longer lookback periods give more influence to the more significant,
        longer-term price cycles. A signal line (a Simple Moving Average of the KST itself)
        is used to spot crossovers, in the same way the MACD line is compared to its signal
        line.

        The formula is as follows:

        - RCMA(i) = SMA(ROC(Close, roc_windows[i]), sma_windows[i])
        - KST = Sum(RCMA(i) * weights[i]) for i = 1..4
        - Signal Line = SMA(KST, signal_window)

        Pring smooths with Simple Moving Averages; `moving_average` and `signal_moving_average` smooth
        the Rates of Change and the signal line with another moving average instead, as TA-Lib's KSTEXT
        does.

        For more information, see the following sources:

        - Pring, M.J. (1992). "The Know Sure Thing (KST)." Technical Analysis of Stocks & Commodities, 10(6).
        - TA-Lib: KST, the reference implementation (https://ta-lib.org/functions/kst.html).
        - TA-Lib: KSTEXT, the reference implementation (https://ta-lib.org/functions/kstext.html).

        Also known as: KST, KSTEXT, Pring's Know Sure Thing, Summed Rate of Change.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            roc_windows (list[int] | None, optional): The four lookback periods used for the
                underlying Rate of Change calculations. Defaults to the standard
                [10, 15, 20, 30].
            sma_windows (list[int] | None, optional): The four Simple Moving Average
                smoothing periods applied to each Rate of Change series. Defaults to the
                standard [10, 10, 10, 15].
            weights (list[int] | None, optional): The four weights applied to each smoothed
                Rate of Change series before summing. Defaults to the standard [1, 2, 3, 4].
            signal_window (int, optional): Number of periods for the Simple Moving Average of
                the KST used as the signal line. Defaults to 9.
            moving_average (str, optional): The moving average that smooths each Rate of Change:
                "simple", "exponential", "double_exponential", "triangular", "weighted", "hull" or
                "kaufman_adaptive". Defaults to "simple".
            signal_moving_average (str, optional): The moving average of the signal line, one of the
                same. Defaults to "simple".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the KST and Signal Line.
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
            KST and Signal Line values, weighted sums of Rates of Change as decimals.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Know Sure Thing values for each asset in the Toolkit instance.
        - The KST and Signal Line are decimals and equal TA-Lib's KST divided by 100, since TA-Lib's
          KST is in percent. TA-Lib starts the KST on the first signal value.
        - With other moving averages they equal TA-Lib's KSTEXT with the same `rocmatype` and
          `signalmatype` divided by 100: from the first value for the simple, weighted and triangular
          averages, and after the warm-up for the recursive ones, since TA-Lib starts their recursion
          later (within 0.01% after about 85 periods for exponential averages).
        - If `growth` is set to True, the method calculates the growth of the KST and Signal
          Line using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_know_sure_thing().xs("^GSPC", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |    KST |   Signal Line |
        |:-----------|-------:|--------------:|
        | 2025-12-24 | 0.1217 |        0.1297 |
        | 2025-12-26 | 0.1149 |        0.1271 |
        | 2025-12-29 | 0.1085 |        0.1233 |
        | 2025-12-30 | 0.1088 |        0.1192 |
        | 2025-12-31 | 0.1105 |        0.1166 |
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

        know_sure_thing = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_know_sure_thing,
            roc_windows,
            sma_windows,
            weights,
            signal_window,
            moving_average,
            signal_moving_average,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=know_sure_thing,
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
    def get_coppock_curve(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        long_roc_window: int = 14,
        short_roc_window: int = 11,
        wma_window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Coppock Curve for a given price series.

        The Coppock Curve is a long-term momentum indicator that Edwin Coppock designed for
        monthly closes of a stock index, to time buying after a bear market (Coppock, 1962,
        "Practical Relative Strength Charting", Trendex Research Group, first presented in
        Barron's). He took the 11 and 14 months from the time an Episcopal bishop told him
        mourning takes, as the time investors need to get over a loss.

        The formula is as follows:

        - Coppock Curve = WMA(ROC(Close, long_roc_window) + ROC(Close, short_roc_window), wma_window)
        - ROC(Close, n) = Close(t) / Close(t - n) — 1

        with WMA the linearly Weighted Moving Average, so the defaults give WMA10(ROC14 + ROC11).
        The windows count periods of the chosen `period`: the curve as Coppock defined it is
        period="monthly", the default "daily" applies the same windows to days, as the other
        technical indicators do. The classic signal is the curve turning up while below zero;
        Coppock did not use it for sell signals. The Rates of Change are decimals, so the curve
        is a weighted sum of decimal returns (0.30 where charting platforms show 30).

        For more information, see the following sources:

        - TA-Lib: COPPOCK, the reference implementation (https://ta-lib.org/functions/coppock.html).

        Also known as: Coppock Indicator, Coppock Guide.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            long_roc_window (int, optional): Number of periods of the longer Rate of Change.
                Defaults to 14.
            short_roc_window (int, optional): Number of periods of the shorter Rate of Change.
                Defaults to 11.
            wma_window (int, optional): Number of periods of the Weighted Moving Average of the
                summed Rates of Change. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Coppock Curve.
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
            pd.Series or pd.DataFrame: Coppock Curve values, weighted sums of Rates of Change as
            decimals.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Coppock Curve for each asset in the Toolkit instance.
        - The curve starts max(`long_roc_window`, `short_roc_window`) + `wma_window` - 1
          periods after the first price: 23 months with the defaults on monthly data, so start
          about two years before the first value you need.
        - The Coppock Curve is a decimal and equals TA-Lib's COPPOCK divided by 100, since TA-Lib's
          is in percent.
        - If `growth` is set to True, the method calculates the growth of the Coppock Curve
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_coppock_curve(period="monthly").tail(5)
        ```

        Which returns:

        | Date    |   ^GSPC |   Benchmark |
        |:--------|--------:|------------:|
        | 2025-08 |  0.3174 |      0.3475 |
        | 2025-09 |  0.3183 |      0.3481 |
        | 2025-10 |  0.3173 |      0.3469 |
        | 2025-11 |  0.3228 |      0.3521 |
        | 2025-12 |  0.3274 |      0.357  |
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

        coppock_curve = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_coppock_curve,
            long_roc_window,
            short_roc_window,
            wma_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=coppock_curve,
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
