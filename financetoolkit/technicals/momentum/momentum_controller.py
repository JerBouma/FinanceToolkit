"""Momentum Module"""

__docformat__ = "google"

from typing import Literal

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.technicals.breadth import breadth_model
from financetoolkit.technicals.cycle import cycle_model
from financetoolkit.technicals.helpers import (
    calculate_per_ticker,
    report_missing_values,
)
from financetoolkit.technicals.momentum import momentum_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
    validate_window,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


@report_missing_values
@filter_shown_columns
@track_sources
class MomentumTechnicals:
    """
    Momentum indicators: oscillators such as the RSI, ConnorsRSI, MACD (also volume weighted), stochastic, Williams %R, Aroon, ADX and the directional movement system, CCI, money flow index, Ichimoku Cloud, rate of change, momentum, Know Sure Thing, the Coppock curve, the indicators built on a linear regression (the Correlation Trend Indicator and the Forecast and Projection Oscillators), Pee's Trend Detection Index, Varadi's DV Intermediate Oscillator and trend flags such as the TTM Trend.
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
        Collect the momentum indicators of every ticker in one table: how fast and how strongly the prices move, and
        whether they are overbought or oversold.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            window (int, optional): The window size for calculating indicators, at least 2 as
                the Commodity Channel Index and the Choppiness Index need two periods.
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
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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

        | Date       |   Money Flow Index |   Williams %R |   Aroon Indicator Up |   Aroon Indicator Down |   Aroon Oscillator |   Commodity Channel Index |   Relative Vigor Index |   Relative Vigor Index Signal Line |       Force Index |   Ultimate Oscillator |   Percentage Price Oscillator |   Percentage Price Oscillator Signal Line |   Percentage Price Oscillator Histogram |   Absolute Price Oscillator |   Percentage Volume Oscillator |   Percentage Volume Oscillator Signal Line |   Percentage Volume Oscillator Histogram |   Detrended Price Oscillator |   Average Directional Index |   Plus Directional Movement |   Minus Directional Movement |   Plus Directional Indicator |   Minus Directional Indicator |   Directional Movement Index |   Average Directional Index Rating |   Chande Momentum Oscillator |   Intraday Momentum Index |   Ichimoku Conversion Line |   Ichimoku Base Line |   Ichimoku Leading Span A |   Ichimoku Leading Span B |   Stochastic %K |   Stochastic %D |   MACD Line |   MACD Signal Line |   MACD Histogram |   Relative Strength Index |   Stochastic RSI %K |   Stochastic RSI %D |   True Strength Index |   True Strength Index Signal Line |   Balance of Power |   Internal Bar Strength |   Qstick |   Awesome Oscillator |   Accelerator Oscillator |   Vortex Indicator VI+ |   Vortex Indicator VI- |   Elder Ray Bull Power |   Elder Ray Bear Power |   Rate of Change |   Momentum |   Choppiness Index |   Efficiency Ratio |   Vertical Horizontal Filter |   Know Sure Thing |   Know Sure Thing Signal Line |   Coppock Curve |   Fisher Transform |   Fisher Transform Trigger |   Inverse Fisher Transform RSI |   KDJ K |   KDJ D |   KDJ J |   Premier Stochastic Oscillator |   Stochastic Momentum Index |   Stochastic Momentum Index Signal Line |   SMI Ergodic Indicator |   SMI Ergodic Signal Line |   SMI Ergodic Oscillator |   Schaff Trend Cycle |   WaveTrend |   WaveTrend Signal Line |   Derivative Oscillator |   Swing Index |   Accumulative Swing Index |     QQE |   QQE RSI Moving Average |   Dynamic Momentum Index |   Price Zone Oscillator |   Connors RSI |   Inertia |    Bias |   BRAR AR |   BRAR BR |   Pretty Good Oscillator |   Psychological Line |   Volatility-Based Momentum |   AMAT Long Run |   AMAT Short Run |   TTM Trend |   Correlation Trend Indicator |   Forecast Oscillator |   Projection Oscillator |
        |:-----------|-------------------:|--------------:|---------------------:|-----------------------:|-------------------:|--------------------------:|-----------------------:|-----------------------------------:|------------------:|----------------------:|------------------------------:|------------------------------------------:|----------------------------------------:|----------------------------:|-------------------------------:|-------------------------------------------:|-----------------------------------------:|-----------------------------:|----------------------------:|----------------------------:|-----------------------------:|-----------------------------:|------------------------------:|-----------------------------:|-----------------------------------:|-----------------------------:|--------------------------:|---------------------------:|---------------------:|--------------------------:|--------------------------:|----------------:|----------------:|------------:|-------------------:|-----------------:|--------------------------:|--------------------:|--------------------:|----------------------:|----------------------------------:|-------------------:|------------------------:|---------:|---------------------:|-------------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------:|-----------:|-------------------:|-------------------:|-----------------------------:|------------------:|------------------------------:|----------------:|-------------------:|---------------------------:|-------------------------------:|--------:|--------:|--------:|--------------------------------:|----------------------------:|----------------------------------------:|------------------------:|--------------------------:|-------------------------:|---------------------:|------------:|------------------------:|------------------------:|--------------:|---------------------------:|--------:|-------------------------:|-------------------------:|------------------------:|--------------:|----------:|--------:|----------:|----------:|-------------------------:|---------------------:|----------------------------:|----------------:|-----------------:|------------:|------------------------------:|----------------------:|------------------------:|
        | 2025-12-17 |            39.7393 |      -98.8222 |              28.5714 |               100      |           -71.4286 |                 -139.905  |                -0.2262 |                            -0.0758 |      -3.27416e+07 |               44.5331 |                        0.0063 |                                    0.0116 |                                 -0.0053 |                      1.7262 |                        -0.0329 |                                    -0.0509 |                                   0.018  |                       0.0306 |                     24.6713 |                     12.185  |                      13.7577 |                      17.868  |                       20.1742 |                       6.0622 |                            27.6362 |                     -20.6209 |                   41.1264 |                    275.43  |              276.008 |                   265.041 |                   250.519 |          1.1778 |          8.6605 |      1.7262 |             3.1596 |          -1.4334 |                   45.1866 |              0      |              1.2037 |               12.3225 |                           19.7487 |            -0.7013 |                  0.0442 |  -0.4015 |               1.2653 |                  -1.836  |                 0.9211 |                 1.0533 |                -0.1328 |                -4.6371 |          -0.0206 |    -5.6902 |            49.8427 |             0.2062 |                       0.5182 |            0.3638 |                        0.5107 |          0.0029 |            -2.2385 |                    -1.8906 |                        -0.8927 | 15.8397 | 25.2454 | -2.9718 |                         -0.84   |                      9.1395 |                                 26.1148 |                 -6.2668 |                    4.4051 |                 -10.6719 |               1.3165 |     -2.5415 |                 12.0151 |                 -6.9465 |      -38.8013 |                    5877.92 | 57.4732 |                  50.8734 |                  13.8158 |                -14.5643 |       20.1105 |   51.7715 | -0.014  |  131.997  |  121.214  |                  -1.3452 |              33.3333 |                     -2.243  |               0 |                1 |          -1 |                       -0.8216 |               -0.013  |                  0.2466 |
        | 2025-12-18 |            34.4002 |      -75.8192 |              21.4286 |               100      |           -78.5714 |                 -162.428  |                -0.2832 |                            -0.1659 |      -2.59752e+07 |               50.9808 |                        0.0047 |                                    0.0102 |                                 -0.0055 |                      1.2731 |                        -0.0132 |                                    -0.0434 |                                   0.0302 |                      -0.3822 |                     24.4324 |                     11.3147 |                      17.4488 |                      16.1683 |                       24.9338 |                      21.3261 |                            28.0928 |                     -24.9064 |                   36.3175 |                    272.6   |              276.008 |                   265.041 |                   250.519 |         24.1808 |         14.0381 |      1.2731 |             2.7823 |          -1.5092 |                   45.9182 |              2.4336 |              2.0149 |                9.3441 |                           17.1476 |            -0.2126 |                  0.7844 |  -0.6157 |               0.0099 |                  -2.1487 |                 0.865  |                 1.0802 |                -2.0699 |                -8.7267 |          -0.0239 |    -6.6369 |            42.7375 |             0.2491 |                       0.5366 |            0.311  |                        0.4834 |         -0.014  |            -2.6249 |                    -2.2385 |                        -0.9368 | 23.7921 | 24.761  | 21.8543 |                         -0.8426 |                      3.2693 |                                 21.5457 |                -10.1477 |                   -0.4458 |                  -9.7019 |               0.6582 |    -12.4226 |                  2.2173 |                 -7.2145 |      -12.5266 |                    5865.39 | 55.8771 |                  49.2217 |                  15.5532 |                  0.5149 |       46.5574 |   50.526  | -0.0123 |  108.702  |  105.088  |                  -1.1882 |              33.3333 |                     -2.7539 |               0 |                1 |          -1 |                       -0.9297 |               -0.003  |                  0.5333 |
        | 2025-12-19 |            43.3299 |      -68.989  |              14.2857 |                92.8571 |           -78.5714 |                 -103.239  |                -0.2694 |                            -0.2349 |       5.93143e+06 |               51.5414 |                        0.0037 |                                    0.0089 |                                 -0.0052 |                      1.0212 |                         0.1453 |                                    -0.0056 |                                   0.1509 |                      -0.4186 |                     23.9078 |                     11.4732 |                      16.2024 |                      16.469  |                       23.2575 |                      17.0882 |                            28.6043 |                     -39.3401 |                   28.2097 |                    272.6   |              276.008 |                   265.041 |                   250.519 |         31.011  |         18.7898 |      1.0212 |             2.4301 |          -1.4089 |                   49.0174 |             12.7427 |              5.0588 |                7.6467 |                           14.7723 |             0.3234 |                  0.8021 |  -0.8698 |              -1.1188 |                  -2.2692 |                 0.903  |                 1.137  |                -0.8132 |                -5.4969 |          -0.0333 |    -9.3972 |            41.2482 |             0.3934 |                       0.5987 |            0.2645 |                        0.4504 |         -0.0269 |            -2.3555 |                    -2.6249 |                        -0.9292 | 32.8313 | 27.4511 | 43.5917 |                         -0.7977 |                     -0.9462 |                                 17.0473 |                 -9.7979 |                   -3.5632 |                  -6.2347 |               0.3291 |    -16.4302 |                 -6.2154 |                 -6.63   |       14.9273 |                    5880.32 | 55.8245 |                  49.1536 |                  57.0396 |                 13.6936 |       64.4303 |   49.4718 | -0.007  |  113.263  |  110.127  |                  -0.7701 |              41.6667 |                     -2.0674 |               0 |                1 |          -1 |                       -0.9192 |                0.0092 |                  0.7123 |
        | 2025-12-22 |            36.4312 |      -81.4487 |               7.1429 |                85.7143 |           -78.5714 |                 -104.733  |                -0.2367 |                            -0.2614 |      -7.97965e+06 |               45.7604 |                        0.0022 |                                    0.0076 |                                 -0.0054 |                      0.5976 |                         0.1018 |                                     0.0158 |                                   0.0859 |                       2.2593 |                     23.4207 |                     10.6537 |                      15.0451 |                      15.6562 |                       22.1097 |                      17.0882 |                            29.1456 |                     -64.5455 |                   17.6515 |                    272.6   |              276.008 |                   264.954 |                   250.519 |         18.5513 |         24.581  |      0.5976 |             2.0636 |          -1.466  |                   44.0574 |              0      |              5.0588 |                4.6682 |                           12.2463 |            -0.5608 |                  0.1365 |  -1.2314 |              -2.0902 |                  -2.1525 |                 0.8075 |                 1.2147 |                -0.8978 |                -4.2561 |          -0.0532 |   -15.1671 |            40.4149 |             0.6455 |                       0.5589 |            0.2055 |                        0.4111 |         -0.0418 |            -2.0419 |                    -2.3555 |                        -0.9498 | 32.0392 | 28.9805 | 38.1567 |                         -0.7692 |                     -4.8407 |                                 12.6697 |                -13.7773 |                   -6.9679 |                  -6.8094 |               0.1646 |    -20.9318 |                -13.0815 |                 -6.456  |      -45.6961 |                    5834.63 | 54.1419 |                  47.4549 |                  37.2576 |                 -1.2512 |       20.8244 |   48.1033 | -0.0165 |  110.181  |  102.622  |                  -1.0926 |              41.6667 |                     -1.9446 |               0 |                1 |          -1 |                       -0.9246 |                0.0011 |                  0.5069 |
        | 2025-12-23 |            37.2984 |      -69.5725 |               0      |                78.5714 |           -78.5714 |                  -97.9344 |                -0.205  |                            -0.2501 |      -1.44128e+06 |               49.1967 |                        0.0014 |                                    0.0063 |                                 -0.005  |                      0.3694 |                         0.0548 |                                     0.0236 |                                   0.0311 |                       2.351  |                     23.1944 |                      9.8927 |                      14.9172 |                      14.9624 |                       22.5619 |                      20.2521 |                            28.9735 |                     -51.4169 |                   23.8077 |                    272.6   |              276.008 |                   265.083 |                   250.519 |         30.4275 |         26.6632 |      0.3694 |             1.7248 |          -1.3553 |                   47.029  |             15.5282 |              9.4236 |                3.0048 |                            9.9359 |             0.517  |                  0.9524 |  -0.9773 |              -2.6623 |                  -1.7431 |                 0.7732 |                 1.2661 |                -1.9284 |                -4.8582 |          -0.0415 |   -11.7491 |            46.4518 |             0.5142 |                       0.4243 |            0.1522 |                        0.3663 |         -0.0501 |            -1.9612 |                    -2.0419 |                        -0.9406 | 35.0211 | 30.994  | 43.0753 |                         -0.727  |                     -6.7949 |                                  8.7768 |                -13.4496 |                   -9.1285 |                  -4.3211 |               0.0823 |    -24.6243 |                -18.6022 |                 -5.7342 |       12.4911 |                    5847.12 | 53.9827 |                  47.3129 |                  56.8024 |                 12.1426 |       57.2709 |   47.6333 | -0.0114 |  105.311  |  103.415  |                  -0.6584 |              50      |                     -1.3037 |               0 |                1 |          -1 |                       -0.9141 |                0.0076 |                  0.7178 |
        | 2025-12-24 |            41.6581 |      -51.6561 |               0      |                71.4286 |           -71.4286 |                  -34.6705 |                -0.1841 |                            -0.2228 |       2.20161e+06 |               54.4448 |                        0.0011 |                                    0.0053 |                                 -0.0042 |                      0.3017 |                        -0.0045 |                                     0.018  |                                  -0.0225 |                       3.0907 |                     22.018  |                     12.1059 |                      13.8517 |                      18.736  |                       21.4379 |                       6.7256 |                            28.2487 |                     -32.919  |                   31.7545 |                    272.6   |              276.008 |                   265.063 |                   250.519 |         48.3439 |         32.4409 |      0.3017 |             1.4402 |          -1.1385 |                   50.0121 |             38.5455 |             18.0246 |                2.4525 |                            8.0651 |             0.4551 |                  0.4985 |  -0.6307 |              -2.8188 |                  -1.0827 |                 0.8892 |                 1.1613 |                 1.0804 |                -2.1383 |          -0.0245 |    -6.8661 |            53.1324 |             0.3292 |                       0.3731 |            0.1033 |                        0.3168 |         -0.0518 |            -1.5593 |                    -1.9612 |                        -0.8894 | 40.6707 | 34.2196 | 53.573  |                         -0.6448 |                     -7.0164 |                                  5.6181 |                -10.3613 |                   -9.5394 |                  -0.8219 |               0.0411 |    -22.4462 |                -21.1081 |                 -4.3189 |       36.2295 |                    5883.35 | 53.9827 |                  48.2126 |                  42.5356 |                 23.8305 |       67.2632 |   47.9699 | -0.0071 |  112.863  |  120.363  |                  -0.2751 |              58.3333 |                     -0.8397 |               0 |                1 |           1 |                       -0.852  |                0.0123 |                  0.8331 |
        | 2025-12-26 |            46.9441 |      -51.1363 |               0      |                64.2857 |           -64.2857 |                  -25.6932 |                -0.1697 |                            -0.1974 |  735552           |               51.2159 |                        0.0008 |                                    0.0044 |                                 -0.0036 |                      0.2126 |                        -0.0488 |                                     0.0046 |                                  -0.0534 |                      -0.6819 |                     20.9257 |                     11.2412 |                      12.8623 |                      17.9861 |                       20.58   |                       6.7256 |                            27.2889 |                     -27.7033 |                   33.1185 |                    272.6   |              276.098 |                   264.961 |                   250.519 |         48.8637 |         42.545  |      0.2126 |             1.1946 |          -0.982  |                   49.1689 |             34.6217 |             29.5651 |                1.7619 |                            6.4893 |            -0.3028 |                  0.2151 |  -0.5595 |              -2.1658 |                   0.0054 |                 0.9091 |                 1.1011 |                 1.1553 |                -1.346  |          -0.0193 |    -5.3614 |            55.49   |             0.277  |                       0.4022 |            0.0564 |                        0.2631 |         -0.0522 |            -1.1484 |                    -1.5593 |                        -0.8224 | 43.4017 | 37.2803 | 55.6445 |                         -0.4625 |                     -6.9074 |                                  3.113  |                 -9.0825 |                   -9.3871 |                   0.3046 |               0.0206 |    -20.284  |                -22.0716 |                 -2.8918 |       -3.8045 |                    5879.54 | 53.9827 |                  48.5314 |                  33.7903 |                  7.3734 |       39.8837 |   48.2486 | -0.0094 |  120.453  |  119.909  |                  -0.284  |              58.3333 |                     -0.7927 |               1 |                0 |           1 |                       -0.7986 |                0.0095 |                  0.7973 |
        | 2025-12-29 |            48.1203 |      -48.4089 |              35.7143 |                57.1429 |           -21.4286 |                  -30.5671 |                -0.1563 |                            -0.1782 |       1.7719e+06  |               57.3149 |                        0.0006 |                                    0.0036 |                                 -0.003  |                      0.169  |                        -0.0817 |                                    -0.0126 |                                  -0.0691 |                       0.1103 |                     20.0593 |                     10.4382 |                      12.4518 |                      17.3861 |                       20.7398 |                       8.7965 |                            26.5057 |                     -21.8633 |                   36.4164 |                    270.612 |              276.183 |                   265.068 |                   250.519 |         51.5911 |         49.5995 |      0.169  |             0.9895 |          -0.8205 |                   49.9666 |             40.0245 |             37.7306 |                1.4003 |                            5.217  |             0.5323 |                  0.7015 |  -0.4662 |              -2.0654 |                   0.2951 |                 0.9316 |                 1.0955 |                 0.2129 |                -1.7901 |          -0.0149 |    -4.1157 |            54.4742 |             0.2186 |                       0.4134 |            0.0118 |                        0.2104 |         -0.0496 |            -0.8195 |                    -1.1484 |                        -0.6882 | 53.5816 | 42.714  | 75.3166 |                         -0.1913 |                     -6.4822 |                                  1.194  |                 -7.5042 |                   -8.7594 |                   1.2553 |               0.0103 |    -19.6328 |                -21.7468 |                 -1.5792 |        5.5886 |                    5885.13 | 53.9827 |                  49.0098 |                  34.5775 |                 19.7029 |       57.1631 |   49.2247 | -0.0088 |  110.988  |  114.995  |                  -0.1531 |              58.3333 |                     -1.0766 |               1 |                0 |           1 |                       -0.7244 |                0.0103 |                  0.854  |
        | 2025-12-30 |            43.4593 |      -53.5607 |              28.5714 |                50      |           -21.4286 |                  -35.5284 |                -0.1288 |                            -0.1608 | -464807           |               54.1042 |                        0.0003 |                                    0.003  |                                 -0.0027 |                      0.0788 |                        -0.1122 |                                    -0.0325 |                                  -0.0797 |                      -2.3582 |                     19.2762 |                      9.6926 |                      11.6321 |                      16.8441 |                       20.2146 |                       9.095  |                            25.712  |                     -21.7389 |                   38.6752 |                    270.612 |              276.183 |                   265.068 |                   250.519 |         46.4393 |         48.9647 |      0.0788 |             0.8074 |          -0.7285 |                   48.4209 |             29.5551 |             34.7338 |                0.6843 |                            4.0838 |             0.15   |                  0.4444 |  -0.3773 |              -1.9448 |                   0.3866 |                 0.9145 |                 1.1096 |                 0.0857 |                -1.708  |          -0.0148 |    -4.0858 |            53.5704 |             0.2174 |                       0.4141 |           -0.0182 |                        0.1612 |         -0.0465 |            -0.4022 |                    -0.8195 |                        -0.6065 | 57.9069 | 47.7783 | 78.1641 |                          0.0877 |                     -6.3714 |                                 -0.3191 |                 -7.7102 |                   -8.4097 |                   0.6995 |               0.0051 |    -20.2725 |                -20.6589 |                 -0.7486 |       -3.2998 |                    5881.83 | 53.9827 |                  48.8135 |                  50.5959 |                  3.8001 |       36.7617 |   50.4639 | -0.0122 |  113.905  |  105.452  |                  -0.2379 |              58.3333 |                     -1.0873 |               0 |                1 |           1 |                       -0.6718 |                0.0069 |                  0.7814 |
        | 2025-12-31 |            38.2003 |      -62.8029 |              21.4286 |                42.8571 |           -21.4286 |                  -56.9136 |                -0.0887 |                            -0.1381 |      -4.82705e+06 |               46.4079 |                       -0.0003 |                                    0.0023 |                                 -0.0026 |                     -0.0897 |                        -0.1258 |                                    -0.0512 |                                  -0.0746 |                      -1.5169 |                     18.7176 |                      9.0003 |                      11.3294 |                      16.2589 |                       20.4663 |                      11.4565 |                            24.6539 |                     -37.4456 |                   34.0262 |                    270.248 |              276.821 |                   265.6   |                   250.867 |         37.1971 |         45.0758 |     -0.0897 |             0.628  |          -0.7176 |                   45.6902 |             12.0094 |             27.1964 |               -0.6702 |                            2.8953 |            -0.6218 |                  0.057  |  -0.536  |              -1.6049 |                   0.515  |                 0.9113 |                 1.1148 |                -0.0091 |                -1.9324 |          -0.0248 |    -6.896  |            52.5944 |             0.3745 |                       0.3956 |           -0.0452 |                        0.1157 |         -0.0431 |            -0.1073 |                    -0.4022 |                        -0.6548 | 57.905  | 51.1539 | 71.4071 |                          0.1423 |                     -7.1079 |                                 -1.6769 |                -10.1732 |                   -8.9975 |                  -1.1757 |               0.0026 |    -23.361  |                -20.8876 |                 -0.6816 |      -19.3794 |                    5862.45 | 53.9827 |                  47.7724 |                  43.3308 |                 -9.9432 |       24.882  |   52.0466 | -0.0166 |   99.0847 |   92.3301 |                  -0.3998 |              50      |                     -1.4421 |               0 |                1 |          -1 |                       -0.6071 |                0.0023 |                  0.6255 |
        """
        self._check_arguments(period, close_column, window=window)
        validate_window("window", window, 2)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday" and self._historical_data[period].empty:
            raise InvalidArgumentError(
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
        percentage_price_oscillator = self.get_moving_average_convergence_divergence(
            period=period, close_column=close_column, method="percentage"
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
        # The MACD line, in the order of the tickers of the single-output indicators.
        momentum_indicators["Absolute Price Oscillator"] = (
            self.get_moving_average_convergence_divergence(
                period=period, close_column=close_column
            )["MACD Line"].reindex(
                columns=self._historical_data[period]["Close"].columns
            )
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
            self.get_average_directional_index(
                period=period, close_column=close_column, window=window, method="rating"
            )
        )
        momentum_indicators["Chande Momentum Oscillator"] = (
            self.get_chande_momentum_oscillator(
                period=period, close_column=close_column, window=window
            )
        )
        momentum_indicators["Intraday Momentum Index"] = (
            self.get_relative_strength_index(
                period=period,
                close_column=close_column,
                window=window,
                method="intraday",
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

        stochastic_rsi = self.get_stochastic_oscillator(
            period=period, close_column=close_column, window=window, method="rsi"
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
        momentum_indicators["Accelerator Oscillator"] = self.get_awesome_oscillator(
            period=period, close_column=close_column, method="accelerator"
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
        momentum_indicators["Momentum"] = self.get_rate_of_change(
            period=period, close_column=close_column, window=window, method="absolute"
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

        fisher_transform = self.get_fisher_transform(
            period=period, close_column=close_column
        )

        momentum_indicators["Fisher Transform"] = fisher_transform["Fisher Transform"]
        momentum_indicators["Fisher Transform Trigger"] = fisher_transform["Trigger"]
        momentum_indicators["Inverse Fisher Transform RSI"] = (
            self.get_inverse_fisher_transform_rsi(
                period=period, close_column=close_column
            )
        )

        kdj_indicator = self.get_kdj_indicator(period=period, close_column=close_column)

        momentum_indicators["KDJ K"] = kdj_indicator["K"]
        momentum_indicators["KDJ D"] = kdj_indicator["D"]
        momentum_indicators["KDJ J"] = kdj_indicator["J"]
        momentum_indicators["Premier Stochastic Oscillator"] = (
            self.get_premier_stochastic_oscillator(
                period=period, close_column=close_column
            )
        )

        stochastic_momentum_index = self.get_stochastic_momentum_index(
            period=period, close_column=close_column
        )

        momentum_indicators["Stochastic Momentum Index"] = stochastic_momentum_index[
            "Stochastic Momentum Index"
        ]
        momentum_indicators["Stochastic Momentum Index Signal Line"] = (
            stochastic_momentum_index["Signal Line"]
        )

        smi_ergodic_indicator = self.get_true_strength_index(
            period=period, close_column=close_column, method="ergodic"
        )

        momentum_indicators["SMI Ergodic Indicator"] = smi_ergodic_indicator[
            "SMI Ergodic Indicator"
        ]
        momentum_indicators["SMI Ergodic Signal Line"] = smi_ergodic_indicator[
            "Signal Line"
        ]
        momentum_indicators["SMI Ergodic Oscillator"] = smi_ergodic_indicator[
            "Histogram"
        ]
        momentum_indicators["Schaff Trend Cycle"] = self.get_schaff_trend_cycle(
            period=period, close_column=close_column
        )

        wavetrend_oscillator = self.get_wavetrend_oscillator(
            period=period, close_column=close_column
        )

        momentum_indicators["WaveTrend"] = wavetrend_oscillator["WaveTrend"]
        momentum_indicators["WaveTrend Signal Line"] = wavetrend_oscillator[
            "Signal Line"
        ]
        momentum_indicators["Derivative Oscillator"] = self.get_derivative_oscillator(
            period=period, close_column=close_column
        )

        swing_index = self.get_swing_index(period=period, close_column=close_column)

        momentum_indicators["Swing Index"] = swing_index["Swing Index"]
        momentum_indicators["Accumulative Swing Index"] = swing_index[
            "Accumulative Swing Index"
        ]

        quantitative_qualitative_estimation = (
            self.get_quantitative_qualitative_estimation(
                period=period, close_column=close_column
            )
        )

        momentum_indicators["QQE"] = quantitative_qualitative_estimation["QQE"]
        momentum_indicators["QQE RSI Moving Average"] = (
            quantitative_qualitative_estimation["RSI Moving Average"]
        )
        momentum_indicators["Dynamic Momentum Index"] = (
            self.get_relative_strength_index(
                period=period, close_column=close_column, method="dynamic"
            )
        )
        momentum_indicators["Price Zone Oscillator"] = self.get_price_zone_oscillator(
            period=period, close_column=close_column
        )
        momentum_indicators["Connors RSI"] = self.get_relative_strength_index(
            period=period, close_column=close_column, method="connors"
        )
        momentum_indicators["Inertia"] = self.get_inertia(
            period=period, close_column=close_column
        )
        momentum_indicators["Bias"] = self.get_bias(
            period=period, close_column=close_column
        )

        brar = self.get_brar(period=period, close_column=close_column)

        momentum_indicators["BRAR AR"] = brar["AR"]
        momentum_indicators["BRAR BR"] = brar["BR"]
        momentum_indicators["Pretty Good Oscillator"] = self.get_pretty_good_oscillator(
            period=period, close_column=close_column
        )
        momentum_indicators["Psychological Line"] = self.get_psychological_line(
            period=period, close_column=close_column
        )
        momentum_indicators["Volatility-Based Momentum"] = (
            self.get_volatility_based_momentum(period=period, close_column=close_column)
        )

        archer_moving_average_trends = self.get_archer_moving_average_trends(
            period=period, close_column=close_column
        )

        momentum_indicators["AMAT Long Run"] = archer_moving_average_trends["Long Run"]
        momentum_indicators["AMAT Short Run"] = archer_moving_average_trends[
            "Short Run"
        ]
        momentum_indicators["TTM Trend"] = self.get_ttm_trend(
            period=period, close_column=close_column
        )
        momentum_indicators["Correlation Trend Indicator"] = (
            self.get_correlation_trend_indicator(
                period=period, close_column=close_column, window=window
            )
        )
        momentum_indicators["Forecast Oscillator"] = self.get_forecast_oscillator(
            period=period, close_column=close_column, window=window
        )
        momentum_indicators["Projection Oscillator"] = self.get_projection_oscillator(
            period=period, close_column=close_column, window=window
        )

        trend_detection_index = self.get_trend_detection_index(
            period=period, close_column=close_column
        )

        momentum_indicators["Trend Detection Index"] = trend_detection_index["TDI"]
        momentum_indicators["TDI Direction Indicator"] = trend_detection_index[
            "Direction Indicator"
        ]
        momentum_indicators["DV Intermediate Oscillator"] = (
            self.get_dv_intermediate_oscillator(
                period=period, close_column=close_column
            )["DVI"]
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
        Calculate the Money Flow Index (MFI): how strongly money flows into or out of a security, using price and
        volume.

        The Money Flow Index is a momentum indicator that measures the strength and
        direction of money flowing in and out of a security by considering both price
        and volume.

        The formula is as follows:

        - MFI = 100 — (100 / (1 + (positive_money_flow / negative_money_flow)))

        For more information, see the following sources:

        - Quong, G. & Soudack, A. (1989). "Volume-Weighted RSI: Money Flow." Technical Analysis of Stocks &
          Commodities, 7(3), 76-77.

        Referenced against:
        - TA-Lib, MFI: https://ta-lib.org/functions/mfi.html

        Also known as: MFI, volume-weighted RSI.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The MFI agrees with TA-Lib's MFI.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Williams Percent R (Williams %R): where the close lies within the high-low range of a window,
        from 0 (at the high) to -100 (at the low), to show overbought and oversold levels.

        The Williams %R is a momentum indicator that measures the level of the close price
        relative to the high-low range over a certain number of periods.

        The formula is as follows:

        - Williams %R = (Highest High — Close) / (Highest High — Lowest Low) * —100

        For more information, see the following sources:

        - Williams, L.R. (1973). "How I Made One Million Dollars Last Year Trading Commodities." Windsor Books.

        Referenced against:
        - TA-Lib, WILLR: https://ta-lib.org/functions/willr.html

        Also known as: Williams percent R, overbought oversold oscillator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - Williams %R agrees with TA-Lib's WILLR.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Aroon Indicator: how many periods ago the highest high and the lowest low of a window were,
        which shows whether a trend is starting and how strong it is.

        The Aroon Indicator is an oscillator that measures the strength of a trend and the
        likelihood of its continuation or reversal.

        The formula is as follows:

        - Aroon Up = ((Number of periods) — (Number of periods since highest high)) / (Number of periods) * 100
        - Aroon Down = ((Number of periods) — (Number of periods since lowest low)) / (Number of periods) * 100
        - Aroon Oscillator = Aroon Up — Aroon Down

        The Aroon Oscillator runs from -100 to 100: above zero the latest high is more recent than
        the latest low, which reads as an uptrend, and a reading near 100 or -100 as a strong trend.

        For more information, see the following sources:

        - Chande, T.S. (1995). "The Time Price Oscillator." Technical Analysis of Stocks & Commodities, 13(9),
          369-374.

        Referenced against:
        - TA-Lib, AROON: https://ta-lib.org/functions/aroon.html
        - TA-Lib, AROONOSC: https://ta-lib.org/functions/aroonosc.html

        Also known as: Aroon Up, Aroon Down, Aroon Oscillator, AROONOSC, trend strength.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - Aroon Up and Aroon Down agree with TA-Lib's AROON, and the Aroon Oscillator TA-Lib's AROONOSC.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Commodity Channel Index (CCI): how far the typical price lies from its average, in units of
        its mean deviation, to show cyclical turns and overbought and oversold levels.

        The Commodity Channel Index is an oscillator that measures the current price level
        relative to an average price level over a specified period.

        The formula is as follows:

        - CCI = (Typical Price — SMA(Typical Price)) / (constant * Mean Deviation)

        For more information, see the following sources:

        - Lambert, D.R. (1980). "Commodity Channel Index: Tool for Trading Cyclic Trends." Commodities, October
          1980; reprinted in Technical Analysis of Stocks & Commodities, 1(5), 120-122.

        Referenced against:
        - TA-Lib, CCI: https://ta-lib.org/functions/cci.html

        Also known as: CCI, cyclical trend indicator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column in the historical data that represents
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for calculating the CCI, at least 2.
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
        - With the default `constant` of 0.015, the CCI agrees with TA-Lib's CCI.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        # The mean deviation of the typical price needs two periods.
        validate_window("window", window, 2)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Relative Vigor Index (RVI): how strongly the close ends above the open relative to the range,
        which shows the conviction behind a move.

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

        For more information, see the following sources:

        - Ehlers, J.F. (2002). "Relative Vigor Index." Technical Analysis of Stocks & Commodities, 20(1), 16-20.

        Also known as: RVI, RVGI, vigor index.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        window: int | None = None,
        method: Literal["standard", "cumulative"] = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Force Index or the Cumulative Force Index (CFI): the price change times the volume, smoothed
        (and added up over time for the CFI), which shows the force of the buyers or sellers behind a move.

        The Force Index of Elder measures the strength behind price movements: each period's
        change in price times its volume, smoothed with an exponential moving average. The
        Cumulative Force Index is its running total. Where On-Balance Volume adds the whole
        volume of an up close however small the move, the force weights the volume with the
        size of the move, so a heavy session that barely moves price adds little. With a window
        of 1 there is no smoothing and the Cumulative Force Index is the Weighted On-Balance
        Volume.

        The formula for the Force Index is as follows:

        - Raw Force Index = (Close — Close(1)) * Volume
        - Force Index = EMA(Raw Force Index, window)

        The formula for the Cumulative Force Index is as follows:

        - Force(t) = (Close(t) — Close(t-1)) * Volume(t)
        - CFI(t) = CFI(t-1) + EMA(Force, window)(t), starting on the start date

        As with On-Balance Volume, the level of the Cumulative Force Index depends on the start
        date and the number of shares traded; only the direction and the divergences from price
        carry meaning.

        For more information, see the following sources:

        - Elder, A. (1993). "Trading for a Living: Psychology, Trading Tactics, Money Management." John Wiley &
          Sons.

        Referenced against:
        - TA-Lib, EFI: https://ta-lib.org/functions/efi.html
        - finta, CFI (cumulative Force Index): https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1585
        - finta, WOBV (cumulative Force Index): https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1506

        Also known as: FI, Elder's Force Index, EFI, CFI, Cumulative Force Index, Weighted On-Balance
        Volume, WOBV (the Cumulative Force Index with a window of 1).

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int | None, optional): The number of periods of the EMA that smooths the raw
                Force Index. Defaults to None, which is 13 with method="standard", Elder's setting
                for the trend (2 is his setting for timing entries), and 10 with
                method="cumulative"; 1 gives the Weighted On-Balance Volume.
            method (str, optional): The index. Options are: "standard" (the Force Index), "cumulative" (its
                running total from the start date, the Cumulative Force Index). Defaults to "standard".
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
            pd.Series or pd.DataFrame: Force Index or Cumulative Force Index values, in price times
            volume.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Force Index values for each asset in the Toolkit instance; the Cumulative Force Index
          as a running total from the start date.
        - The Force Index agrees with TA-Lib's EFI.
        - The Cumulative Force Index with a window of 1 agrees with finta's WOBV. With a larger window
          every change agrees with that of finta's CFI once the EMAs have converged (after about
          200 periods), but the EMA starts from the simple average of its first `window` values
          while finta's averages from the first value, and the running total keeps the difference
          of the warm-up as a constant offset.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_option("method", method, ("standard", "cumulative"))
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        if method == "cumulative":
            # A running total starts at the start date, not at the earlier prices.
            force_index = calculate_per_ticker(
                self._historical_data[period].loc[self._start_date :],
                [close_column, "Volume"],
                breadth_model.get_cumulative_force_index,
                window if window is not None else 10,
            )
        else:
            force_index = calculate_per_ticker(
                self._historical_data[period],
                [close_column, "Volume"],
                momentum_model.get_force_index,
                window if window is not None else 13,
            )

        return finalize_dataset(
            dataset=force_index.loc[self._start_date : self._end_date],
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
        Calculate the Ultimate Oscillator: the buying pressure over three windows combined into one value from 0 to
        100, to show overbought and oversold levels with fewer false signals.

        The Ultimate Oscillator is a momentum oscillator that combines short-term, mid-term,
        and long-term price momentum into a single value.

        The formula is as follows:

        - Average(i) = Sum(Buying Pressure, window_i) / Sum(True Range, window_i)
        - Ultimate Oscillator = 100 * [(4 * Average_1) + (2 * Average_2) + Average_3] / 7

        For more information, see the following sources:

        - Williams, L.R. (1985). "The Ultimate Oscillator." Technical Analysis of Stocks & Commodities, 3(4), 140-141.

        Referenced against:
        - TA-Lib, ULTOSC: https://ta-lib.org/functions/ultosc.html

        Also known as: UO, ultimate momentum oscillator. See
        `momentum_model.get_ultimate_oscillator` for the Buying Pressure and True Range
        definitions.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The Ultimate Oscillator agrees with TA-Lib's ULTOSC. The first period counts with its High —
          Low range and its close minus its low, so the oscillator starts on period `window_3`, one
          period before TA-Lib's.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(
            period,
            close_column,
            window_1=window_1,
            window_2=window_2,
            window_3=window_3,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
    def get_percentage_volume_oscillator(
        self,
        period: str = "daily",
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
        Calculate the Percentage Volume Oscillator (PVO): the difference between a fast and a slow average of the
        volume as a share of the slow one, which shows whether volume is rising or falling.

        The Percentage Volume Oscillator is the Percentage Price Oscillator applied to volume: the
        difference between a short-term and a long-term EMA of volume as a share of the long-term
        EMA, with a signal line and a histogram as for the MACD.

        The formula is as follows:

        - PVO = (MA(Volume, short_window) — MA(Volume, long_window)) /
          MA(Volume, long_window)
        - Signal Line = EMA(PVO, signal_window)
        - Histogram = PVO — Signal Line

        with MA the Exponential Moving Average by default. With the Simple Moving Average
        it is the Volume Oscillator (VOSC) of StockCharts,
        pandas-ta-classic and Tulip Indicators, often used with 14 and 28 periods.

        A positive PVO means recent volume runs above its longer-term average, which gives weight to
        a breakout or a trend in price; a PVO below zero in a rising market says the move happens on
        thinning volume. Because it is a share of the average volume, the PVO can be compared across
        stocks whatever the number of shares they trade. It says nothing about the direction of
        price.

        For more information, see the following sources:

        - Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method." Signalert Corp., the
          MACD of which the PVO is the volume counterpart in percent.

        Referenced against:
        - TA-Lib, PVO: https://ta-lib.org/functions/pvo.html
        - pandas-ta-classic, vosc: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/vosc.py#L12
        - Tulip Indicators, vosc: https://tulipindicators.org/vosc

        Also known as: PVO, volume oscillator, VOSC. See `momentum_model.get_percentage_volume_oscillator`
        for the references.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            short_window (int, optional): The number of periods for the short-term moving average.
                Defaults to 12, as for the PPO and MACD.
            long_window (int, optional): The number of periods for the long-term moving average.
                Defaults to 26, as for the PPO and MACD.
            signal_window (int, optional): The number of periods for the signal line.
                Defaults to 9.
            moving_average (str, optional): The moving average of the volume. Options are:
                "exponential" (Exponential Moving Average), "simple" (Simple Moving Average, which gives the
                Volume Oscillator). The signal line is always an EMA. Defaults to "exponential".
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
          volume. It agrees with TA-Lib's PVO (with its default EMA) divided by 100: TA-Lib's PVO is in
          percent, returns 0 without volume and has no signal line or histogram. With the
          Simple Moving Average it agrees with TA-Lib's PVO with that moving average and the `vosc` of
          pandas-ta-classic and Tulip Indicators, divided by 100.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(
            period,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday" and self._historical_data[period].empty:
            raise InvalidArgumentError(
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
            moving_average,
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
        Calculate the Detrended Price Oscillator (DPO): the price minus a moving average shifted back in time, which
        removes the trend to show the cycles.

        The Detrended Price Oscillator (DPO) is an indicator that helps identify short-term cycles
        by removing longer-term trends from prices.

        The formula is as follows:

        - Displacement = floor(Number of Periods / 2) + 1
        - DPO = Close(t — Displacement) — SMA(Close, Number of Periods)(t)

        For more information, see the following sources:

        - Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York Institute of Finance.

        Referenced against:
        - TA-Lib, DPO: https://ta-lib.org/functions/dpo.html

        Also known as: DPO, detrended price oscillator. Note the moving average itself is not
        shifted — only the close price used for the comparison is looked up further back in
        time; see `momentum_model.get_detrended_price_oscillator` for the full explanation.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The DPO agrees with TA-Lib's DPO.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        method: Literal["standard", "rating"] = "standard",
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX) or its rating (ADXR): how strong a trend is, whatever its
        direction, from the smoothed difference between upward and downward movement.

        The Average Directional Index (ADX) is an indicator that measures the strength of a trend,
        whether it's an uptrend or a downtrend. The Average Directional Movement Index Rating
        (ADXR) is Wilder's smoothed ADX: the average of today's ADX and the ADX of
        window — 1 periods ago. Wilder rated markets with it on how strongly they trend, to choose
        the ones to trade with his trend-following systems; it moves more slowly than the ADX and
        lags it.

        The formula is as follows, per method:

        - ADX: ADX = Wilder's Smoothed Moving Average of DX, where DX = 100 * |+DI — -DI| /
          (+DI + -DI)
        - ADXR: ADXR = (ADX(t) + ADX(t — (window — 1))) / 2

        The ADX measures trend strength, not direction: a reading above about 25 is usually taken as a
        trending market and below 20 as a range, whether prices are rising or falling. Traders combine it
        with the +DI and -DI lines it is built from (`get_directional_movement_index`), or with the price
        itself, to know the direction. With Wilder's smoothing it reacts slowly and needs a few times the
        window of history before its values settle. The same readings apply to the ADXR.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.

        Referenced against:
        - TA-Lib, ADX: https://ta-lib.org/functions/adx.html
        - TA-Lib, ADXR: https://ta-lib.org/functions/adxr.html

        Also known as: ADX, trend strength indicator, ADXR, Average Directional Movement Rating,
        Average Directional Movement Index Rating.

        See `momentum_model.get_average_directional_index` for the full formula, which uses Wilder's
        smoothing (not a plain SMA) throughout.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column in the historical data that represents
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods to consider for the ADX calculation.
                Defaults to 14. With method="rating" also the lag of the rating.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): Options are: "standard" (the ADX), "rating" (the ADXR). Defaults to
                "standard".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series: Average Directional Index (ADX) or Average Directional
            Movement Index Rating (ADXR) values, between 0 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the ADX values for each asset in the Toolkit instance.
        - Wilder's smoothing is seeded with the average of the first `window` values, while TA-Lib's
          ADX seeds its running sums with `window` - 1 values, so the first values differ from
          TA-Lib's until the two converge (within 0.01% after about 120 periods with the default
          window).
        - ADXR: the ADXR starts 3 * `window` — 2 periods after the first price, as in TA-Lib,
          and its first values differ from TA-Lib's ADXR in the same way until the two converge
          (within 0.01% after about 100 to 125 periods with the default window).
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        methods = ("standard", "rating")
        check_option("method", method, methods)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        average_directional_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            (
                momentum_model.get_average_directional_index_rating
                if method == "rating"
                else momentum_model.get_average_directional_index
            ),
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
        Calculate Wilder's Directional Movement System: the smoothed Plus and Minus Directional Movement (+DM, -DM),
        the Directional Indicators (+DI, -DI) and the Directional Movement Index (DX).

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
        and the ADXR (the rating of `get_average_directional_index`) smooth it.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.

        Referenced against:
        - TA-Lib, PLUS_DM: https://ta-lib.org/functions/plus_dm.html
        - TA-Lib, MINUS_DM: https://ta-lib.org/functions/minus_dm.html
        - TA-Lib, PLUS_DI: https://ta-lib.org/functions/plus_di.html
        - TA-Lib, MINUS_DI: https://ta-lib.org/functions/minus_di.html
        - TA-Lib, DX: https://ta-lib.org/functions/dx.html

        Also known as: DMI, Directional Movement System, +DM, -DM, +DI, -DI, DX, PLUS_DM, MINUS_DM,
        PLUS_DI, MINUS_DI, directional indicators.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - When growth is requested, the method calculates the growth of the Directional Movement Index
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Chande Momentum Oscillator (CMO): the gains minus the losses over a window as a share of their
        sum, from -100 to 100, to show momentum and overbought and oversold levels.

        The Chande Momentum Oscillator is an indicator that measures the momentum of a price
        series and identifies overbought and oversold conditions.

        The formula is as follows:

        - CMO = 100 * ((Sum of Upward Change) — (Sum of Downward Change)) / ((Sum of Upward Change) +
          (Sum of Downward Change)), bounded between -100 and 100.
        - Smoothed CMO = 100 * (Wilder's MA(Upward Change) — Wilder's MA(Downward Change)) /
          (Wilder's MA(Upward Change) + Wilder's MA(Downward Change))

        Chande sums the changes over the window, so every change in the window counts equally and the
        CMO jumps when a large change leaves the window. The smoothed CMO averages the changes
        with Wilder's smoothing, as in the RSI, which gives TA-Lib's CMO: a smoother line that equals
        2 * RSI — 100.

        For more information, see the following sources:

        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

        Referenced against:
        - TA-Lib, CMOU: https://ta-lib.org/functions/cmou.html
        - TA-Lib, CMO: https://ta-lib.org/functions/cmo.html

        Also known as: CMO, CMOU, Chande momentum.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
          on StockCharts, which agrees with TA-Lib's CMOU. TA-Lib's CMO smooths them with Wilder's method
          instead, as in its RSI, so the two differ throughout (by up to about 55 points on daily data);
          the smoothed CMO gives TA-Lib's CMO, from the same period on.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Ichimoku Cloud: the midpoints of the highs and lows over three windows, which show the trend,
        the momentum and the support and resistance of the prices at a glance.

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

        - Hosoda, G. (1969). "Ichimoku Kinko Hyo." Published under the pen name Ichimoku Sanjin.
        - Elliott, N. (2007). "Ichimoku Charts." Harriman House.

        Referenced against:
        - TA-Lib, ICHIMOKU: https://ta-lib.org/functions/ichimoku.html

        Also known as: Ichimoku Kinko Hyo, cloud indicator. The default windows (9, 26, 52)
        are Goichi Hosoda's original values, both leading spans are conventionally displaced
        forward by the base (Kijun-sen) period.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The Conversion Line and Base Line agree with TA-Lib's ICHIMOKU, and so do the leading spans
          `base_window` periods earlier: TA-Lib places them on the period they are calculated on
          instead of displacing them forward, and starts every line on the `lead_span_b_window`-th
          period.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.
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
        self._check_arguments(
            period,
            close_column,
            conversion_window=conversion_window,
            base_window=base_window,
            lead_span_b_window=lead_span_b_window,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        method: Literal["standard", "rsi"] = "standard",
        stochastic_window: int = 14,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Stochastic Oscillator or the Stochastic RSI, from 0 to 100, which show momentum and overbought
        and oversold levels.

        It is where the close, or for the Stochastic RSI the Relative Strength Index, lies within its range over a
        window.

        The Stochastic Oscillator is a momentum indicator that shows the location of the close
        relative to the high-low range over a set number of periods. It consists of the %K line
        (fast) and the %D line (slow). The Stochastic RSI applies the Stochastic
        Oscillator to the Relative Strength Index instead of to price: it shows where the current
        RSI sits within its own high-low range over the last `stochastic_window` periods. Chande
        and Kroll (1994, "The New Technical Trader") designed it because the RSI can spend long
        stretches between 30 and 70 without reaching either extreme; the Stochastic RSI reaches 0
        and 100 far more often.

        The formula is as follows, per method:

        - Stochastic Oscillator: %K = 100 * ((Close — Lowest Low) / (Highest High — Lowest Low)),
          %D = SMA(%K, smooth_window), and with `slow` Slow %K = SMA(%K, smooth_window) and
          Slow %D = SMA(Slow %K, smooth_window)
        - Stochastic RSI: RSI = Relative Strength Index(Close, window), with Wilder's smoothing,
          %K = 100 * (RSI — Lowest RSI) / (Highest RSI — Lowest RSI) over the last
          stochastic_window RSI values, and %D = SMA(%K, smooth_window)

        The fast stochastic (the default) reacts to every close and gives many crossovers. The slow
        stochastic smooths %K once more, so its %K is the fast %D; it is the stochastic
        George Lane charted and the one most platforms show by default (14, 3, 3).

        The defaults of the Stochastic RSI follow Chande and Kroll's definition, a 14-period
        stochastic of a 14-period RSI, which StockCharts and TradingView use as well. StockCharts
        shows this %K on a scale of 0 to 1; TradingView smooths it over 3 periods, so its %K is the
        %D here. TA-Lib's STOCHRSI defaults to a 5-period stochastic instead (timeperiod 14,
        fastk_period 5 and fastd_period 3 with a simple average): a stochastic window of 5 gives its
        %K and %D. A %K of 100 means the RSI is at its highest of the window, not that the price is
        overbought, so the readings above 80 and below 20 come and go quickly and are mostly read
        together with the trend.

        For more information, see the following sources:

        - Lane, G.C. (1984). "Lane's Stochastics." Technical Analysis of Stocks & Commodities, 2(3).
        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

        Referenced against:
        - TA-Lib, STOCHF: https://ta-lib.org/functions/stochf.html
        - TA-Lib, STOCH: https://ta-lib.org/functions/stoch.html
        - TA-Lib, STOCHRSI: https://ta-lib.org/functions/stochrsi.html

        Also known as: stochastic oscillator, percent K, percent D, fast stochastic, slow
        stochastic, Stochastic RSI, StochRSI, Stoch RSI.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods to consider for the %K line calculation,
                and with method="rsi" the number of periods of the RSI. Defaults to 14.
            smooth_window (int, optional): The number of periods used to smooth the %K line
                into the %D signal line, and with `slow` the raw %K into the slow %K. Defaults to 3.
            slow (bool, optional): Whether to return the slow stochastic instead of the fast one.
                Only used when method="standard". Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the %K and %D values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): The stochastic. Options are: "standard" (of the price), "rsi" (the
                Stochastic RSI). Defaults to "standard".
            stochastic_window (int, optional): The number of periods of RSI values whose high-low
                range gives %K. Only used when method="rsi". Defaults to 14.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame: Stochastic Oscillator (%K and %D) values, between 0 and
            100, as "Stochastic %K" and "Stochastic %D", or "Stochastic RSI %K" and
            "Stochastic RSI %D" for the Stochastic RSI.

        Raises:
            ValueError: If the specified `period` or `method` is not one of the valid options.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Stochastic Oscillator values for each asset in the Toolkit instance.
        - %K and %D agree with TA-Lib's STOCHF (the fast stochastic) with `fastd_period` set to
          `smooth_window` and a simple moving average, which TA-Lib's STOCH gives
          with a `slowk_period` of 1. TA-Lib starts %K on the first %D.
        - The slow stochastic's %K and %D agree with TA-Lib's STOCH with `slowk_period` and `slowd_period` set to
          `smooth_window` and simple moving averages, and
          pandas-ta-classic's stoch. TA-Lib starts %K on the first %D.
        - Stochastic RSI: the RSI is seeded with a simple average of its first `window` changes, as Wilder
          and TA-Lib do, so %K starts `window` + `stochastic_window` - 1 periods after the first
          price and %D `smooth_window` - 1 periods later. TA-Lib starts both on the first %D,
          with the same values from there. An RSI that did not move over the stochastic window
          has no range, which leaves %K missing (TA-Lib sets it to 0). With `fastk_period` set to
          `stochastic_window`, `fastd_period` to `smooth_window` and a simple moving average,
          %K and %D agree with TA-Lib's STOCHRSI.
        - When growth is requested, the method calculates the growth of the %K and %D values
          over the given lag.

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
        self._check_arguments(
            period,
            close_column,
            window=window,
            smooth_window=smooth_window,
            stochastic_window=stochastic_window,
        )
        methods = ("standard", "rsi")
        check_option("method", method, methods)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        if method == "rsi":
            stochastic_oscillator = calculate_per_ticker(
                self._historical_data[period],
                [close_column],
                momentum_model.get_stochastic_relative_strength_index,
                window,
                stochastic_window,
                smooth_window,
            )
        else:
            stochastic_oscillator = calculate_per_ticker(
                self._get_historical_data(period, close_column),
                ["High", "Low", close_column],
                momentum_model.get_stochastic_oscillator,
                window,
                smooth_window,
                slow,
            )
        stochastic_oscillator = stochastic_oscillator.loc[
            self._start_date : self._end_date
        ]

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
        method: Literal["absolute", "percentage"] = "absolute",
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Moving Average Convergence Divergence (MACD) or the Percentage Price Oscillator, which show
        the direction and momentum of a trend.

        It is the difference between a fast and a slow moving average, in price units or, for the Percentage Price
        Oscillator, as a percentage of the slow average, with its signal line.

        The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator
        that shows the relationship between two moving averages of a security's price. It consists
        of the MACD line and its signal line; their difference is the MACD histogram. The MACD line
        is the Absolute Price Oscillator (APO), the difference between a short-term and a long-term
        moving average of the price, in price units: with exponential moving averages it is
        Appel's MACD line, and `moving_average` lets the average be chosen, such as simple moving
        averages for a slower oscillator. The Percentage Price Oscillator
        measures the difference between the two moving averages as a percentage of the longer one.

        The formula is as follows, per method:

        - MACD: MACD Line (APO) = MA(Close, short_window) — MA(Close, long_window), with the
          EMA by default, Signal Line = EMA(MACD Line, signal_window) and Histogram = MACD Line —
          Signal Line
        - Percentage Price Oscillator: PPO = (MA(Close, short_window) — MA(Close, long_window)) /
          MA(Close, long_window), with the EMA by default, Signal Line = EMA(PPO, signal_window)
          and Histogram = PPO — Signal Line

        The MACD line crossing its signal line is the usual trigger, and the MACD crossing zero means
        the short average has crossed the long one. The histogram is the distance between the two lines:
        it crosses zero with every crossover and shrinks before one, which is why it is read as an early
        warning. Because the MACD is a price difference, its level scales with the share price and cannot
        be compared across stocks; the Percentage Price Oscillator divides by the long average for that.
        Both lines lag price, so crossovers come late in fast reversals. The PPO is the MACD line as
        a percentage of the long average, so unlike the MACD its level can be compared across stocks
        and over long periods; its signal line is always an EMA.

        `moving_average` and `signal_moving_average` replace the EMAs of the MACD line and of the signal
        line with another moving average (TA-Lib's MACDEXT), and the fixed MACD smooths the 12- and
        26-period EMAs with the fixed constants 0.15 and 0.075 instead of 2 / (window + 1) (TA-Lib's
        MACDFIX). With the Volume Weighted Moving Average it is Dormeier's Volume Weighted MACD: the
        Volume Weighted Moving Averages, Sum(Close × Volume) / Sum(Volume), let the periods with heavy
        volume count more, so a move on heavy volume moves the MACD more than one on light volume.
        The Elastic Volume Weighted MACD uses Fries' Elastic Volume Weighted Moving Average,
        which moves towards the price by the share of the window's volume traded in a period (the
        20 and 40 periods in finta).

        For more information, see the following sources:

        - Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method." Signalert Corp.
        - Aspray, T. (1988). "MACD Momentum." Technical Analysis of Stocks & Commodities, 6(8), 294-297, for
          the MACD histogram.
        - Achelis, S.B. (2000). "Technical Analysis from A to Z." 2nd ed. McGraw-Hill.
        - Appel, G. (2005). "Technical Analysis: Power Tools for Active Investors." FT Press.
        - Dormeier, B.P. (2011). "Investing with Volume Analysis." FT Press.
        - Fries, C.P. (2001). "Elastic Moving Averages." Technical Analysis of Stocks & Commodities, 19(6), 52-56
          (https://www.christian-fries.de/evwma/).

        Referenced against:
        - TA-Lib, MACD: https://ta-lib.org/functions/macd.html
        - TA-Lib, MACDEXT: https://ta-lib.org/functions/macdext.html
        - TA-Lib, MACDFIX: https://ta-lib.org/functions/macdfix.html
        - pandas-ta-classic, vwmacd: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/vwmacd.py#L12
        - TA-Lib, APO: https://ta-lib.org/functions/apo.html
        - TA-Lib, PPO: https://ta-lib.org/functions/ppo.html

        Shares similar logic:
        - finta, VW_MACD: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L575
        - finta, EV_MACD: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L617
        - pandas-ta-classic, ppo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/ppo.py#L48
        - ta, PercentagePriceOscillator: https://github.com/bukosabino/ta/blob/0.11.0/ta/momentum.py#L622
        - finta, PPO: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L541

        Also known as: MACD, MACD histogram, MACDEXT, MACDFIX, VWMACD, volume weighted MACD, EVMACD,
        elastic volume weighted MACD, momentum indicator, APO, absolute price oscillator, price
        oscillator, PPO, percentage price oscillator, PPO signal line, PPO histogram.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): The number of periods for the shorter moving average.
                Defaults to 12.
            long_window (int, optional): The number of periods for the longer moving average.
                Defaults to 26.
            signal_window (int, optional): The number of periods for the signal line.
                Defaults to 9.
            moving_average (str, optional): The moving average of the short and long term. Options are:
                "simple" (Simple Moving Average), "exponential" (Exponential Moving Average),
                "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular Moving
                Average), "weighted" (Weighted Moving Average), "hull" (Hull Moving Average),
                "kaufman_adaptive" (Kaufman Adaptive Moving Average), "volume_weighted" (Volume Weighted Moving
                Average, the Volume Weighted MACD), "elastic_volume_weighted" (Elastic Volume Weighted Moving
                Average, the Elastic Volume Weighted MACD). Defaults to "exponential".
            signal_moving_average (str, optional): The moving average of the signal line. Options are:
                "simple" (Simple Moving Average), "exponential" (Exponential Moving Average),
                "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular Moving
                Average), "weighted" (Weighted Moving Average), "hull" (Hull Moving Average),
                "kaufman_adaptive" (Kaufman Adaptive Moving Average), "volume_weighted" (Volume Weighted Moving
                Average, the Volume Weighted MACD), "elastic_volume_weighted" (Elastic Volume Weighted Moving
                Average, the Elastic Volume Weighted MACD). Only used when method="absolute"; the signal line of
                the PPO is an EMA. Defaults to "exponential".
            fixed (bool, optional): Whether to smooth the 12- and 26-period EMAs with the fixed
                constants 0.15 and 0.075 (TA-Lib's MACDFIX). Needs the exponential moving average and
                the default windows. Only used when method="absolute". Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the MACD and signal values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): Options are: "absolute" (the MACD in price units), "percentage" (the
                Percentage Price Oscillator, a decimal of the long average). The volume weighted moving averages
                and `fixed` apply to the MACD only. Defaults to "absolute".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame:
            MACD line, signal line and histogram values in price units ("MACD Line", "Signal Line"
            and "Histogram"), or for the Percentage Price Oscillator the "Percentage Price Oscillator",
            "Signal Line" and "Histogram" as decimals (0.02 is 2%).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the MACD and signal line values for each asset in the Toolkit instance.
        - TA-Lib's MACD starts the short EMA on the same period as the long EMA and shows the MACD
          line from the first signal value, so the first values differ from TA-Lib's until the two
          converge (within 0.01% after about 85 periods with the default windows). The histogram
          agrees with TA-Lib's `macdhist` and ta's `macd_diff` from then on.
        - TA-Lib's MACDEXT and MACDFIX start their averages in the same way. With simple, weighted or
          triangular moving averages, which have no recursion, the lines equal MACDEXT from the
          first value; with exponential ones (also MACDFIX) they converge within 0.01% after about 90
          periods, with double exponential ones after about 130 and with Kaufman adaptive ones, whose
          smoothing can be very slow, after about 210.
        - With the Volume Weighted Moving Average for both the MACD line and the signal line the
          lines agree with pandas-ta-classic's vwmacd. finta's VW_MACD divides exponential averages of
          Close × Volume and of the volume instead. finta's EV_MACD starts every eVWMA from 0 where the
          Toolkit seeds it with a simple average, so the two converge as the start decays (after
          several hundred periods for a 40-period eVWMA).
        - The MACD line agrees with TA-Lib's APO with the same moving average (its moving average type:
          SMA 0, EMA 1, WMA 2, DEMA 3, TRIMA 5, KAMA 6) from the first value.
        - Percentage Price Oscillator: the PPO is a decimal and agrees with TA-Lib's PPO with the same moving average
          divided by 100, since TA-Lib's PPO is in percent. TA-Lib has no signal line or
          histogram; with the default EMAs all three agree with pandas-ta-classic's ppo with
          its exponential moving average, divided by 100. ta and finta seed their EMAs with the first price instead of
          a simple average, so their first values differ until the two converge. Every EMA is seeded
          with the simple average of its first values, so the PPO starts on the `long_window`-th
          period and the signal line and histogram `signal_window` - 1 periods later.
        - When growth is requested, the method calculates the growth of the MACD and signal values
          over the given lag.

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
        self._check_arguments(
            period,
            close_column,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
        )
        methods = ("absolute", "percentage")
        check_option("method", method, methods)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        if method == "percentage":
            macd = calculate_per_ticker(
                historical_data,
                [close_column],
                momentum_model.get_percentage_price_oscillator,
                short_window,
                long_window,
                signal_window,
                moving_average,
            )
        elif {moving_average, signal_moving_average} & set(
            momentum_model.VOLUME_MOVING_AVERAGES
        ):
            # The volume weighted averages take each ticker's volume next to its prices.
            macd = calculate_per_ticker(
                historical_data,
                [close_column, "Volume"],
                lambda prices, volumes: (
                    momentum_model.get_moving_average_convergence_divergence(
                        prices,
                        short_window,
                        long_window,
                        signal_window,
                        moving_average,
                        signal_moving_average,
                        fixed,
                        volumes=volumes,
                    )
                ),
            )
        else:
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
            )
        macd = macd.loc[self._start_date : self._end_date]

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
        window: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        method: Literal[
            "standard", "connors", "laguerre", "dynamic", "intraday"
        ] = "standard",
        streak_window: int = 2,
        rank_window: int = 100,
        gamma: float = 0.5,
        volatility_window: int = 5,
        average_window: int = 10,
        minimum_window: int = 5,
        maximum_window: int = 30,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI): the size of the gains against the losses over a window, from 0
        to 100, to show momentum and overbought and oversold levels.

        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and
        change of price movements. It ranges from 0 to 100 and is used to identify overbought or
        oversold conditions in an asset's price. The variants share that scale:

        - RSI: Wilder's RSI of the close.
        - ConnorsRSI: the ConnorsRSI of Connors, Alvarez and Radtke averages three short-term
          measures of how overbought or oversold a price is, each on a scale of 0 to 100: a
          3-period RSI of the close, a 2-period RSI of the streak (the number of periods in a row
          the close rose, positive, or fell, negative) and the percent rank of the latest
          one-period return among the returns of the previous 100 periods. It was designed for
          short-term mean reversion: below 10 (or 5) is oversold and a pullback in an uptrend to
          buy, above 90 (or 95) overbought.
        - Laguerre RSI: the Laguerre RSI of Ehlers (2004) is an RSI over the four elements of a
          Laguerre filter instead of four prices. The first element is an exponential moving
          average and each next one an all-pass element that delays the low frequencies more
          than the high ones. That warps time: four elements look as far back as a much longer
          window, so the RSI is smooth yet reacts quickly. Ehlers buys when it crosses back above
          20 and sells when it crosses back below 80.
        - Dynamic Momentum Index: the Dynamic Momentum Index of Chande and Kroll (1994) is an RSI whose period
          changes with the volatility: it shortens when the recent volatility is above its
          average and lengthens when it is below, between 5 and 30 periods. The index thereby
          reacts quickly in volatile markets and stays calm in quiet ones.
        - Intraday Momentum Index: the Intraday Momentum Index is an RSI of the moves within each period: it
          compares the bodies of the periods that closed above their open with all bodies over
          the window. Tushar Chande designed it for traders who care about the move from open to
          close rather than from close to close.

        The formula is as follows, per method:

        - RSI: RSI = 100 — (100 / (1 + RS)), with RS the ratio of Wilder's averages of the
          gains and losses over the window
        - ConnorsRSI: Streak = the number of consecutive periods the close rose (positive) or fell
          (negative), 0 when it was unchanged, Return = Close(t) / Close(t — 1) — 1, and
          ConnorsRSI = (RSI(Close, window) + RSI(Streak, streak_window) + 100 × Percent
          Rank(Return, rank_window)) / 3, with Wilder's RSI and the Percent Rank the share of the
          previous `rank_window` returns below the latest one (`get_percent_rank`)
        - Laguerre RSI: L0 = (1 - Gamma) * Price + Gamma * L0(t-1),
          L1 = -Gamma * L0 + L0(t-1) + Gamma * L1(t-1), L2 = -Gamma * L1 + L1(t-1) + Gamma *
          L2(t-1), L3 = -Gamma * L2 + L2(t-1) + Gamma * L3(t-1), CU = the sum of L0 - L1, L1 - L2
          and L2 - L3 where positive, CD = the sum of L1 - L0, L2 - L1 and L3 - L2 where positive,
          and Laguerre RSI = 100 * CU / (CU + CD)
        - Dynamic Momentum Index: Volatility Index = StdDev(Close, volatility_window) /
          SMA(StdDev(Close, volatility_window), average_window), Period = Int(window /
          Volatility Index) bounded to minimum_window and maximum_window, and DMI = 100 *
          Sum(Up Changes, Period) / (Sum(Up Changes, Period) + Sum(Down Changes, Period))
        - Intraday Momentum Index: Gain = Close — Open where Close > Open, else 0, Loss = Open — Close where
          Close < Open, else 0, and IMI = 100 * Sum(Gain, window) / (Sum(Gain, window) +
          Sum(Loss, window))

        Readings above 70 are conventionally read as overbought and below 30 as oversold, but in a
        strong trend the RSI can stay above 70 or below 30 for weeks, so a crossing is not a signal on
        its own. Practitioners look more at divergences, a new price high with a lower RSI high, and at
        the 50 line as a trend filter. The value depends on the window: a shorter window swings more and
        reaches the extremes more often. Chande's Dynamic Momentum Index uses a 14-period base, a
        5-period standard deviation and its 10-period average.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - Connors, L., Alvarez, C. & Radtke, M. (2012). "An Introduction to ConnorsRSI." Connors
          Research Trading Strategy Series.
        - Connors, L. & Alvarez, C. (2009). "Short Term Trading Strategies That Work."
          TradingMarkets Publishing.
        - Ehlers, J.F. (2004). "Cybernetic Analysis for Stocks and Futures." John Wiley & Sons,
          chapter 14, "Time Warp - Without Space Travel."
        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

        Referenced against:
        - TA-Lib, RSI: https://ta-lib.org/functions/rsi.html
        - TA-Lib, CRSI (ConnorsRSI): https://ta-lib.org/functions/crsi.html
        - pandas-ta-classic, lrsi (Laguerre RSI): https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/lrsi.py#L29
        - TA-Lib, IMI (Intraday Momentum Index): https://ta-lib.org/functions/imi.html

        Shares similar logic:
        - finta, DYMI (Dynamic Momentum Index): https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L752

        Also known as: RSI, momentum oscillator, overbought, oversold, CRSI, ConnorsRSI, Connors
        RSI, Laguerre RSI, LRSI, Ehlers' Laguerre RSI, DYMI, Dynamic Momentum Index, Chande's
        Dynamic Momentum Index, variable-term RSI, IMI, Intraday Momentum Index, intraday momentum.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open of method="intraday" is adjusted
                for dividends in the same way.
            window (int | None, optional): The number of periods for RSI calculation: of the RSI
                of the close ("connors"), the base period that the volatility index divides
                ("dynamic") and the number of periods to sum the gains and losses over
                ("intraday"). Not used by "laguerre". Defaults to None, which uses 3 for
                "connors" and 14 for the others.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the RSI.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): The RSI. Options are: "standard" (Wilder's RSI), "connors" (ConnorsRSI),
                "laguerre" (Ehlers' Laguerre RSI), "dynamic" (Chande's Dynamic Momentum Index),
                "intraday" (Chande's Intraday Momentum Index). Defaults to "standard".
            streak_window (int, optional): The number of periods of the RSI of the streak. Only
                used when method="connors". Defaults to 2.
            rank_window (int, optional): The number of previous returns of the percent rank. Only
                used when method="connors". Defaults to 100.
            gamma (float, optional): The damping factor of the Laguerre filter, at least 0 and
                below 1: a higher value smooths more and lags more. Only used when
                method="laguerre". Defaults to 0.5.
            volatility_window (int, optional): The number of periods of the standard deviation.
                Only used when method="dynamic". Defaults to 5.
            average_window (int, optional): The number of periods of the average of the standard
                deviation. Only used when method="dynamic". Defaults to 10.
            minimum_window (int, optional): The shortest period of the RSI. Only used when
                method="dynamic". Defaults to 5.
            maximum_window (int, optional): The longest period of the RSI. Only used when
                method="dynamic". Defaults to 30.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame or pd.Series:
            Relative Strength Index (RSI) values, between 0 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          RSI for each asset in the Toolkit instance.
        - RSI: the RSI agrees with TA-Lib's RSI.
        - ConnorsRSI: equals TA-Lib's CRSI. It starts `rank_window` + 1 periods after the first
          price, the first percent rank of a return.
        - Laguerre RSI: on the same scale of 0 to 100 as the other variants, as pandas-ta-classic's
          lrsi, where Ehlers' listing and the model function
          `cycle_model.get_laguerre_relative_strength_index` give a decimal between 0 and 1;
          otherwise they are equal. The filter elements start on the first price of each
          ticker's own history, as pandas-ta-classic starts them, so the first period is NaN;
          Ehlers' listing starts them from zero. When the four elements are equal (CU + CD is
          zero) the value before is kept, as in Ehlers' listing, where pandas-ta-classic gives 0.
        - Dynamic Momentum Index: the period changes every period, which leaves no single period for Wilder's
          recursive average, so each period's RSI sums the gains and losses of its own last
          `Period` changes (Cutler's RSI), the summed form Chande also uses for his Momentum
          Oscillator (`get_chande_momentum_oscillator`). The standard deviation is the sample
          standard deviation (ddof=1). finta's DYMI rounds the Volatility Index before the
          division and takes an exponentially smoothed RSI over the periods before the current
          one, so it differs from this by up to 50 points.
        - Intraday Momentum Index: the IMI agrees with TA-Lib's IMI, except on a window in which every period
          closed at its open, which has no IMI (NaN) where TA-Lib returns 50.
        - When growth is requested, the method calculates the growth of the RSI
          over the given lag.

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
        self._check_arguments(
            period,
            close_column,
            window=window,
            streak_window=streak_window,
            rank_window=rank_window,
            volatility_window=volatility_window,
            average_window=average_window,
            minimum_window=minimum_window,
            maximum_window=maximum_window,
        )
        methods = ("standard", "connors", "laguerre", "dynamic", "intraday")
        check_option("method", method, methods)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        if window is None:
            window = 3 if method == "connors" else 14

        # The prices each variant takes, its calculation and its parameters.
        columns, function, arguments = {
            "standard": (
                [close_column],
                momentum_model.get_relative_strength_index,
                (window,),
            ),
            "connors": (
                [close_column],
                momentum_model.get_connors_relative_strength_index,
                (window, streak_window, rank_window),
            ),
            "laguerre": (
                [close_column],
                cycle_model.get_laguerre_relative_strength_index,
                (gamma,),
            ),
            "dynamic": (
                [close_column],
                momentum_model.get_dynamic_momentum_index,
                (
                    window,
                    volatility_window,
                    average_window,
                    minimum_window,
                    maximum_window,
                ),
            ),
            "intraday": (
                ["Open", close_column],
                momentum_model.get_intraday_momentum_index,
                (window,),
            ),
        }[method]

        # The open of the Intraday Momentum Index is adjusted for dividends with the close.
        historical_data = (
            self._get_historical_data(period, close_column)
            if method in ("intraday", "laguerre")
            else self._historical_data[period]
        )

        relative_strength_index = calculate_per_ticker(
            historical_data, columns, function, *arguments
        ).loc[self._start_date : self._end_date]

        # Ehlers' Laguerre RSI is a decimal, shown on the 0 to 100 scale of the others.
        if method == "laguerre":
            relative_strength_index = relative_strength_index * 100

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
    def get_true_strength_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        long_window: int | None = None,
        short_window: int | None = None,
        signal_window: int | None = None,
        method: Literal["standard", "ergodic"] = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the True Strength Index (TSI) or the SMI Ergodic Indicator and Oscillator, which show the
        direction and strength of momentum.

        Both are the double-smoothed price change as a share of the double-smoothed absolute price change.

        The True Strength Index of Blau (1995, "Momentum, Direction, and Divergence") smooths
        the one-period price change twice with Exponential Moving Averages and divides it by
        the same double smoothing of the absolute change. The double smoothing filters out
        noise while lagging less than a single long average, and the division bounds the
        result between -100 (every change down) and +100 (every change up).

        The SMI Ergodic Indicator, also of Blau (1995), is the True Strength Index with short
        smoothing periods, and its signal line an EMA of the indicator. The SMI Ergodic
        Oscillator, the histogram, is the difference between the two, in the way the MACD
        histogram is the MACD line minus its signal line.

        The formula for the True Strength Index is as follows:

        - Momentum = Close(t) — Close(t - 1)
        - TSI = 100 * EMA(EMA(Momentum, long_window), short_window) / EMA(EMA(|Momentum|, long_window), short_window)
        - Signal Line = EMA(TSI, signal_window)

        The formula for the SMI Ergodic Indicator is as follows:

        - Momentum = Close(t) — Close(t - 1)
        - Smoothed Momentum = EMA(EMA(Momentum, long_window), short_window)
        - Smoothed Absolute Momentum = EMA(EMA(|Momentum|, long_window), short_window)
        - SMI Ergodic Indicator = 100 * Smoothed Momentum / Smoothed Absolute Momentum
        - Signal Line = EMA(SMI Ergodic Indicator, signal_window)
        - Histogram (SMI Ergodic Oscillator) = SMI Ergodic Indicator — Signal Line

        Blau smooths the TSI over 25 and then 13 periods; the 7-period signal line is the one
        StockCharts uses. The common parameters of the SMI Ergodic Indicator are 20, 5 and 5.
        The TSI crossing zero means the smoothed momentum has turned, crossing its signal
        line is the faster trigger, and readings beyond about +25 or -25 are read as overbought
        or oversold. Blau trades the direction of the SMI Ergodic Oscillator: a turn up from
        below zero is a buy signal, a turn down from above zero a sell signal. Like the RSI both
        are on a scale of -100 to 100 rather than a decimal.

        For more information, see the following sources:

        - Blau, W. (1995). "Momentum, Direction, and Divergence." Wiley.

        Referenced against:
        - TA-Lib, TSI: https://ta-lib.org/functions/tsi.html
        - pandas-ta-classic, smi (SMI Ergodic Indicator): https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/smi.py#L13

        Also known as: TSI, double smoothed momentum, SMI Ergodic, SMIIO, SMI Ergodic Oscillator,
        ergodic oscillator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            long_window (int | None, optional): The number of periods of the first EMA of the price
                change. Defaults to None, which is 25 with method="standard" and 20 with
                method="ergodic".
            short_window (int | None, optional): The number of periods of the second EMA. Defaults
                to None, which is 13 with method="standard" and 5 with method="ergodic".
            signal_window (int | None, optional): The number of periods of the EMA of the indicator
                that gives the signal line. Defaults to None, which is 7 with method="standard" and 5
                with method="ergodic".
            method (str, optional): Options are: "standard" (the True Strength Index and its signal line),
                "ergodic" (the SMI Ergodic Indicator, its signal line and the histogram, the SMI Ergodic
                Oscillator). Defaults to "standard".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the indicator lines.
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
            pd.Series or pd.DataFrame: True Strength Index and Signal Line values, or for the
            ergodic form the SMI Ergodic Indicator, Signal Line and Histogram (the SMI Ergodic
            Oscillator) values, between -100 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the True Strength Index or the SMI Ergodic lines for each asset in the Toolkit instance.
        - Every EMA is seeded with the simple average of its first values and is missing
          before it, as in TA-Lib and StockCharts, so the indicator starts `long_window` +
          `short_window` - 1 periods after the first price and the signal line (and histogram)
          `signal_window` - 1 periods later.
        - A stretch without any price change leaves the indicator missing.
        - The TSI agrees with TA-Lib's TSI, which has no signal line, with either set of windows.
          The three SMI Ergodic columns agree with pandas-ta-classic's smi times 100, as
          pandas-ta-classic uses a scale of -1 to 1.
        - When growth is requested, the method calculates the growth of the indicator lines
          over the given lag.

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
        self._check_arguments(
            period,
            close_column,
            long_window=long_window,
            short_window=short_window,
            signal_window=signal_window,
        )
        check_option("method", method, ("standard", "ergodic"))
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        # Blau's windows for the TSI and the common ones of the SMI Ergodic Indicator.
        default_windows = (20, 5, 5) if method == "ergodic" else (25, 13, 7)
        long_window = long_window if long_window is not None else default_windows[0]
        short_window = short_window if short_window is not None else default_windows[1]
        signal_window = (
            signal_window if signal_window is not None else default_windows[2]
        )

        historical_data = self._historical_data[period]

        true_strength_index = calculate_per_ticker(
            historical_data,
            [close_column],
            (
                momentum_model.get_smi_ergodic_indicator
                if method == "ergodic"
                else momentum_model.get_true_strength_index
            ),
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
        Calculate the Balance of Power (BOP): how far the close lies from the open as a share of the range, which
        shows whether buyers or sellers control the period.

        The Balance of Power (BOP) indicator measures the strength of buyers versus sellers
        in the market. It relates the price change to the change in the asset's trading range.

        The formula is as follows:

        - BOP = (Close — Open) / (High — Low)

        For more information, see the following sources:

        - Igor Livshin, who created the indicator; described in Kaufman, P.J. (2005). "New Trading Systems and
          Methods." 4th ed. Wiley.

        Referenced against:
        - TA-Lib, BOP: https://ta-lib.org/functions/bop.html

        Also known as: BOP, bull bear power.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The BOP agrees with TA-Lib's BOP.
        - When growth is requested, the method calculates the growth of the BOP
          over the given lag.

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
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Internal Bar Strength (IBS): where the close lies within the range of its period, from 0 (at
        the low) to 1 (at the high).

        The Internal Bar Strength is where the close sits within the period's own range: 0 for a close
        on the low, 1 for a close on the high. It is a mean-reversion measure: in equity indices and ETFs
        a close near the low of the day (an IBS below about 0.2) has tended to be followed by higher
        returns the next day, and a close near the high (above about 0.8) by lower returns.

        The formula is as follows:

        - IBS = (Close — Low) / (High — Low)

        For more information, see the following sources:

        - Pagonidis, A.S. (2014). "The IBS Effect: Mean Reversion in Equity ETFs." NAAIM Founders Award paper.

        Referenced against:
        - TA-Lib, IBS: https://ta-lib.org/functions/ibs.html

        Also known as: IBS, internal bar strength.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The IBS agrees with TA-Lib's IBS, except on a period without a range (High equal to Low), which
          has no IBS (NaN) where TA-Lib returns 0.5.
        - When growth is requested, the method calculates the growth of the IBS
          over the given lag.

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
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Qstick: the average of the close minus the open over a window, which shows whether candles
        have been mostly rising or falling.

        Qstick is the moving average of the candlestick bodies, Close — Open, over the window. It turns
        the colour of the recent candles into a number: above zero the periods closed above their open on
        average (white candles, buying pressure), below zero below their open (black candles, selling
        pressure). A cross of zero is read as a change of the pressure.

        The formula is as follows:

        - Qstick = SMA(Close — Open, window)

        For more information, see the following sources:

        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

        Referenced against:
        - TA-Lib, QSTICK: https://ta-lib.org/functions/qstick.html

        Also known as: Qstick, QSTICK, Q-stick.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - Qstick agrees with TA-Lib's QSTICK and pandas-ta-classic's qstick.
        - When growth is requested, the method calculates the growth of the Qstick
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        signal_window: int = 5,
        method: Literal["standard", "accelerator"] = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Awesome Oscillator (AO) or the Accelerator/Decelerator Oscillator (AC), which show the
        momentum of the market (AO) and whether it speeds up or slows down (AC).

        The AO is the difference between a fast and a slow average of the median price.

        The Awesome Oscillator measures market momentum by comparing a short-term and a
        long-term Simple Moving Average of the median price (the midpoint of each period's high
        and low, rather than the closing price). The Accelerator Oscillator is the Awesome
        Oscillator minus its own 5-period Simple Moving Average: where the Awesome Oscillator
        measures the momentum of the median price, the Accelerator Oscillator measures how that
        momentum changes, on Bill Williams' premise that momentum changes direction before price
        does. Both were developed by Bill Williams as part of his "Trading Chaos" collection of
        momentum indicators.

        The formula for the Awesome Oscillator is as follows:

        - Median Price = (High + Low) / 2
        - AO = SMA(Median Price, short_window) — SMA(Median Price, long_window)

        The formula for the Accelerator/Decelerator Oscillator is as follows:

        - Median Price = (High + Low) / 2
        - AO = SMA(Median Price, short_window) — SMA(Median Price, long_window)
        - AC = AO — SMA(AO, signal_window)

        A cross of the AO above zero occurs exactly when the short-window SMA of the median
        price crosses above the long-window SMA, and vice versa for a cross below zero. Williams
        reads the AC by the colour of its bars: two rising bars in a row above zero (three below
        zero) are a buy signal, two falling bars below zero (three above zero) a sell signal.

        For more information, see the following sources:

        - Williams, B. (1995). "Trading Chaos: Applying Expert Techniques to Maximize Your
          Profits." Wiley.
        - Williams, B. (1998). "New Trading Dimensions: How to Profit from Chaos in Stocks, Bonds, and
          Commodities." Wiley.

        Referenced against:
        - TA-Lib, AO: https://ta-lib.org/functions/ao.html
        - TA-Lib, AC: https://ta-lib.org/functions/ac.html

        Also known as: AO, Bill Williams Awesome Oscillator, AC, Accelerator Oscillator,
        Acceleration/Deceleration Oscillator, AC oscillator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            short_window (int, optional): The number of periods for the short-term SMA of the
                median price. Defaults to 5.
            long_window (int, optional): The number of periods for the long-term SMA of the
                median price. Defaults to 34.
            signal_window (int, optional): The number of periods for the SMA of the Awesome
                Oscillator, with method="accelerator" only. Defaults to 5.
            method (str, optional): Options are: "standard" (the Awesome Oscillator), "accelerator" (the
                Accelerator Oscillator). Defaults to "standard".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the oscillator.
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
            pd.Series or pd.DataFrame: Awesome Oscillator (AO) or Accelerator Oscillator (AC)
            values, in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the AO or AC for each asset in the Toolkit instance.
        - There is no academic journal citation for the Awesome and Accelerator Oscillators.
          Like most of Bill Williams' indicators, they are practitioner-developed tools rather
          than ones derived from a published financial paper; the standard textbook sources are
          Williams' books above.
        - The AO agrees with TA-Lib's AO, and the AC with TA-Lib's AC; the AC starts
          `long_window` + `signal_window` — 2 periods after the first price.
        - When growth is requested, the method calculates the growth of the oscillator
          over the given lag.

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
        self._check_arguments(
            period,
            close_column,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
        )
        check_option("method", method, ("standard", "accelerator"))
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        if method == "accelerator":
            awesome_oscillator = calculate_per_ticker(
                historical_data,
                ["High", "Low"],
                momentum_model.get_accelerator_oscillator,
                short_window,
                long_window,
                signal_window,
            )
        else:
            awesome_oscillator = calculate_per_ticker(
                historical_data,
                ["High", "Low"],
                momentum_model.get_awesome_oscillator,
                short_window,
                long_window,
            )

        return finalize_dataset(
            dataset=awesome_oscillator.loc[self._start_date : self._end_date],
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
        Calculate the Vortex Indicator: the upward and downward movement of the prices relative to their true range,
        whose crossings mark the start of a new trend.

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
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
          of Stocks & Commodities, 28(1), 20-30.
        - VI+ and VI- agree with TA-Lib's VORTEX.
        - When growth is requested, the method calculates the growth of the VI+ and VI-
          over the given lag.

        Referenced against:
        - TA-Lib, VORTEX: https://ta-lib.org/functions/vortex.html

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Elder Ray Index: how far the high (Bull Power) and the low (Bear Power) lie from an
        exponential moving average, which shows the strength of the buyers and the sellers.

        The Elder Ray Index measures buying and selling pressure in the market relative to a
        trend baseline (an Exponential Moving Average of the closing price). Bull Power captures
        how far the high extends above the EMA (buying pressure), while Bear Power captures how
        far the low extends below the EMA (selling pressure).

        The formula is as follows:

        - Bull Power = High — EMA(Close, window)
        - Bear Power = Low — EMA(Close, window)

        Also known as: Elder Ray, Bull Power, Bear Power.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - Bull Power and Bear Power agree with TA-Lib's ERI.
        - When growth is requested, the method calculates the growth of the Bull and Bear Power
          over the given lag.

        Referenced against:
        - TA-Lib, ERI: https://ta-lib.org/functions/eri.html

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        window: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        method: Literal["percentage", "absolute"] = "percentage",
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Rate of Change (ROC), as a percentage or in price units (the Momentum): the change of the
        price over a window, the speed at which the price moves.

        The Rate of Change is a pure momentum oscillator that measures the percentage
        change in price between the current period and the price a fixed number of periods
        ago. It oscillates around zero: positive values indicate price is higher than
        `window` periods ago (upward momentum), while negative values indicate price is
        lower (downward momentum). The Momentum is the same change in price
        units: the simplest measure of the speed of a move. Momentum that falls while the price
        still rises says the move slows down. The Rate of Change is the change as a share of the
        earlier price, which can be compared across stocks; the Momentum is in the currency of the
        price.

        The formula is as follows, per method:

        - Rate of Change: ROC = Close(t) / Close(t - window) — 1
        - Momentum: Momentum = Close(t) — Close(t — window)

        For more information, see the following sources:

        - Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York Institute of
          Finance.
        - Pring, M.J. (1991). "Technical Analysis Explained." 3rd ed. McGraw-Hill.

        Referenced against:
        - TA-Lib, ROCP: https://ta-lib.org/functions/rocp.html
        - TA-Lib, MOM: https://ta-lib.org/functions/mom.html

        Also known as: ROC, Price Rate of Change, momentum, MOM, price momentum, momentum oscillator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to look back for the rate of change
                calculation. Defaults to None, which uses 12 for "percentage" and 10 for "absolute".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Rate of Change.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): Options are: "percentage" (the Rate of Change as a decimal), "absolute" (the
                Momentum in price units). Defaults to "percentage".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame: Rate of Change values as a decimal (0.05 is 5%), or for the
            Momentum, the change in price in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Rate of Change for each asset in the Toolkit instance.
        - The Rate of Change is a decimal and agrees with TA-Lib's ROCP; TA-Lib's ROC is the same in
          percent.
        - Momentum: the Momentum agrees with TA-Lib's MOM, pandas-ta-classic's mom and finta's MOM.
        - When growth is requested, the method calculates the growth of the Rate of Change
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        methods = ("percentage", "absolute")
        check_option("method", method, methods)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        if window is None:
            window = 10 if method == "absolute" else 12

        historical_data = self._historical_data[period]

        rate_of_change = calculate_per_ticker(
            historical_data,
            [close_column],
            (
                momentum_model.get_momentum
                if method == "absolute"
                else momentum_model.get_rate_of_change
            ),
            window,
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
        Calculate the Choppiness Index (CHOP): how much of the range of a window the price travelled, from trending
        (near 0) to choppy and sideways (near 100).

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

        With the true range as the box, the net range is that of the true highs and lows,
        Max(High, Previous Close) and Min(Low, Previous Close), the range the summed True Ranges span,
        so a gap between periods counts in both the path and the net range (TA-Lib's CHOPTR).

        For more information, see the following sources:

        - Kaufman, P.J. (2013). "Trading Systems and Methods." 5th ed. Wiley.

        Referenced against:
        - TA-Lib, CHOP: https://ta-lib.org/functions/chop.html
        - TA-Lib, CHOPTR: https://ta-lib.org/functions/choptr.html

        Also known as: CHOP, CHOPTR, Choppiness Index.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods to consider for the Choppiness Index
                calculation, at least 2. Defaults to 14.
            range_method (str, optional): The net range the summed True Range is divided by. Options are:
                "high_low" (the highest high minus the lowest low), "true_range" (the highest true high minus
                the lowest true low, TA-Lib's CHOPTR). Defaults to "high_low".
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
        - The Choppiness Index agrees with TA-Lib's CHOP, and with the true range as the box TA-Lib's
          CHOPTR. The first period counts with its High — Low range, so the index starts on period
          `window`, one period before TA-Lib's.
        - When growth is requested, the method calculates the growth of the Choppiness
          Index over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        # The denominator log10(window) is 0 for a window of one period.
        validate_window("window", window, 2)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate Kaufman's Efficiency Ratio (ER): the net change over a window as a share of the path travelled,
        from 0 (noise) to 1 (a straight trend).

        The Efficiency Ratio compares the net change in price over the window with the path the price
        travelled to get there, the sum of the absolute changes of each period. A price that moves in a
        straight line has a ratio of 1; a price that ends where it started, however much it moved in
        between, has a ratio of 0. Perry Kaufman uses it to adapt the speed of his Adaptive Moving
        Average (the Kaufman adaptive method of `get_moving_average`) to the noise in the market.

        The formula is as follows:

        - Change = |Close(t) — Close(t — window)|
        - Volatility = Sum(|Close(i) — Close(i — 1)|, window)
        - Efficiency Ratio = Change / Volatility

        For more information, see the following sources:

        - Kaufman, P.J. (1995). "Smarter Trading: Improving Performance in Changing Markets." McGraw-Hill.

        Referenced against:
        - TA-Lib, ER: https://ta-lib.org/functions/er.html

        Also known as: ER, Kaufman Efficiency Ratio, fractal efficiency.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The Efficiency Ratio agrees with TA-Lib's ER, pandas-ta-classic's er and finta's ER, except on a
          window without any change (flat prices), which has no ratio (NaN) where TA-Lib returns 1.
        - When growth is requested, the method calculates the growth of the Efficiency Ratio
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Vertical Horizontal Filter (VHF): the range of a window as a share of the path travelled,
        which shows whether prices trend or move sideways.

        The Vertical Horizontal Filter tells a trending market from a congested one by dividing the range
        of the closes over the window (the vertical move) by the sum of the absolute changes from period
        to period (the path, horizontal in a range). A high or rising VHF says the price moves in one
        direction, which suits trend-following indicators; a low or falling VHF says it moves back and
        forth, which suits oscillators.

        The formula is as follows:

        - VHF = (Max(Close, window) — Min(Close, window)) / Sum(|Close(i) — Close(i — 1)|, window)

        For more information, see the following sources:

        - White, A. (1991). "Trending or Trading? The Vertical Horizontal Filter." Futures Magazine.

        Referenced against:
        - TA-Lib, VHF: https://ta-lib.org/functions/vhf.html

        Also known as: VHF, vertical horizontal filter.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The VHF agrees with TA-Lib's VHF and pandas-ta-classic's vhf and starts `window` periods after the
          first price, except on a window without any change (flat prices), which has no VHF (NaN) where TA-Lib
          returns 0.
        - When growth is requested, the method calculates the growth of the VHF
          over the given lag.

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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Know Sure Thing (KST): a weighted sum of four smoothed rates of change over increasing
        windows, which shows the momentum of the major price cycles.

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

        - Pring, M.J. (1992). "Summed Rate of Change (KST)." Technical Analysis of Stocks & Commodities, 10(9),
          365-369.

        Referenced against:
        - TA-Lib, KST: https://ta-lib.org/functions/kst.html
        - TA-Lib, KSTEXT: https://ta-lib.org/functions/kstext.html

        Also known as: KST, KSTEXT, Pring's Know Sure Thing, Summed Rate of Change.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
            moving_average (str, optional): The moving average that smooths each Rate of Change. Options are:
                "simple" (Simple Moving Average), "exponential" (Exponential Moving Average),
                "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular Moving
                Average), "weighted" (Weighted Moving Average), "hull" (Hull Moving Average),
                "kaufman_adaptive" (Kaufman Adaptive Moving Average). Defaults to "simple".
            signal_moving_average (str, optional): The moving average of the signal line. Options are:
                "simple" (Simple Moving Average), "exponential" (Exponential Moving Average),
                "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular Moving
                Average), "weighted" (Weighted Moving Average), "hull" (Hull Moving Average),
                "kaufman_adaptive" (Kaufman Adaptive Moving Average). Defaults to "simple".
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
        - The KST and Signal Line are decimals and agree with TA-Lib's KST divided by 100, since TA-Lib's
          KST is in percent. TA-Lib starts the KST on the first signal value.
        - With other moving averages they agree with TA-Lib's KSTEXT with the same `rocmatype` and
          `signalmatype` divided by 100: from the first value for the simple, weighted and triangular
          averages, and after the warm-up for the recursive ones, since TA-Lib starts their recursion
          later (within 0.01% after about 85 periods for exponential averages).
        - When growth is requested, the method calculates the growth of the KST and Signal
          Line over the given lag.

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
        self._check_arguments(period, close_column, signal_window=signal_window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        Calculate the Coppock Curve: a weighted moving average of the sum of a long and a short rate of change,
        which shows long-term buying opportunities after a decline.

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
        monthly, the default of daily periods applies the same windows to days, as the other
        technical indicators do. The classic signal is the curve turning up while below zero;
        Coppock did not use it for sell signals. The Rates of Change are decimals, so the curve
        is a weighted sum of decimal returns (0.30 where charting platforms show 30).

        Also known as: Coppock Indicator, Coppock Guide.

        Referenced against:
        - TA-Lib, COPPOCK: https://ta-lib.org/functions/coppock.html

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The Coppock Curve is a decimal and agrees with TA-Lib's COPPOCK divided by 100, since TA-Lib's
          is in percent.
        - When growth is requested, the method calculates the growth of the Coppock Curve
          over the given lag.

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
        self._check_arguments(
            period,
            close_column,
            long_roc_window=long_roc_window,
            short_roc_window=short_roc_window,
            wma_window=wma_window,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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

    @handle_portfolio
    @handle_errors
    def get_fisher_transform(
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
        Calculate the Fisher Transform: the prices' position within their range turned into a near-normal
        distribution, which makes the turning points sharper and easier to see.

        The Fisher Transform of Ehlers (2002) turns the position of the median price within its
        recent range into a series that is close to normally distributed. An oscillator of the
        position spends most of its time in the middle and gives vague turning points; the
        transform stretches the values near the edges of the range, so the extremes, and the
        turns after them, become sharp and rare. The trigger is the previous value of the
        transform, and the transform crossing its trigger marks a turn.

        The formula is as follows:

        - Median Price = (High + Low) / 2
        - Lowest = Min(Median Price, window) and Highest = Max(Median Price, window)
        - Position = (Median Price — Lowest) / (Highest — Lowest)
        - Value(t) = 0.66 * (Position — 0.5) + 0.67 * Value(t - 1), set to 0.999 above 0.99 and to -0.999 below -0.99
        - Fisher Transform(t) = 0.5 * ln((1 + Value(t)) / (1 — Value(t))) + 0.5 * Fisher Transform(t - 1)
        - Trigger(t) = Fisher Transform(t - 1)

        Value and Fisher Transform start from 0, as in Ehlers' published formula. The transform is centred on
        zero and bounded by ln(1999), about -7.6 and 7.6.

        For more information, see the following sources:

        - Ehlers, J.F. (2002). "Using the Fisher Transform." Technical Analysis of Stocks &
          Commodities, 20(11).
        - Ehlers, J.F. (2004). "Cybernetic Analysis for Stocks and Futures." Wiley.

        Referenced against:
        - TA-Lib, FISHER: https://ta-lib.org/functions/fisher.html

        Shares similar logic:
        - pandas-ta-classic, fisher: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/fisher.py#L36
        - finta, FISH: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1827

        Also known as: Fisher, Ehlers Fisher Transform, FISHT.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the range of the median price.
                Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Fisher Transform and trigger.
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
            pd.Series or pd.DataFrame: Fisher Transform and Trigger values, centred on zero.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Fisher Transform and trigger for each asset in the Toolkit instance.
        - The Fisher Transform agrees with TA-Lib's FISHER and starts `window` - 1 periods after the first price,
          the trigger one period later (TA-Lib gives the first trigger the value 0).
        - A window in which the median price did not move gives the neutral position 0.5, as in TA-Lib, so the
          transform decays towards zero.
        - pandas-ta-classic's fisher (9 periods by default) starts one period later and equals it after the
          first values; finta's FISH smooths with exponential averages of 5 and 3 periods instead of
          Ehlers' constants, a different series.
        - When growth is requested, the method calculates the growth of the Fisher Transform and trigger
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_fisher_transform().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Fisher Transform |   Trigger |
        |:-----------|-------------------:|----------:|
        | 2025-12-24 |            -1.5593 |   -1.9612 |
        | 2025-12-26 |            -1.1485 |   -1.5593 |
        | 2025-12-29 |            -0.8195 |   -1.1485 |
        | 2025-12-30 |            -0.4022 |   -0.8195 |
        | 2025-12-31 |            -0.1073 |   -0.4022 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        fisher_transform = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            momentum_model.get_fisher_transform,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=fisher_transform,
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
    def get_inverse_fisher_transform_rsi(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rsi_window: int = 5,
        wma_window: int = 9,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Inverse Fisher Transform of the RSI: the smoothed Relative Strength Index squeezed between -1
        and 1, which makes its overbought and oversold signals clearer.

        The inverse Fisher transform (the hyperbolic tangent) of Ehlers (2004) squeezes a series
        into the range from -1 to 1 with most values close to the edges. Applied to a rescaled
        and smoothed short Relative Strength Index, it turns the RSI into a nearly binary signal
        that switches quickly between overbought and oversold. Ehlers buys when it crosses above
        -0.5 and sells when it crosses below 0.5.

        The formula is as follows:

        - Value 1 = 0.1 * (RSI(Close, rsi_window) — 50)
        - Value 2 = WMA(Value 1, wma_window)
        - IFT RSI = (exp(2 * Value 2) — 1) / (exp(2 * Value 2) + 1) = tanh(Value 2)

        Ehlers uses a 5-period RSI and a 9-period weighted moving average.

        For more information, see the following sources:

        - Ehlers, J.F. (2004). "The Inverse Fisher Transform." Technical Analysis of Stocks &
          Commodities, 22(5).

        Shares similar logic:
        - finta, IFT_RSI: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L714

        Also known as: IFT RSI, IFTRSI, Inverse Fisher RSI.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rsi_window (int, optional): The number of periods of the RSI. Defaults to 5.
            wma_window (int, optional): The number of periods of the weighted moving average of
                the rescaled RSI. Defaults to 9.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the IFT RSI.
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
            pd.Series or pd.DataFrame: IFT RSI values, between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the IFT RSI for each asset in the Toolkit instance.
        - The RSI is that of `get_relative_strength_index` (Wilder's smoothing, seeded with a simple average,
          as TA-Lib's RSI), so the IFT RSI starts `rsi_window` + `wma_window` - 1 periods after the
          first price.
        - finta's IFT_RSI returns (Value 2² - 1) / (Value 2² + 1) instead of the hyperbolic tangent, with an
          RSI from an unseeded exponential average; this follows Ehlers' formula.
        - When growth is requested, the method calculates the growth of the IFT RSI
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_inverse_fisher_transform_rsi().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -0.8895 | 0.6215 |      0.6121 |
        | 2025-12-26 | -0.8224 | 0.765  |      0.8322 |
        | 2025-12-29 | -0.6882 | 0.8175 |      0.8765 |
        | 2025-12-30 | -0.6065 | 0.8526 |      0.8845 |
        | 2025-12-31 | -0.6548 | 0.6957 |      0.759  |
        """
        self._check_arguments(
            period, close_column, rsi_window=rsi_window, wma_window=wma_window
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        inverse_fisher_transform_rsi = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_inverse_fisher_transform_rsi,
            rsi_window,
            wma_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=inverse_fisher_transform_rsi,
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
    def get_kdj_indicator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 9,
        k_window: int = 3,
        d_window: int = 3,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the KDJ indicator: the Stochastic Oscillator's %K and %D with the J line that leads them, to show
        overbought and oversold levels and turning points.

        The KDJ indicator is a stochastic oscillator, widely used in Chinese and other Asian
        markets, that smooths the raw stochastic with Wilder's average instead of a simple one
        and adds a third line, J, which extrapolates K away from D. J leaves the 0 to 100 range
        when the momentum is strong, so J above 100 (or below 0) is read as overbought (or
        oversold) before K and D get there, and K crossing D is the trade signal, as with the
        stochastic.

        The formula is as follows:

        - RSV = 100 * (Close — Lowest Low(window)) / (Highest High(window) — Lowest Low(window))
        - K = Wilder's MA(RSV, k_window)
        - D = Wilder's MA(K, d_window)
        - J = 3 * K — 2 * D

        Wilder's moving average is the recursion K(t) = K(t - 1) + (RSV(t) — K(t - 1)) / k_window,
        written SMA(RSV, 3, 1) in the formula language of Asian charting platforms. The KDJ has
        no single primary source; it extends Lane's stochastic (see `get_stochastic_oscillator`).

        For more information, see the following sources:

        - Lane, G.C. (1984). "Lane's Stochastics." Technical Analysis of Stocks & Commodities, 2(3).

        Referenced against:
        - TA-Lib, KDJ: https://ta-lib.org/functions/kdj.html

        Shares similar logic:
        - pandas-ta-classic, kdj: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/kdj.py#L17

        Also known as: KDJ, KDJ stochastic, random index.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the high-low range of the RSV.
                Defaults to 9.
            k_window (int, optional): The number of periods of Wilder's average of the RSV that
                gives K. Defaults to 3.
            d_window (int, optional): The number of periods of Wilder's average of K that gives
                D. Defaults to 3.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the K, D and J.
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
            pd.Series or pd.DataFrame: K, D and J values; K and D between 0 and 100, J unbounded.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the K, D and J for each asset in the Toolkit instance.
        - K, D and J agree with TA-Lib's KDJ (with its default RMA averages) and pandas-ta-classic's kdj. Every
          Wilder average is seeded with the simple average of its first values, so K starts `window`
          + `k_window` - 2 periods after the first price and D and J `d_window` - 1 periods later;
          TA-Lib starts K on the first D. Platforms that start K and D at 50 differ in the first few
          dozen periods.
        - A window without a range has no RSV (NaN), where TA-Lib and pandas-ta-classic use 0; the averages
          carry on over it.
        - When growth is requested, the method calculates the growth of the K, D and J
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_kdj_indicator().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |       D |       J |       K |
        |:-----------|--------:|--------:|--------:|
        | 2025-12-24 | 34.2194 | 53.5727 | 40.6705 |
        | 2025-12-26 | 37.2801 | 55.6444 | 43.4016 |
        | 2025-12-29 | 42.7139 | 75.3166 | 53.5815 |
        | 2025-12-30 | 47.7783 | 78.1645 | 57.907  |
        | 2025-12-31 | 51.1539 | 71.4073 | 57.905  |
        """
        self._check_arguments(
            period, close_column, window=window, k_window=k_window, d_window=d_window
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        kdj_indicator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_kdj_indicator,
            window,
            k_window,
            d_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=kdj_indicator,
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
    def get_premier_stochastic_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 8,
        smooth_window: int = 5,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Premier Stochastic Oscillator (PSO): the smoothed and normalised Stochastic %K squeezed
        between -1 and 1, which gives clearer overbought and oversold signals.

        The Premier Stochastic Oscillator of Leibfarth (2008) rescales a short stochastic around
        zero, smooths it twice with an exponential average and squeezes the result between -1
        and 1 with the inverse Fisher transform. The smoothing removes the noise of the
        stochastic and the transform keeps the oscillator near its extremes most of the time,
        so a reading that crosses back through 0.9 or -0.9 (or 0.2 and -0.2) stands out.

        The formula is as follows:

        - %K = 100 * (Close — Lowest Low(window)) / (Highest High(window) — Lowest Low(window))
        - Normalized Stochastic = 0.1 * (%K — 50)
        - Smoothed Stochastic = EMA(EMA(Normalized Stochastic, smooth_window), smooth_window)
        - PSO = (exp(Smoothed Stochastic) — 1) / (exp(Smoothed Stochastic) + 1)

        Leibfarth uses an 8-period stochastic and a smoothing of 5 periods (the square root of 25).

        For more information, see the following sources:

        - Leibfarth, L. (2008). "Premier Stochastic Oscillator." Technical Analysis of Stocks &
          Commodities, 26(8).

        Referenced against:
        - TA-Lib, PSO: https://ta-lib.org/functions/pso.html

        Also known as: PSO, Premier Stochastic.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the high-low range of %K.
                Defaults to 8.
            smooth_window (int, optional): The number of periods of each of the two EMAs.
                Defaults to 5.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the PSO.
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
            pd.Series or pd.DataFrame: PSO values, between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the PSO for each asset in the Toolkit instance.
        - The PSO agrees with TA-Lib's PSO. Every EMA is seeded with the simple average of its first values and is
          missing before it, so the PSO starts `window` + 2 * `smooth_window` - 3 periods after the
          first price.
        - A window without a range has no %K (NaN), where TA-Lib uses 50; the averages carry on over it.
        - When growth is requested, the method calculates the growth of the PSO
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_premier_stochastic_oscillator().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -0.6448 | 0.5466 |      0.5421 |
        | 2025-12-26 | -0.4625 | 0.7272 |      0.7504 |
        | 2025-12-29 | -0.1913 | 0.8145 |      0.8373 |
        | 2025-12-30 |  0.0877 | 0.8577 |      0.8676 |
        | 2025-12-31 |  0.1423 | 0.7679 |      0.8168 |
        """
        self._check_arguments(
            period, close_column, window=window, smooth_window=smooth_window
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        premier_stochastic_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_premier_stochastic_oscillator,
            window,
            smooth_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=premier_stochastic_oscillator,
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
    def get_stochastic_momentum_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 13,
        long_window: int = 25,
        short_window: int = 2,
        signal_window: int = 9,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Stochastic Momentum Index (SMI): where the close lies relative to the middle of the high-low
        range of a window, smoothed, from -100 to 100.

        The Stochastic Momentum Index of Blau (1993) measures where the close sits relative to
        the midpoint of the high-low range, rather than to its low as the stochastic does, and
        smooths both the distance and the range twice with an exponential average before
        dividing. It runs from -100 (a close on the low) to +100 (a close on the high) around
        zero, is less erratic than the stochastic, and is read with its signal line and the
        levels -40 and 40.

        The formula is as follows:

        - Distance = Close — (Highest High(window) + Lowest Low(window)) / 2
        - Range = Highest High(window) — Lowest Low(window)
        - SMI = 100 * EMA(EMA(Distance, long_window), short_window) / (0.5 * EMA(EMA(Range, long_window), short_window))
        - Signal Line = EMA(SMI, signal_window)

        Blau uses a 13-period range, smoothed over 25 and then 2 periods.

        For more information, see the following sources:

        - Blau, W. (1993). "Stochastic Momentum." Technical Analysis of Stocks & Commodities, 11(1).
        - Blau, W. (1995). "Momentum, Direction, and Divergence." Wiley.

        Referenced against:
        - TA-Lib, SMI: https://ta-lib.org/functions/smi.html

        Also known as: SMI, Blau's stochastic momentum.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the high-low range. Defaults to 13.
            long_window (int, optional): The number of periods of the first EMA of the distance
                and the range. Defaults to 25.
            short_window (int, optional): The number of periods of the second EMA. Defaults to 2.
            signal_window (int, optional): The number of periods of the EMA of the SMI that gives
                the signal line. Defaults to 9.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the SMI and signal line.
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
            pd.Series or pd.DataFrame: Stochastic Momentum Index and Signal Line values, between -100 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the SMI and signal line for each asset in the Toolkit instance.
        - The SMI and signal line agree with TA-Lib's SMI. Every EMA is seeded with the simple average of its first
          values and is missing before it, so the SMI starts `window` + `long_window` +
          `short_window` - 3 periods after the first price and the signal line `signal_window` - 1
          periods later; TA-Lib starts both on the first signal value.
        - A stretch in which the smoothed range is zero has no SMI (NaN), where TA-Lib returns 0.
        - When growth is requested, the method calculates the growth of the SMI and signal line
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_stochastic_momentum_index().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Signal Line |   Stochastic Momentum Index |
        |:-----------|--------------:|----------------------------:|
        | 2025-12-24 |        5.6181 |                     -7.0164 |
        | 2025-12-26 |        3.113  |                     -6.9074 |
        | 2025-12-29 |        1.194  |                     -6.4823 |
        | 2025-12-30 |       -0.3191 |                     -6.3715 |
        | 2025-12-31 |       -1.6769 |                     -7.108  |
        """
        self._check_arguments(
            period,
            close_column,
            window=window,
            long_window=long_window,
            short_window=short_window,
            signal_window=signal_window,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        stochastic_momentum_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_stochastic_momentum_index,
            window,
            long_window,
            short_window,
            signal_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=stochastic_momentum_index,
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
    def get_schaff_trend_cycle(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 23,
        long_window: int = 50,
        cycle_window: int = 10,
        factor: float = 0.5,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Schaff Trend Cycle (STC): a stochastic of the MACD, applied twice, from 0 to 100, which shows
        the turns of a trend faster than the MACD.

        The Schaff Trend Cycle of Schaff (2002, 2008) applies a smoothed stochastic twice to the
        MACD line: it measures where the MACD sits within its range over the last cycle, smooths
        that, and does the same again. The result moves between 0 and 100 and turns earlier than
        the MACD, as it follows the MACD's cycle rather than its level. A turn up from below 25
        is read as bullish and a turn down from above 75 as bearish.

        The formula is as follows:

        - MACD = EMA(Close, short_window) — EMA(Close, long_window)
        - Fraction 1 = 100 * (MACD — Min(MACD, cycle_window)) / (Max(MACD, cycle_window) — Min(MACD, cycle_window))
        - PF(t) = PF(t - 1) + factor * (Fraction 1(t) — PF(t - 1))
        - Fraction 2 = 100 * (PF — Min(PF, cycle_window)) / (Max(PF, cycle_window) — Min(PF, cycle_window))
        - STC(t) = STC(t - 1) + factor * (Fraction 2(t) — STC(t - 1))

        A fraction over a window without a range keeps its previous value and each smoothing
        starts on its first input, as in Schaff (2008). Schaff uses 23, 50 and 10 with a factor
        of 0.5.

        For more information, see the following sources:

        - Schaff, D. (2008). "Releasing the Code to the Schaff Trend Cycle." FX-Strategy.com.
        - Schaff, D. (2002). "Catching Currency Moves with the Schaff Trend Cycle Indicator."
          Chartpoint, July/August 2002.

        Referenced against:
        - TA-Lib, STC: https://ta-lib.org/functions/stc.html

        Shares similar logic:
        - pandas-ta-classic, stc: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/stc.py#L100
        - ta, STCIndicator: https://github.com/bukosabino/ta/blob/0.11.0/ta/trend.py#L1093
        - finta, STC: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L2121

        Also known as: STC, Schaff Trend Cycle indicator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): The number of periods of the short EMA of the MACD.
                Defaults to 23.
            long_window (int, optional): The number of periods of the long EMA of the MACD.
                Defaults to 50.
            cycle_window (int, optional): The number of periods of the range of each stochastic.
                Defaults to 10.
            factor (float, optional): The smoothing factor of each stochastic. Defaults to 0.5.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the STC.
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
            pd.Series or pd.DataFrame: STC values, between 0 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the STC for each asset in the Toolkit instance.
        - The EMAs of the MACD are seeded with the simple average of their first values with an empty warm-up,
          as in `get_moving_average_convergence_divergence`. TA-Lib's STC starts the short EMA on
          the same period as the long one, so the two differ in the first values (up to 50 points)
          and agree within 0.01 after about 200 periods and within 1e-5 after about 250 at the
          default windows; on the same MACD line the two stochastic stages agree with TA-Lib's.
        - pandas-ta-classic's stc (12 and 26 periods by default) equals it with the same windows after the
          first values, and ta's STCIndicator, which smooths with a 3-period EMA (a factor of 0.5)
          without a seed, within 1e-6 after about 500 periods. finta's STC smooths with 3-period
          simple averages, a different series.
        - When growth is requested, the method calculates the growth of the STC
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_schaff_trend_cycle().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |    MSFT |   Benchmark |
        |:-----------|-------:|--------:|------------:|
        | 2025-12-24 | 0.0411 | 95.6845 |     18.5194 |
        | 2025-12-26 | 0.0206 | 97.8422 |     35.7456 |
        | 2025-12-29 | 0.0103 | 98.9211 |     60.1545 |
        | 2025-12-30 | 0.0051 | 99.4606 |     80.0773 |
        | 2025-12-31 | 0.0026 | 99.7303 |     87.8871 |
        """
        self._check_arguments(
            period,
            close_column,
            short_window=short_window,
            long_window=long_window,
            cycle_window=cycle_window,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        schaff_trend_cycle = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_schaff_trend_cycle,
            short_window,
            long_window,
            cycle_window,
            factor,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=schaff_trend_cycle,
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
    def get_wavetrend_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        channel_window: int = 10,
        average_window: int = 21,
        signal_window: int = 4,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the WaveTrend Oscillator: how far the typical price lies from its average in units of its average
        distance, smoothed, to show overbought and oversold levels and turning points.

        The WaveTrend Oscillator of LazyBear (2014) is a Commodity Channel Index built from
        exponential averages: the distance of the typical price from its EMA, divided by 0.015
        times the EMA of that absolute distance, and smoothed with a longer EMA. Its signal line
        is a short simple average. Readings beyond 53 or 60 (or -53 and -60) are read as
        overbought (or oversold), and the oscillator crossing its signal line there as a turn.

        The formula is as follows:

        - Typical Price = (High + Low + Close) / 3
        - ESA = EMA(Typical Price, channel_window)
        - D = EMA(|Typical Price — ESA|, channel_window)
        - CI = (Typical Price — ESA) / (0.015 * D)
        - WaveTrend = EMA(CI, average_window)
        - Signal Line = SMA(WaveTrend, signal_window)

        LazyBear's parameters are 10, 21 and 4.

        For more information, see the following sources:

        - LazyBear (2014). "WaveTrend Oscillator [WT]." TradingView published script.

        Referenced against:
        - TA-Lib, WAVETREND: https://ta-lib.org/functions/wavetrend.html

        Shares similar logic:
        - finta, WTO: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1802

        Also known as: WaveTrend, WT, WTO, WT1 and WT2.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted
                for dividends in the same way.
            channel_window (int, optional): The number of periods of the EMA of the typical
                price and of its distance. Defaults to 10.
            average_window (int, optional): The number of periods of the EMA of the CI.
                Defaults to 21.
            signal_window (int, optional): The number of periods of the SMA that gives the
                signal line. Defaults to 4.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the WaveTrend and signal line.
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
            pd.Series or pd.DataFrame: WaveTrend and Signal Line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the WaveTrend and signal line for each asset in the Toolkit instance.
        - The WaveTrend and signal line agree with TA-Lib's WAVETREND. Every EMA is seeded with the simple average
          of its first values and is missing before it, so the WaveTrend starts 2 * `channel_window`
          + `average_window` - 3 periods after the first price and the signal line `signal_window` -
          1 periods later. finta's WTO uses averages without a seed and equals it after the first
          values.
        - A stretch without any distance from the average has no CI (NaN), where TA-Lib returns 0.
        - When growth is requested, the method calculates the growth of the WaveTrend and signal line
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_wavetrend_oscillator().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Signal Line |   WaveTrend |
        |:-----------|--------------:|------------:|
        | 2025-12-24 |      -21.1082 |    -22.4463 |
        | 2025-12-26 |      -22.0717 |    -20.284  |
        | 2025-12-29 |      -21.7469 |    -19.6328 |
        | 2025-12-30 |      -20.6589 |    -20.2723 |
        | 2025-12-31 |      -20.8875 |    -23.3609 |
        """
        self._check_arguments(
            period,
            close_column,
            channel_window=channel_window,
            average_window=average_window,
            signal_window=signal_window,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        wavetrend_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_wavetrend_oscillator,
            channel_window,
            average_window,
            signal_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=wavetrend_oscillator,
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
    def get_derivative_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        first_window: int = 5,
        second_window: int = 3,
        signal_window: int = 9,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Derivative Oscillator: a double-smoothed Relative Strength Index minus its average, which
        shows the momentum of the RSI itself.

        The Derivative Oscillator of Constance Brown (1994) smooths the Relative Strength Index
        twice with an exponential average and subtracts a simple average of the result, the way
        the MACD histogram subtracts the signal line from the MACD. It shows the momentum of the
        smoothed RSI: a crossing of zero marks a turn of the RSI, and the distance from zero how
        far the smoothed RSI has moved from its mean.

        The formula is as follows:

        - Double Smoothed RSI = EMA(EMA(RSI(Close, window), first_window), second_window)
        - Derivative Oscillator = Double Smoothed RSI — SMA(Double Smoothed RSI, signal_window)

        Brown uses a 14-period RSI, smoothed over 5 and then 3 periods, and a 9-period signal.

        For more information, see the following sources:

        - Brown, C.M. (1994). "The Derivative Oscillator: A New Approach for an Old Problem." MTA
          Journal, 42.
        - Brown, C.M. (1999). "Technical Analysis for the Trading Professional." McGraw-Hill.

        Referenced against:
        - TA-Lib, DOSC: https://ta-lib.org/functions/dosc.html

        Also known as: DOSC, Brown's Derivative Oscillator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the RSI. Defaults to 14.
            first_window (int, optional): The number of periods of the first EMA of the RSI.
                Defaults to 5.
            second_window (int, optional): The number of periods of the second EMA. Defaults to 3.
            signal_window (int, optional): The number of periods of the SMA subtracted from the
                smoothed RSI. Defaults to 9.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Derivative Oscillator.
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
            pd.Series or pd.DataFrame: Derivative Oscillator values, in RSI points around zero.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Derivative Oscillator for each asset in the Toolkit instance.
        - The Derivative Oscillator agrees with TA-Lib's DOSC and starts `window` + `first_window` +
          `second_window` + `signal_window` - 3 periods after the first price.
        - When growth is requested, the method calculates the growth of the Derivative Oscillator
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_derivative_oscillator().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -4.319  | 3.3478 |      1.4327 |
        | 2025-12-26 | -2.8918 | 3.7216 |      3.1297 |
        | 2025-12-29 | -1.5792 | 3.6241 |      3.7512 |
        | 2025-12-30 | -0.7486 | 3.2821 |      3.6025 |
        | 2025-12-31 | -0.6816 | 2.1574 |      2.0585 |
        """
        self._check_arguments(
            period,
            close_column,
            window=window,
            first_window=first_window,
            second_window=second_window,
            signal_window=signal_window,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        derivative_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_derivative_oscillator,
            window,
            first_window,
            second_window,
            signal_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=derivative_oscillator,
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
    def get_swing_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        limit_move: float = 3.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Wilder's Swing Index and Accumulative Swing Index: the strength of each period's move compared
        with the previous period, and its running total, which shows the real direction of the prices.

        The Swing Index of Wilder (1978) weighs the change of the close against the previous
        period's open and close and against the range of the current period, to measure the real
        direction of a period's move. Wilder accumulates it into the Accumulative Swing Index, a
        line that follows the price but filters out the moves that the open, high and low do not
        confirm: a breakout of the ASI through a previous high or low swing point confirms a
        breakout of the price.

        The formula is as follows:

        - A = |High — Close(t - 1)|, B = |Low — Close(t - 1)|, C = |High — Low|, D = |Close(t - 1) — Open(t - 1)|
        - R = A — 0.5 * B + 0.25 * D when A is the largest of A, B and C
        - R = B — 0.5 * A + 0.25 * D when B is the largest, else R = C + 0.25 * D
        - K = max(A, B)
        - N = (Close — Close(t - 1)) + 0.5 * (Close — Open) + 0.25 * (Close(t - 1) — Open(t - 1))
        - Swing Index = 50 * (N / R) * (K / limit_move)
        - Accumulative Swing Index = Sum of the Swing Index since the first period

        The limit move is the largest move a futures contract may make in one period. A stock has
        none, so `limit_move` is a scale in price units: the Swing Index stays within -100 and
        100 only for moves within it, and only the shape of the ASI, not its level, carries
        meaning.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research,
          section VIII "The Swing Index."

        Referenced against:
        - TA-Lib, SI: https://ta-lib.org/functions/si.html
        - TA-Lib, ASI: https://ta-lib.org/functions/asi.html

        Also known as: SI, ASI, Wilder Swing Index, Accumulation Swing Index.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            limit_move (float, optional): The limit move that scales the index, in price units.
                Defaults to 3.0, TA-Lib's default.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Swing Index and Accumulative Swing Index.
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
            pd.Series or pd.DataFrame: Swing Index and Accumulative Swing Index values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Swing Index and Accumulative Swing Index for each asset in the Toolkit instance.
        - The Swing Index agrees with TA-Lib's SI. The first period has no previous period and no Swing Index, so
          the Accumulative Swing Index starts on the second period, where it agrees with TA-Lib's ASI,
          which gives the first period 0. The ASI accumulates from the first date of the Toolkit's
          data.
        - A period without any move (R equal to 0) has no Swing Index (NaN), where TA-Lib uses 0; the ASI
          carries on over it. Neither is rounded, where Wilder rounded to whole numbers by hand.
        - When growth is requested, the method calculates the growth of the Swing Index and Accumulative Swing Index
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_swing_index().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Accumulative Swing Index |   Swing Index |
        |:-----------|---------------------------:|--------------:|
        | 2025-12-24 |                    3959.09 |       36.229  |
        | 2025-12-26 |                    3955.29 |       -3.8043 |
        | 2025-12-29 |                    3960.88 |        5.5885 |
        | 2025-12-30 |                    3957.58 |       -3.2993 |
        | 2025-12-31 |                    3938.2  |      -19.3803 |
        """
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        swing_index = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            momentum_model.get_swing_index,
            limit_move,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=swing_index,
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
    def get_quantitative_qualitative_estimation(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        smooth_window: int = 5,
        multiplier: float = 4.236,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Quantitative Qualitative Estimation (QQE): a smoothed Relative Strength Index with a trailing
        stop line a multiple of its own average move away, whose crossings show the turns of a trend.

        The Quantitative Qualitative Estimation smooths the Relative Strength Index with an
        exponential average (the fast line) and trails it with a stop line (the QQE, the slow
        line), placed a multiple of the RSI's own average move away, in the way an Average True
        Range stop trails the price. The stop only moves towards the RSI while the RSI moves
        away from it and jumps to the other side when the RSI crosses it, so the RSI crossing
        the QQE line marks a change of trend. The QQE is a community indicator of the MetaTrader
        platform without a published primary source.

        The formula is as follows:

        - RSI Moving Average = EMA(RSI(Close, window), smooth_window)
        - Average Move = EMA(EMA(|RSI Moving Average(t) — RSI Moving Average(t - 1)|, 2 * window - 1), 2 * window - 1)
        - Distance = multiplier * Average Move
        - QQE(t) = max(QQE(t - 1), RSI Moving Average(t) — Distance(t)) while the RSI Moving Average stays above QQE
        - QQE(t) = min(QQE(t - 1), RSI Moving Average(t) + Distance(t)) while it stays below QQE
        - QQE(t) = RSI Moving Average(t) -/+ Distance(t) on the period it crosses QQE

        The EMA over 2 * window - 1 periods has the smoothing constant 1 / window of Wilder's
        average. The common parameters are 14, 5 and 4.236.

        Also known as: QQE, QQE fast and slow line.

        Referenced against:
        - pandas-ta-classic, qqe: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/qqe.py#L71

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the RSI. Defaults to 14.
            smooth_window (int, optional): The number of periods of the EMA of the RSI. Defaults
                to 5.
            multiplier (float, optional): The multiple of the average move of the RSI Moving Average
                that sets the distance of the QQE line. Defaults to 4.236.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the QQE and RSI Moving Average.
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
            pd.Series or pd.DataFrame: QQE and RSI Moving Average values, in RSI points.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the QQE and RSI Moving Average for each asset in the Toolkit instance.
        - The RSI is that of `get_relative_strength_index` and every EMA is seeded with the simple average of
          its first values with an empty warm-up, so the RSI Moving Average starts `window` +
          `smooth_window` - 1 periods after the first price and the QQE 2 * (2 * `window` - 1) - 1
          periods later, at the RSI Moving Average minus the distance.
        - The RSI Moving Average and QQE agree with pandas-ta-classic's qqe, except that its line switches sides
          when the RSI Moving Average crosses the band on the other side, which can keep it on the
          old side for a few more periods (3 of 757 days on the test data); this follows the
          original MetaTrader indicator, which switches on the period of the crossing.
        - When growth is requested, the method calculates the growth of the QQE and RSI Moving Average
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_quantitative_qualitative_estimation().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |     QQE |   RSI Moving Average |
        |:-----------|--------:|---------------------:|
        | 2025-12-24 | 53.9827 |              48.2126 |
        | 2025-12-26 | 53.9827 |              48.5314 |
        | 2025-12-29 | 53.9827 |              49.0098 |
        | 2025-12-30 | 53.9827 |              48.8135 |
        | 2025-12-31 | 53.9827 |              47.7724 |
        """
        self._check_arguments(
            period, close_column, window=window, smooth_window=smooth_window
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        quantitative_qualitative_estimation = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_quantitative_qualitative_estimation,
            window,
            smooth_window,
            multiplier,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=quantitative_qualitative_estimation,
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
    def get_price_zone_oscillator(
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
        Calculate the Price Zone Oscillator (PZO): the share of the recent closes that moved up, from -100 to 100,
        which shows overbought and oversold levels and the direction of a trend.

        The Price Zone Oscillator of Khalil and Steckler (2011) counts each close as positive
        when it closed above the previous close and as negative when it closed below, and
        divides the EMA of these signed closes by the EMA of the closes. It is thereby the share
        of the recent price action that went up, on a scale of -100 to 100: above 40 is
        overbought and below -40 oversold in a trendless market, while in a trend Khalil and
        Steckler use the crossings of zero, 15 and -5.

        The formula is as follows:

        - Signed Close = sign(Close(t) — Close(t - 1)) * Close(t)
        - PZO = 100 * EMA(Signed Close, window) / EMA(Close, window)

        Both EMAs run over the periods with a change, so they start on the same period.

        For more information, see the following sources:

        - Khalil, W. & Steckler, D. (2011). "Entering The Price Zone." Technical Analysis of
          Stocks & Commodities, 29(6), 28-35.

        Shares similar logic:
        - finta, PZO: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1542

        Also known as: PZO.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the EMAs. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the PZO.
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
            pd.Series or pd.DataFrame: PZO values, between -100 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the PZO for each asset in the Toolkit instance.
        - Every EMA is seeded with the simple average of its first values and is missing before it, so the PZO
          starts `window` periods after the first price. finta's PZO uses averages without a seed
          and equals it after the first values.
        - When growth is requested, the method calculates the growth of the PZO
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_price_zone_oscillator().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 23.8305 | 30.886  |     41.0541 |
        | 2025-12-26 |  7.3734 | 13.3243 |     22.033  |
        | 2025-12-29 | 19.7029 | -1.8518 |      5.6525 |
        | 2025-12-30 |  3.8001 | 11.7892 |     -8.5015 |
        | 2025-12-31 | -9.9432 | -3.071  |    -20.6729 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        price_zone_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_price_zone_oscillator,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=price_zone_oscillator,
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
    def get_inertia(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        rvi_window: int = 14,
        std_window: int = 10,
        method: Literal["original", "refined"] = "original",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Dorsey's Inertia: the Relative Volatility Index smoothed with a linear regression, which shows the
        direction of a trend, up above 50 and down below it.

        Inertia is Dorsey's Relative Volatility Index (RVI) smoothed with a linear regression: the end
        point of the least-squares line through the last 20 RVI values. Dorsey regarded a trend as
        momentum that persists, inertia, with its direction given by the direction of volatility, so
        Inertia above 50 marks an up trend and below 50 a down trend. The regression smooths the RVI
        with less lag than a moving average.

        The formula is as follows:

        - RVI = Relative Volatility Index(Close, rvi_window, std_window), or for the refined RVI
          (RVI(High) + RVI(Low)) / 2
        - Inertia = Linear Regression(RVI, window), the end point of the least-squares line

        with the RVI of `get_relative_volatility_index` (Wilder's smoothing of the standard deviation
        on up and down periods) and the linear regression of `get_linear_regression`.

        For more information, see the following sources:

        - Dorsey, D. (1995). "Refining the Relative Volatility Index." Technical Analysis of Stocks &
          Commodities, 13(9).
        - Dorsey, D. (1993). "The Relative Volatility Index." Technical Analysis of Stocks &
          Commodities, 11(6).

        Referenced against:
        - pandas-ta-classic, inertia: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/inertia.py#L61

        Also known as: INERTIA, Dorsey Inertia.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the linear regression. Defaults to 20.
            rvi_window (int, optional): The number of periods of Wilder's smoothing in the RVI.
                Defaults to 14.
            std_window (int, optional): The number of periods of the standard deviation in the RVI.
                Defaults to 10.
            method (str, optional): The Relative Volatility Index it uses. Options are: "original" (the RVI of
                the close), "refined" (the average of the RVI of the highs and of the lows; Dorsey, 1995).
                Defaults to "original".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Inertia.
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
            pd.Series or pd.DataFrame: Inertia values, around 0 to 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Inertia for each asset in the Toolkit instance.
        - Equals the linear regression of `get_relative_volatility_index` with the same parameters,
          and pandas-ta-classic's inertia with Wilder's smoothing and `std_window` equal to `rvi_window`. By
          default pandas-ta-classic smooths its RVI with an EMA and takes the standard deviation over
          the RVI window. The linear regression can overshoot the 0 to 100 range of the RVI.
        - When growth is requested, the method calculates the growth of the Inertia
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_inertia().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 47.9699 | 57.9156 |     67.4073 |
        | 2025-12-26 | 48.2486 | 57.5311 |     66.9018 |
        | 2025-12-29 | 49.2247 | 56.9374 |     65.6764 |
        | 2025-12-30 | 50.4639 | 56.4007 |     62.9555 |
        | 2025-12-31 | 52.0466 | 55.6652 |     59.8904 |
        """
        self._check_arguments(
            period,
            close_column,
            window=window,
            rvi_window=rvi_window,
            std_window=std_window,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        inertia = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_inertia,
            window,
            rvi_window,
            std_window,
            method,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=inertia,
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
    def get_bias(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 26,
        moving_average: str = "simple",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Bias: the distance of the price from its moving average, as a share of that average.

        The Bias, or deviation rate, is a standard indicator on Asian trading platforms. It measures how
        far the close has run from its moving average as a share of that average: a price far above
        its average is read as overbought and likely to return to it, and far below as oversold. What
        counts as far depends on the security and the window, so it is usually compared with the
        Bias's own history.

        The formula is as follows:

        - Bias = Close / MA(Close, window) — 1

        with MA the Simple Moving Average by default.

        Also known as: BIAS, deviation rate, BIAS ratio, distance to the moving average.

        Referenced against:
        - pandas-ta-classic, bias: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/bias.py#L12

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the moving average. Defaults to 26.
            moving_average (str, optional): The moving average. Options are: "simple" (Simple Moving Average),
                "exponential" (Exponential Moving Average), "double_exponential" (Double Exponential Moving
                Average), "triangular" (Triangular Moving Average), "weighted" (Weighted Moving Average),
                "hull" (Hull Moving Average), "kaufman_adaptive" (Kaufman Adaptive Moving Average). Defaults to
                "simple".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Bias.
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
            pd.Series or pd.DataFrame: Bias values as decimals (0.05 is 5% above the average).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Bias for each asset in the Toolkit instance.
        - Equals pandas-ta-classic's bias, also a decimal; some platforms multiply it by 100.
        - When growth is requested, the method calculates the growth of the Bias
          over the given lag.
        - The indicator has no single primary source; it is a standard indicator on Asian trading platforms, defined
          here as in pandas-ta-classic.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_bias().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -0.0071 | 0.0105 |      0.0205 |
        | 2025-12-26 | -0.0094 | 0.0102 |      0.0185 |
        | 2025-12-29 | -0.0088 | 0.0089 |      0.0133 |
        | 2025-12-30 | -0.0122 | 0.009  |      0.01   |
        | 2025-12-31 | -0.0166 | 0      |      0.0011 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        bias = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_bias,
            window,
            moving_average,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=bias,
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
    def get_brar(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 26,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the AR and BR sentiment indicators (BRAR), which show how popular a security is and how willing
        the market is to buy it.

        They set how far the prices rose against how far they fell, measured from the open (AR) and from the
        previous close (BR).

        AR (popularity) compares how far the price rose above the open with how far it fell below it
        over the window, and BR (willingness) does the same against the previous close, which also
        counts the gaps between periods. Both are 100 when the upward and downward moves balance. AR
        from 150 and BR from 300 are read as overheated and AR below 50 and BR below 40 as depressed;
        BR crossing AR from below is a buy signal.

        The formula is as follows:

        - AR = 100 × Sum(High — Open, window) / Sum(Open — Low, window)
        - BR = 100 × Sum(Max(High — Close(t — 1), 0), window) / Sum(Max(Close(t — 1) — Low, 0), window)

        Also known as: BRAR, AR and BR, popularity and willingness indicators, sentiment indicator.

        Referenced against:
        - pandas-ta-classic, brar: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/brar.py#L18

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the sums. Defaults to 26.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the AR and BR.
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
            pd.Series or pd.DataFrame: AR and BR, 100 for a balance.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the AR and BR for each asset in the Toolkit instance.
        - Equals pandas-ta-classic's brar, on its scale of 100 for a balance. A window without any
          downward move has no AR or BR (missing) where pandas-ta-classic gives an infinite value.
        - When growth is requested, the method calculates the growth of the AR and BR
          over the given lag.
        - The indicator has no single primary source; it is a standard indicator on Asian trading platforms, defined
          here as in pandas-ta-classic.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_brar().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |       AR |       BR |
        |:-----------|---------:|---------:|
        | 2025-12-24 | 112.898  | 120.363  |
        | 2025-12-26 | 120.492  | 119.908  |
        | 2025-12-29 | 110.988  | 114.995  |
        | 2025-12-30 | 113.905  | 105.452  |
        | 2025-12-31 |  99.0846 |  92.3299 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        brar = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            momentum_model.get_brar,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=brar,
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
    def get_pretty_good_oscillator(
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
        Calculate Johnson's Pretty Good Oscillator (PGO): how far the close lies from its moving average, in units
        of the Average True Range, which shows breakouts from a range.

        The Pretty Good Oscillator of Mark Johnson measures the distance of the close from its Simple
        Moving Average in units of the average true range, so the distance compares across securities
        and periods of different volatility. Johnson used it for breakouts: a close more than 3 average
        ranges above its average is a long entry, more than 3 below a short entry, and the trade is
        closed when the oscillator returns to zero.

        The formula is as follows:

        - PGO = (Close — SMA(Close, window)) / EMA(ATR(window), window)

        with ATR Wilder's Average True Range and the EMA seeded with a simple average.

        For more information, see the following sources:

        - Johnson, M. "Pretty Good Oscillator." FM Labs technical analysis library
          (https://www.fmlabs.com/reference/default.htm?url=PGO.htm).

        Referenced against:
        - pandas-ta-classic, pgo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/pgo.py#L14

        Also known as: PGO, Pretty Good Oscillator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the SMA, the ATR and its EMA. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the PGO.
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
            pd.Series or pd.DataFrame: PGO values, in average true ranges.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the PGO for each asset in the Toolkit instance.
        - Equals pandas-ta-classic's pgo once Wilder's smoothing of the ATR has forgotten its seed (within
          1e-7 after 200 periods): the Toolkit's ATR includes the High — Low of the first period, as
          StockCharts does. A period without any range has no PGO (missing) where pandas-ta-classic
          gives 0.
        - When growth is requested, the method calculates the growth of the PGO
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_pretty_good_oscillator().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 | -0.2751 | 0.5238 |      1.1828 |
        | 2025-12-26 | -0.284  | 0.466  |      1.1293 |
        | 2025-12-29 | -0.1531 | 0.4401 |      0.7459 |
        | 2025-12-30 | -0.2379 | 0.5319 |      0.5822 |
        | 2025-12-31 | -0.3998 | 0.0483 |     -0.1073 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        pretty_good_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_pretty_good_oscillator,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=pretty_good_oscillator,
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
    def get_psychological_line(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 12,
        method: Literal["close", "open"] = "close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Psychological Line (PSL): the share of periods in a window that closed higher, which shows
        whether buyers or sellers have the upper hand.

        The Psychological Line is the percentage of the last `window` periods that closed higher, a
        measure of the mood of the market: after many periods up in a row buyers may be exhausted
        (overbought above 75) and after many down sellers (oversold below 25). The open-based form
        counts the periods that closed above their own open instead.

        The formula is as follows:

        - PSL = 100 × Count(Close(t) > Close(t — 1), window) / window

        or in the open-based form, 100 × Count(Close > Open, window) / window.

        Also known as: PSL, PSY, psychological line, psychological index.

        Referenced against:
        - pandas-ta-classic, psl: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/psl.py#L19

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods counted. Defaults to 12.
            method (str, optional): What counts as an up period. Options are: "close" (a close above the
                previous close), "open" (a close above the period's open). Defaults to "close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the PSL.
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
            pd.Series or pd.DataFrame: PSL values, between 0 and 100.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the PSL for each asset in the Toolkit instance.
        - Equals pandas-ta-classic's psl, also on the scale of 0 to 100.
        - When growth is requested, the method calculates the growth of the PSL
          over the given lag.
        - The indicator has no single primary source; it is a standard indicator on Asian trading platforms, defined
          here as in pandas-ta-classic.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_psychological_line().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 58.3333 | 58.3333 |     58.3333 |
        | 2025-12-26 | 58.3333 | 50      |     58.3333 |
        | 2025-12-29 | 58.3333 | 50      |     50      |
        | 2025-12-30 | 58.3333 | 50      |     41.6667 |
        | 2025-12-31 | 50      | 50      |     41.6667 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        psychological_line = calculate_per_ticker(
            historical_data,
            ["Open", close_column],
            momentum_model.get_psychological_line,
            window,
            method,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=psychological_line,
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
    def get_volatility_based_momentum(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 12,
        atr_window: int = 26,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Volatility-Based Momentum (VBM): the change of the price over a window in units of the Average
        True Range, which compares momentum across tickers of different volatility.

        Volatility-Based Momentum is the change in the close over `window` periods in units of the
        Average True Range, instead of as a share of the earlier close as in the Rate of Change. A move
        of 3 average ranges is then large for any security, whatever its price or volatility, which
        makes the momentum comparable across securities and over time.

        The formula is as follows:

        - VBM = (Close(t) — Close(t — window)) / ATR(atr_window)

        with ATR Wilder's Average True Range.

        For more information, see the following sources:

        - Pring, M.J. (1991). "Technical Analysis Explained." 3rd ed. McGraw-Hill (momentum).
        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research
          (Average True Range).

        Shares similar logic:
        - finta, VBM: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L664

        Also known as: VBM, volatility-based momentum, volatility adjusted momentum.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the price change. Defaults to 12.
            atr_window (int, optional): The number of periods of the Average True Range. Defaults to 26.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the VBM.
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
            pd.Series or pd.DataFrame: VBM values, in average true ranges.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the VBM for each asset in the Toolkit instance.
        - finta documents this formula, but its VBM subtracts the earlier close twice
          (Close.diff(n) — Close.shift(n)) and averages the true range with a simple moving average,
          so its values differ.
        - When growth is requested, the method calculates the growth of the VBM
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_volatility_based_momentum().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | -0.8397 | -0.3403 |      1.236  |
        | 2025-12-26 | -0.7927 | -0.5035 |      1.3433 |
        | 2025-12-29 | -1.0766 |  1.0178 |      0.3387 |
        | 2025-12-30 | -1.0873 |  0.4873 |     -0.0195 |
        | 2025-12-31 | -1.4421 |  0.6285 |      0.3294 |
        """
        self._check_arguments(
            period, close_column, window=window, atr_window=atr_window
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        volatility_based_momentum = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_volatility_based_momentum,
            window,
            atr_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=volatility_based_momentum,
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
    def get_archer_moving_average_trends(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 8,
        long_window: int = 21,
        lookback: int = 2,
        moving_average: str = "exponential",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Archer's Moving Averages Trends (AMAT): whether a fast and a slow moving average both rise (a long
        run) or both fall (a short run), which shows the direction of a trend.

        AMAT flags a long run when the short moving average rose over the last `lookback` periods while
        the long one moved (a rising long average confirms the up trend, a falling one marks a potential
        bottom), and a short run when the short average fell while the long one moved (a potential top
        when the long one still rose). Both flags are 0 when the short average is flat or the long one
        did not move.

        The formula is as follows:

        - Long Run = 1 if MA(Close, short_window) rose over `lookback` periods and MA(Close,
          long_window) changed, else 0
        - Short Run = 1 if MA(Close, short_window) fell over `lookback` periods and MA(Close,
          long_window) changed, else 0

        with MA the exponential moving average by default.

        For more information, see the following sources:

        - Archer's TradingView script "Archer Moving Averages Trends", which defines the indicator; there
          is no published primary source.

        Referenced against:
        - pandas-ta-classic, amat: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/amat.py#L15

        Also known as: AMAT, Archer Moving Averages Trends.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): The number of periods of the short moving average. Defaults to 8.
            long_window (int, optional): The number of periods of the long moving average. Defaults to 21.
            lookback (int, optional): The number of periods over which the averages rise or fall.
                Defaults to 2.
            moving_average (str, optional): The moving average. Options are: "simple" (Simple Moving Average),
                "exponential" (Exponential Moving Average), "double_exponential" (Double Exponential Moving
                Average), "triangular" (Triangular Moving Average), "weighted" (Weighted Moving Average),
                "hull" (Hull Moving Average), "kaufman_adaptive" (Kaufman Adaptive Moving Average). Defaults to
                "exponential".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the AMAT.
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
            pd.Series or pd.DataFrame: Long Run and Short Run, 1 when the run holds and 0 otherwise.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the AMAT for each asset in the Toolkit instance.
        - Equals pandas-ta-classic's amat (its long_run and short_run columns).
        - When growth is requested, the method calculates the growth of the AMAT
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_archer_moving_average_trends().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Long Run |   Short Run |
        |:-----------|-----------:|------------:|
        | 2025-12-24 |          0 |           1 |
        | 2025-12-26 |          1 |           0 |
        | 2025-12-29 |          1 |           0 |
        | 2025-12-30 |          0 |           1 |
        | 2025-12-31 |          0 |           1 |
        """
        self._check_arguments(
            period,
            close_column,
            short_window=short_window,
            long_window=long_window,
            lookback=lookback,
        )
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        archer_moving_average_trends = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_archer_moving_average_trends,
            short_window,
            long_window,
            lookback,
            moving_average,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=archer_moving_average_trends,
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
    def get_ttm_trend(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 6,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Carter's TTM Trend: whether the close lies above the average midpoint of the previous bars (1) or
        not (-1), which shows the direction of the trend.

        The TTM Trend of John Carter colours each bar by the short-term trend: up (1) when the close is
        above the average midpoint, (High + Low) / 2, of the previous six bars, and down (-1) when it is
        not. Carter uses it to stay in a trade while the bars keep their colour and to avoid trading
        against it; two bars of the opposite colour in a row signal that the trend has turned.

        The formula is as follows:

        - Average Midpoint = SMA((High + Low) / 2, window) over the `window` bars before the current one
        - TTM Trend = 1 if Close > Average Midpoint, else -1

        For more information, see the following sources:

        - Carter, J.F. (2005). "Mastering the Trade: Proven Techniques for Profiting from Intraday
          and Swing Trading Setups." McGraw-Hill.

        Referenced against:
        - pandas-ta-classic, ttm_trend: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/ttm_trend.py#L12

        Also known as: TTM Trend, TTM_TRND.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of earlier bars of the average midpoint. Defaults to 6.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the TTM Trend.
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
            pd.Series or pd.DataFrame: 1 for an up trend and -1 for a down trend.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the TTM Trend for each asset in the Toolkit instance.
        - Equals pandas-ta-classic's ttm_trend. It starts `window` periods after the first price.
        - When growth is requested, the method calculates the growth of the TTM Trend
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_ttm_trend().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 |      1 |      1 |           1 |
        | 2025-12-26 |      1 |      1 |           1 |
        | 2025-12-29 |      1 |      1 |           1 |
        | 2025-12-30 |      1 |      1 |           1 |
        | 2025-12-31 |     -1 |     -1 |          -1 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        ttm_trend = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_ttm_trend,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=ttm_trend,
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
    def get_correlation_trend_indicator(
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
        Calculate the Correlation Trend Indicator (CTI) of a price series: the Pearson correlation of the prices
        over the window with a straight line rising over time.

        Ehlers (2020) measures the trend by how closely the prices follow an upward-sloping
        line: +1 when they rise in a straight line, -1 when they fall in one, and around 0 when
        they move sideways or in a cycle. Ehlers reads values above 0 as an uptrend and below 0
        as a downtrend, with the indicator crossing zero as the switch between them. Unlike the
        slope it is bounded and does not depend on the price level, so it compares across
        securities.

        The formula is as follows:

        - CTI = Correlation(Close, x, window), with time x = 0, ..., window - 1 from the oldest
          price of the window to the latest

        For more information, see the following sources:

        - Ehlers, J.F. (2020). "Correlation As A Trend Indicator." Technical Analysis of Stocks &
          Commodities, 38(5), 24-27.

        Referenced against:
        - TA-Lib, CTI: https://ta-lib.org/functions/cti.html
        - pandas-ta-classic, cti: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/cti.py#L12

        Also known as: CTI, correlation trend, Ehlers correlation trend indicator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods, at least 2. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the CTI.
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
            Correlation Trend Indicator values between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          CTI for each asset in the Toolkit instance.
        - The CTI agrees with TA-Lib's CTI and pandas-ta-classic's cti (whose default window is 12).
        - A window of flat prices has no correlation and is missing, where TA-Lib gives 0.
        - When growth is requested, the method calculates the growth of the CTI
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_correlation_trend_indicator().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | -0.8202 | -0.2102 |      0.1879 |
        | 2025-12-26 | -0.8697 | -0.1202 |      0.2628 |
        | 2025-12-29 | -0.8843 |  0.0683 |      0.3433 |
        | 2025-12-30 | -0.8571 |  0.1828 |      0.3518 |
        | 2025-12-31 | -0.8485 |  0.3044 |      0.2916 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        correlation_trend_indicator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_correlation_trend_indicator,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=correlation_trend_indicator,
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
    def get_forecast_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 5,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Forecast Oscillator of a price series: how far the price lies from the Time Series Forecast
        made for it one period earlier, as a share of the price.

        Chande's Forecast Oscillator measures the error of the linear regression forecast: it
        is positive when the price comes in above the forecast of the previous period and
        negative when below it. A forecast that keeps being too low points to a trend that
        accelerates, and the oscillator crossing zero to a change in the direction of the
        error.

        The formula is as follows:

        - Time Series Forecast(t - 1) = the least-squares line over the `window` periods up to
          t - 1, extended to t
        - Forecast Oscillator = (Close - Time Series Forecast(t - 1)) / Close

        For more information, see the following sources:

        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

        Referenced against:
        - TA-Lib, FOSC: https://ta-lib.org/functions/fosc.html
        - Tulip Indicators, fosc: https://tulipindicators.org/fosc

        Shares similar logic:
        - pandas-ta-classic, fosc: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/fosc.py#L12
        - pandas-ta-classic, cfo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/cfo.py#L17

        Also known as: FOSC, Chande Forecast Oscillator, CFO, forecast error.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods the forecast line is fitted over, at
                least 2. Defaults to 5.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Forecast Oscillator.
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
            Forecast Oscillator values as decimals.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Forecast Oscillator for each asset in the Toolkit instance.
        - The oscillator agrees with TA-Lib's FOSC and Tulip Indicators' fosc divided by 100: a decimal (0.01 is 1%)
          where they give a percentage.
        - pandas-ta-classic's fosc and cfo compare the price with the forecast of the same period, a line fitted
          through that price, rather than with the forecast made the period before.
        - When growth is requested, the method calculates the growth of the Forecast Oscillator
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_forecast_oscillator().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 |  0.0061 | -0.0046 |     -0.0048 |
        | 2025-12-26 |  0.0008 | -0.0019 |     -0.0074 |
        | 2025-12-29 |  0.0008 | -0.0033 |     -0.0094 |
        | 2025-12-30 | -0.0065 | -0.0021 |     -0.0055 |
        | 2025-12-31 | -0.0068 | -0.0081 |     -0.008  |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        forecast_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_forecast_oscillator,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=forecast_oscillator,
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
    def get_projection_oscillator(
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
        Calculate the Projection Oscillator of a price series: where the close lies within Widner's Projection
        Bands, 0 on the lower band and 1 on the upper band.

        Widner (1995) projects every high and low of the window forward to the latest period
        along the least-squares slope of the highs and of the lows over the window. The upper
        Projection Band is the highest of the projected highs and the lower band the lowest of
        the projected lows, so the bands follow the trend of the window rather than lying flat
        like the highest high and lowest low of the Stochastic Oscillator. The Projection
        Oscillator is the stochastic within those bands: high readings mean the close is near
        the top of its trend channel, and Widner reads turns from extreme readings as signals.

        The formula is as follows:

        - Slope High, Slope Low = the least-squares slopes of the highs and of the lows over the window
        - Upper Band = Max(High(t - i) + i * Slope High), i = 0, ..., window - 1
        - Lower Band = Min(Low(t - i) + i * Slope Low), i = 0, ..., window - 1
        - Projection Oscillator = (Close - Lower Band) / (Upper Band - Lower Band)

        For more information, see the following sources:

        - Widner, M. (1995). "Signaling Change with Projection Bands." Technical Analysis of
          Stocks & Commodities, 13(7), 275-281.

        Shares similar logic:
        - pandas-ta-classic, po: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/po.py#L12

        Also known as: PO, Projection Oscillator, Projection Bands, Widner.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods, at least 2. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Projection Oscillator.
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
            Projection Oscillator values as decimals.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Projection Oscillator for each asset in the Toolkit instance.
        - Widner gives the oscillator from 0 to 100; the Toolkit gives a decimal from 0 to 1.
        - pandas-ta-classic's po is a simpler measure under the same name, the distance of the close from the
          end point of its linear regression in percent, 100 * (Close - Linear Regression) / Linear Regression,
          which follows from `get_linear_regression`.
        - With the dividend-adjusted close (the default), the high and low are adjusted by the same factor, so
          the close stays within the high and low of its period.
        - When growth is requested, the method calculates the growth of the Projection Oscillator
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_projection_oscillator().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.8331 | 0.7613 |      0.9788 |
        | 2025-12-26 | 0.7973 | 0.7345 |      0.9305 |
        | 2025-12-29 | 0.854  | 0.5524 |      0.7633 |
        | 2025-12-30 | 0.7814 | 0.5129 |      0.6374 |
        | 2025-12-31 | 0.6255 | 0.1683 |      0.2356 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        projection_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            momentum_model.get_projection_oscillator,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=projection_oscillator,
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
    def get_trend_detection_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        multiple: int = 2,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate M.H. Pee's Trend Detection Index (TDI) and its Direction Indicator: how strongly the momentum
        keeps one sign over a window, which shows whether the prices trend and in which direction.

        The Trend Detection Index tells a trending market from a consolidating one. It compares
        the net momentum over the window with how much the momentum moved over a longer window:
        when the momentum keeps one sign, its sum outweighs the absolute momentum of the earlier
        periods and the TDI is positive (a trend); when it changes sign, the sum cancels out and
        the TDI is negative (a consolidation). The Direction Indicator, the sum of the momentum,
        says which way the trend goes.

        The formula is as follows:

        - Momentum = Close(t) — Close(t — window)
        - Direction Indicator = Sum(Momentum, window)
        - TDI = |Direction Indicator| — (Sum(|Momentum|, multiple * window) — Sum(|Momentum|, window))

        Pee buys when both the TDI and the Direction Indicator are positive, and sells when the
        TDI is positive and the Direction Indicator negative. A negative TDI says to stay out, or
        to trade the range. Both are in price units, so they compare over time but not across
        stocks with different prices.

        For more information, see the following sources:

        - Pee, M.H. (2001). "Trend Detection Index." Technical Analysis of Stocks & Commodities,
          19(10).

        Referenced against:
        - TTR (R), TDI: https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/TDI.R#L63

        Also known as: TDI, trend detection index, Pee's trend detection index.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods of the momentum and of the Direction
                Indicator. Defaults to 20, as in TTR's TDI.
            multiple (int, optional): The longer window of the absolute momentum, as a multiple of
                `window`. Defaults to 2, Pee's own.
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
            pd.Series or pd.DataFrame: The TDI and the Direction Indicator, in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Trend Detection Index for each asset in the Toolkit instance.
        - Every sum covers momentum values only, so the TDI starts on period
          (`multiple` + 1) * `window` — 1 and the Direction Indicator on period 2 * `window` — 1.
          TTR's TDI sets the first `window` momentum values, which have no earlier price, to 0, so
          both start `window` periods earlier there, with sums that include those zeros; from
          then on the two agree.
        - When growth is requested, the method calculates the growth of the Trend Detection
          Index over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_trend_detection_index().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Direction Indicator |      TDI |
        |:-----------|----------------------:|---------:|
        | 2025-12-24 |              128.144  | -160.945 |
        | 2025-12-26 |              115.926  | -167.06  |
        | 2025-12-29 |              103.167  | -173.298 |
        | 2025-12-30 |               80.2354 | -196.881 |
        | 2025-12-31 |               48.615  | -233.537 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        trend_detection_index = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_trend_detection_index,
            window,
            multiple,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=trend_detection_index,
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
    def get_dv_intermediate_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 252,
        magnitude_weight: float = 0.8,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate David Varadi's DV Intermediate Oscillator (DVI), from 0 to 1, which shows how stretched the price
        is.

        It blends rankings of how far the price stands from its short average and of the balance of up and down
        days.

        The DVI is a smooth oscillator between 0 and 1 that combines two rankings over the last
        `window` periods. The magnitude ranks how far the price stands above or below its
        3-period average, smoothed over a short and a long window; the stretch ranks the number
        of up days against down days over a short and a long window. Varadi weights the magnitude
        0.8 and the stretch 0.2.

        The formula is as follows:

        - Return = Close(t) / SMA(Close, 3) — 1
        - Magnitude = SMA((SMA(Return, 5) + SMA(Return, 100) / 10) / 2, 5)
        - Direction = +1 if Close(t) > Close(t — 1), else —1
        - Stretch = SMA((Sum(Direction, 10) + Sum(Direction, 100) / 10) / 2, 2)
        - DVI Magnitude = Percent Rank(Magnitude, window)
        - DVI Stretch = Percent Rank(Stretch, window)
        - DVI = magnitude_weight * DVI Magnitude + (1 — magnitude_weight) * DVI Stretch

        The Percent Rank is the share of the previous `window` values that are lower than the
        latest one, as in `get_percent_rank`. A DVI above 0.5 marks a price stretched upwards
        relative to the past year and one below 0.5 a price stretched downwards. Varadi uses it
        for short-term mean reversion: buy below 0.5 and sell above it, preferably in the
        direction of the longer trend.

        For more information, see the following sources:

        - Varadi, D. (2009). "What is the DVI?" CSS Analytics, December 13, 2009
          (https://cssanalytics.wordpress.com/2009/12/13/what-is-the-dvi/).

        Referenced against:
        - TTR (R), DVI: https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/DVI.R#L50

        Also known as: DVI, DV Intermediate, DV indicator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of previous periods of the Percent Ranks. Defaults to
                252, a year of trading days.
            magnitude_weight (float, optional): The weight of the magnitude in the DVI, between 0 and
                1; the stretch gets the rest. Defaults to 0.8, Varadi's own.
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
            pd.Series or pd.DataFrame: The DVI Magnitude, the DVI Stretch and the DVI, as decimals
            between 0 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          DVI for each asset in the Toolkit instance. The first DVI needs `window` + 106 periods,
          so with the default window a Toolkit needs about a year and a half of daily data before
          its start date for a DVI on every date.
        - The Percent Rank lies between 0 (a new low of the window) and 1 (a new high). TTR's DVI
          ranks within a window of `n` values that includes the latest one and counts it, and
          equal values, as lower, so its rank lies between 1/n and 1. Without equal values, TTR's
          magnitude rank with n = `window` + 1 equals (`window` * DVI Magnitude + 1) / (`window` + 1);
          the stretch takes few distinct values, which TTR counts as lower when equal.
        - An unchanged close counts as a down day, as in TTR's DVI.
        - When growth is requested, the method calculates the growth of the DVI over the
          given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_dv_intermediate_oscillator().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |    DVI |   DVI Magnitude |   DVI Stretch |
        |:-----------|-------:|----------------:|--------------:|
        | 2025-12-24 | 0.3921 |          0.3175 |        0.6905 |
        | 2025-12-26 | 0.4524 |          0.3929 |        0.6905 |
        | 2025-12-29 | 0.4778 |          0.4246 |        0.6905 |
        | 2025-12-30 | 0.5198 |          0.4802 |        0.6786 |
        | 2025-12-31 | 0.5111 |          0.5079 |        0.5238 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        dv_intermediate_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            momentum_model.get_dv_intermediate_oscillator,
            window,
            magnitude_weight,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=dv_intermediate_oscillator,
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
