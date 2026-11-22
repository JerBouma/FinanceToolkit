"""Technicals Module"""

__docformat__ = "google"

from collections.abc import Callable
from functools import cached_property

import pandas as pd

from financetoolkit.helpers import get_non_trading_days
from financetoolkit.technicals.breadth.breadth_controller import BreadthTechnicals
from financetoolkit.technicals.cycle.cycle_controller import CycleTechnicals
from financetoolkit.technicals.helpers import (
    FACTOR_COLLECTIONS,
    FACTOR_PRICE_DIFFERENCES,
    FACTOR_PRICE_LINES,
    PERIODS,
)
from financetoolkit.technicals.momentum.momentum_controller import MomentumTechnicals
from financetoolkit.technicals.overlap.overlap_controller import OverlapTechnicals
from financetoolkit.technicals.pattern.pattern_controller import PatternTechnicals
from financetoolkit.technicals.seasonality.seasonality_controller import (
    SeasonalityTechnicals,
)
from financetoolkit.technicals.statistics.statistics_controller import (
    StatisticsTechnicals,
)
from financetoolkit.technicals.volatility.volatility_controller import (
    VolatilityTechnicals,
)
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    apply_rounding,
    calculate_growth,
    calculate_standardization,
)

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


@filter_shown_columns
@track_sources
class TechnicalsBase:
    """
    The data every technicals topic shares: the tickers, the historical data per period and the
    close prices, with the collection of all indicators and the factor analysis across topics.
    """

    def __init__(
        self,
        tickers: list[str] | Callable[[], list[str]],
        historical_data: dict[str, pd.DataFrame],
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Technicals Controller Class.

        The Toolkit passes the tickers as a function without arguments and the historical data
        as a LazyDict (financetoolkit.helpers), so an indicator retrieves the data of the period
        it is calculated over when it is first needed, and nothing else.

        Args:
            tickers (list[str] | Callable[[], list[str]]): The tickers to use for the calculation.
            historical_data (dict[str, pd.DataFrame]): The historical data per period.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculation. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculation. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        average_directional_index = toolkit.technicals.get_average_directional_index()
        ```

        Which returns:

        | Date       |    AAPL |    TSLA |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 21.9724 | 24.0777 |     13.6364 |
        | 2025-12-26 | 20.8908 | 24.3229 |     14.1712 |
        | 2025-12-29 | 20.0343 | 23.2159 |     13.912  |
        | 2025-12-30 | 19.2603 | 21.7824 |     13.6713 |
        | 2025-12-31 | 18.7103 | 20.3475 |     12.9966 |
        """
        self._tickers_source = tickers
        self._historical_data = historical_data
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._portfolio_weights: dict | None = None
        # The historical data with the open, high and low adjusted like the close, per period.
        self._consistent_data: dict[tuple, pd.DataFrame] = {}

        # Technical Indicators
        self._all_indicators: pd.DataFrame = pd.DataFrame()
        self._all_indicators_growth: pd.DataFrame = pd.DataFrame()
        self._breadth_indicators: pd.DataFrame = pd.DataFrame()
        self._breadth_indicators_growth: pd.DataFrame = pd.DataFrame()
        self._momentum_indicators: pd.DataFrame = pd.DataFrame()
        self._momentum_indicators_growth: pd.DataFrame = pd.DataFrame()
        self._overlap_indicators: pd.DataFrame = pd.DataFrame()
        self._overlap_indicators_growth: pd.DataFrame = pd.DataFrame()
        self._volatility_indicators: pd.DataFrame = pd.DataFrame()
        self._volatility_indicators_growth: pd.DataFrame = pd.DataFrame()

    @cached_property
    def _tickers(self) -> list[str]:
        """The tickers, retrieved when first needed."""
        return (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

    @handle_errors
    def collect_all_indicators(
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
        Calculates all Technical Indicators based on the data provided.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "daily".
            window (int, optional): The number of periods to use for the calculation. Defaults to 14.
                The Bollinger Bands and their %B and BandWidth keep their own window of 20
                periods (with 2 standard deviations), as defined by Bollinger, and the indicators
                without a single window, such as the MACD, the True Strength Index, the Coppock
                Curve or the Supertrend, their own defaults.
            close_column (str, optional): The column to use for the calculation. Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame: Technical indicators calculated based on the specified parameters.

        Notes:
        - The method calculates various types of technical indicators for each asset in the Toolkit instance.
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

        toolkit.technicals.collect_all_indicators().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   McClellan Oscillator |   Advancers - Decliners |   On-Balance Volume |   Price Volume Trend |   Accumulation/Distribution Line |   Chaikin Oscillator |     TRIN |   New Highs - New Lows |   Chaikin Money Flow |   Ease of Movement |   Negative Volume Index |   Positive Volume Index |   Money Flow Index |   Williams %R |   Aroon Indicator Up |   Aroon Indicator Down |   Commodity Channel Index |   Relative Vigor Index |   Relative Vigor Index Signal Line |       Force Index |   Ultimate Oscillator |   Percentage Price Oscillator |   Percentage Volume Oscillator |   Percentage Volume Oscillator Signal Line |   Percentage Volume Oscillator Histogram |   Detrended Price Oscillator |   Average Directional Index |   Chande Momentum Oscillator |   Ichimoku Conversion Line |   Ichimoku Base Line |   Ichimoku Leading Span A |   Ichimoku Leading Span B |   Stochastic %K |   Stochastic %D |   MACD Line |   MACD Signal Line |   Relative Strength Index |   Stochastic RSI %K |   Stochastic RSI %D |   True Strength Index |   True Strength Index Signal Line |   Balance of Power |   Awesome Oscillator |   Vortex Indicator VI+ |   Vortex Indicator VI- |   Elder Ray Bull Power |   Elder Ray Bear Power |   Rate of Change |   Choppiness Index |   Know Sure Thing |   Know Sure Thing Signal Line |   Coppock Curve |   Simple Moving Average (SMA) |   Exponential Moving Average (EMA) |   Double Exponential Moving Average (DEMA) |   TRIX |   Triangular Moving Average |   Weighted Moving Average (WMA) |   Hull Moving Average (HMA) |   Kaufman Adaptive Moving Average (KAMA) |   Volume Weighted Average Price (VWAP) |   Parabolic SAR |   Pivot Point |   Pivot Point Resistance 1 |   Pivot Point Support 1 |   Fibonacci Retracement 50.0% |   Fibonacci Retracement 61.8% |   Bollinger Band Upper |   Bollinger Band Middle |   Bollinger Band Lower |   Bollinger %B |   Bollinger BandWidth |   True Range |   Average True Range |   Mass Index |   Supertrend |   Supertrend Trend Direction |   Chandelier Exit Long |   Chandelier Exit Short |   Keltner Channel Upper |   Keltner Channel Middle |   Keltner Channel Lower |   Donchian Channel Upper |   Donchian Channel Middle |   Donchian Channel Lower |   Linear Regression |   Linear Regression Slope |   Linear Regression Intercept |   Linear Regression Angle |   Time Series Forecast |   Slope |   Correlation Trend Indicator |   Forecast Oscillator |   Projection Oscillator |   Rolling Median |   Rolling Percentile |   Percent Rank |   Rolling Z-Score |   Rolling Mean Absolute Deviation |   Rolling Standard Error |   Rolling Entropy |   Rolling Correlation |
        |:-----------|-----------------------:|------------------------:|--------------------:|---------------------:|---------------------------------:|---------------------:|---------:|-----------------------:|---------------------:|-------------------:|------------------------:|------------------------:|-------------------:|--------------:|---------------------:|-----------------------:|--------------------------:|-----------------------:|-----------------------------------:|------------------:|----------------------:|------------------------------:|-------------------------------:|-------------------------------------------:|-----------------------------------------:|-----------------------------:|----------------------------:|-----------------------------:|---------------------------:|---------------------:|--------------------------:|--------------------------:|----------------:|----------------:|------------:|-------------------:|--------------------------:|--------------------:|--------------------:|----------------------:|----------------------------------:|-------------------:|---------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------:|-------------------:|------------------:|------------------------------:|----------------:|------------------------------:|-----------------------------------:|-------------------------------------------:|-------:|----------------------------:|--------------------------------:|----------------------------:|-----------------------------------------:|---------------------------------------:|----------------:|--------------:|---------------------------:|------------------------:|------------------------------:|------------------------------:|-----------------------:|------------------------:|-----------------------:|---------------:|----------------------:|-------------:|---------------------:|-------------:|-------------:|-----------------------------:|-----------------------:|------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|-------------------------:|--------------------:|--------------------------:|------------------------------:|--------------------------:|-----------------------:|--------:|------------------------------:|----------------------:|------------------------:|-----------------:|---------------------:|---------------:|------------------:|----------------------------------:|-------------------------:|------------------:|----------------------:|
        | 2025-12-17 |                -0.0981 |                      -1 |         2.26688e+09 |          6.59403e+07 |                      3.99771e+09 |         -1.74849e+07 | nan      |                      0 |              -0.0358 |            -4.7166 |                 2735.54 |                 788.838 |            39.7393 |      -98.8222 |              28.5714 |               100      |                 -139.905  |                -0.2262 |                            -0.0758 |      -3.27416e+07 |               44.5331 |                        0.0063 |                        -0.0329 |                                    -0.0509 |                                   0.018  |                       0.0306 |                     24.6713 |                     -20.6209 |                    275.43  |              276.008 |                   265.041 |                   250.519 |          1.1778 |          8.6605 |      1.7262 |             3.1596 |                   45.1866 |              0      |              1.2037 |               12.3225 |                           19.7487 |            -0.7013 |               1.2653 |                 0.9211 |                 1.0533 |                -0.1328 |                -4.6371 |          -0.0206 |            49.8427 |            0.3638 |                        0.5107 |          0.0029 |                       277.781 |                            275.314 |                                    275.767 | 0.0016 |                     277.982 |                         276.128 |                     273.116 |                                  274.218 |                                277.974 |         285.493 |       273.015 |                    275.185 |                 271.488 |                       279.158 |                       277.161 |                285.548 |                 276     |                266.452 |         0.2327 |                0.0692 |       4.5043 |               4.871  |      24.5798 |      268.465 |                            1 |                272.67  |                 279.347 |                 284.794 |                  275.314 |                 265.833 |                  287.618 |                   279.232 |                  270.846 |             272.822 |                   -0.763  |                       282.741 |                  -37.3449 |                272.059 | -0.4064 |                       -0.8216 |               -0.013  |                  0.2466 |          277.563 |              277.314 |         0      |           -1.7721 |                            2.7483 |                   1.0384 |            3.8072 |                0.4103 |
        | 2025-12-18 |                -0.0378 |                       1 |         2.31851e+09 |          6.60068e+07 |                      4.02708e+09 |         -1.16789e+07 | nan      |                      0 |               0.0024 |            -7.4094 |                 2735.54 |                 789.853 |            34.4002 |      -75.8192 |              21.4286 |               100      |                 -162.428  |                -0.2832 |                            -0.1659 |      -2.59752e+07 |               50.9808 |                        0.0047 |                        -0.0132 |                                    -0.0434 |                                   0.0302 |                      -0.3822 |                     24.4324 |                     -24.9064 |                    272.6   |              276.008 |                   265.041 |                   250.519 |         24.1808 |         14.0381 |      1.2731 |             2.7823 |                   45.9182 |              2.4336 |              2.0149 |                9.3441 |                           17.1476 |            -0.2126 |               0.0099 |                 0.865  |                 1.0802 |                -2.0699 |                -8.7267 |          -0.0239 |            42.7375 |            0.311  |                        0.4834 |         -0.014  |                       277.308 |                            274.771 |                                    274.694 | 0.0014 |                     277.238 |                         275.257 |                     271.901 |                                  274.081 |                                277.321 |         284.309 |       272.265 |                    273.833 |                 269.328 |                       276.821 |                       274.272 |                285.211 |                 276.181 |                267.151 |         0.2267 |                0.0654 |       6.6568 |               4.9986 |      24.6224 |      268.465 |                            1 |                272.441 |                 279.575 |                 284.635 |                  274.771 |                 264.907 |                  287.618 |                   279.158 |                  270.697 |             271.155 |                   -0.9465 |                       283.46  |                  -43.4252 |                270.209 | -0.4741 |                       -0.9297 |               -0.003  |                  0.5333 |          277.189 |              277.065 |         0.0714 |           -1.4234 |                            3.1412 |                   1.1383 |            3.8072 |                0.4952 |
        | 2025-12-19 |                 0.0139 |                       1 |         2.46314e+09 |          6.67933e+07 |                      4.11448e+09 |          1.95656e+07 | nan      |                      0 |               0.1361 |            -9.4904 |                 2735.54 |                 794.148 |            43.3299 |      -68.989  |              14.2857 |                92.8571 |                 -103.239  |                -0.2694 |                            -0.2349 |       5.93143e+06 |               51.5414 |                        0.0037 |                         0.1453 |                                    -0.0056 |                                   0.1509 |                      -0.4186 |                     23.9078 |                     -39.3401 |                    272.6   |              276.008 |                   265.041 |                   250.519 |         31.011  |         18.7898 |      1.0212 |             2.4301 |                   49.0174 |             12.7427 |              5.0588 |                7.6467 |                           14.7723 |             0.3234 |              -1.1188 |                 0.903  |                 1.137  |                -0.8132 |                -5.4969 |          -0.0333 |            41.2482 |            0.2645 |                        0.4504 |         -0.0269 |                       276.636 |                            274.498 |                                    274.194 | 0.0012 |                     276.433 |                         274.645 |                     271.147 |                                  273.957 |                                276.002 |         282.481 |       269.983 |                    273.942 |                 267.286 |                       276.821 |                       274.272 |                284.285 |                 276.55  |                268.816 |         0.2524 |                0.0559 |       4.6837 |               4.9761 |      24.6659 |      268.465 |                            1 |                272.493 |                 279.704 |                 284.312 |                  274.498 |                 264.683 |                  287.618 |                   276.821 |                  266.023 |             270.663 |                   -0.919  |                       282.61  |                  -42.5829 |                269.744 | -0.6712 |                       -0.9192 |                0.0092 |                  0.7123 |          276.995 |              276.925 |         0.1429 |           -0.9363 |                            3.132  |                   1.1178 |            3.8072 |                0.5375 |
        | 2025-12-22 |                -0.0419 |                      -1 |         2.42657e+09 |          6.64325e+07 |                      4.08789e+09 |          2.21089e+07 | nan      |                      0 |               0.079  |           -12.8267 |                 2708.55 |                 794.148 |            36.4312 |      -81.4487 |               7.1429 |                85.7143 |                 -104.733  |                -0.2367 |                            -0.2614 |      -7.97965e+06 |               45.7604 |                        0.0022 |                         0.1018 |                                     0.0158 |                                   0.0859 |                       2.2593 |                     23.4207 |                     -64.5455 |                    272.6   |              276.008 |                   264.954 |                   250.519 |         18.5513 |         24.581  |      0.5976 |             2.0636 |                   44.0574 |              0      |              5.0588 |                4.6682 |                           12.2463 |            -0.5608 |              -2.0902 |                 0.8075 |                 1.2147 |                -0.8978 |                -4.2561 |          -0.0532 |            40.4149 |            0.2055 |                        0.4111 |         -0.0418 |                       275.553 |                            273.902 |                                    273.122 | 0.0009 |                     275.578 |                         273.764 |                     270.399 |                                  273.151 |                                275.063 |         280.835 |       271.777 |                    274.59  |                 269.906 |                       276.821 |                       274.272 |                284.343 |                 276.525 |                268.707 |         0.0846 |                0.0565 |       3.3583 |               4.8605 |      24.609  |      268.465 |                            1 |                272.722 |                 279.644 |                 283.406 |                  273.902 |                 264.397 |                  287.618 |                   276.821 |                  266.023 |             270.187 |                   -0.8256 |                       280.919 |                  -39.5418 |                269.361 | -1.0834 |                       -0.9246 |                0.0011 |                  0.5069 |          276.572 |              276.218 |         0      |           -1.4788 |                            3.0874 |                   0.9982 |            3.8072 |                0.4958 |
        | 2025-12-23 |                 0.0106 |                       1 |         2.45621e+09 |          6.65845e+07 |                      4.11471e+09 |          2.96727e+07 | nan      |                      0 |               0.0651 |           -14.4654 |                 2722.44 |                 794.148 |            37.2984 |      -69.5725 |               0      |                78.5714 |                  -97.9344 |                -0.205  |                            -0.2501 |      -1.44128e+06 |               49.1967 |                        0.0014 |                         0.0548 |                                     0.0236 |                                   0.0311 |                       2.351  |                     23.1944 |                     -51.4169 |                    272.6   |              276.008 |                   265.083 |                   250.519 |         30.4275 |         26.6632 |      0.3694 |             1.7248 |                   47.029  |             15.5282 |              9.4236 |                3.0048 |                            9.9359 |             0.517  |              -2.6623 |                 0.7732 |                 1.2661 |                -1.9284 |                -4.8582 |          -0.0415 |            46.4518 |            0.1522 |                        0.3663 |         -0.0501 |                       274.714 |                            273.57  |                                    272.607 | 0.0007 |                     274.764 |                         273.212 |                     269.985 |                                  272.908 |                                274.256 |         279.354 |       270.843 |                    272.115 |                 268.757 |                       274.882 |                       272.792 |                284.455 |                 276.347 |                268.24  |         0.1958 |                0.0587 |       2.9298 |               4.7226 |      24.5236 |      268.465 |                            1 |                273     |                 279.366 |                 282.71  |                  273.57  |                 264.43  |                  287.618 |                   276.821 |                  266.023 |             270.21  |                   -0.6929 |                       279.217 |                  -34.7183 |                269.517 | -0.8392 |                       -0.9141 |                0.0076 |                  0.7178 |          274.937 |              273.657 |         0.2143 |           -1.0405 |                            2.8394 |                   0.8474 |            3.8073 |                0.3674 |
        | 2025-12-24 |                 0.0554 |                       1 |         2.47412e+09 |          6.66799e+07 |                      4.11466e+09 |          3.00518e+07 | nan      |                      0 |               0.0846 |            -6.6454 |                 2736.94 |                 794.148 |            41.6581 |      -51.6561 |               0      |                71.4286 |                  -34.6705 |                -0.1841 |                            -0.2228 |       2.20161e+06 |               54.4448 |                        0.0011 |                        -0.0045 |                                     0.018  |                                  -0.0225 |                       3.0907 |                     22.018  |                     -32.919  |                    272.6   |              276.008 |                   265.063 |                   250.519 |         48.3439 |         32.4409 |      0.3017 |             1.4402 |                   50.0121 |             38.5455 |             18.0246 |                2.4525 |                            8.0651 |             0.4551 |              -2.8188 |                 0.8892 |                 1.1613 |                 1.0804 |                -2.1383 |          -0.0245 |            53.1324 |            0.1033 |                        0.3168 |         -0.0518 |                       274.223 |                            273.476 |                                    272.559 | 0.0006 |                     274.049 |                         272.965 |                     270.161 |                                  272.904 |                                273.798 |         278.021 |       270.531 |                    272.438 |                 269.508 |                       273.094 |                       271.425 |                284.438 |                 276.19  |                267.941 |         0.2981 |                0.0597 |       3.2188 |               4.6152 |      24.4118 |      268.465 |                            1 |                273.225 |                 280.416 |                 282.345 |                  273.476 |                 264.606 |                  283.741 |                   274.882 |                  266.023 |             270.449 |                   -0.5807 |                       277.998 |                  -30.1419 |                269.868 | -0.4904 |                       -0.852  |                0.0123 |                  0.8331 |          273.408 |              273.158 |         0.3571 |           -0.4784 |                            2.5438 |                   0.7619 |            3.8073 |                0.2687 |
        | 2025-12-26 |                -0.0066 |                      -1 |         2.4526e+09  |          6.66476e+07 |                      4.10239e+09 |          2.35736e+07 | nan      |                      0 |               0.0857 |            -5.4396 |                 2736.94 |                 792.959 |            46.9441 |      -51.1363 |               0      |                64.2857 |                  -25.6932 |                -0.1697 |                            -0.1974 |  735552           |               51.2159 |                        0.0008 |                        -0.0488 |                                     0.0046 |                                  -0.0534 |                      -0.6819 |                     20.9257 |                     -27.7033 |                    272.6   |              276.098 |                   264.961 |                   250.519 |         48.8637 |         42.545  |      0.2126 |             1.1946 |                   49.1689 |             34.6217 |             29.5651 |                1.7619 |                            6.4893 |            -0.3028 |              -2.1658 |                 0.9091 |                 1.1011 |                 1.1553 |                -1.346  |          -0.0193 |            55.49   |            0.0564 |                        0.2631 |         -0.0522 |                       273.84  |                            273.339 |                                    272.426 | 0.0004 |                     273.429 |                         272.729 |                     270.664 |                                  272.88  |                                273.418 |         276.821 |       272.863 |                    274.471 |                 271.252 |                       272.6   |                       271.048 |                284.387 |                 275.983 |                267.579 |         0.2899 |                0.0609 |       2.5013 |               4.4642 |      24.2185 |      268.465 |                            1 |                273.538 |                 280.103 |                 281.822 |                  273.339 |                 264.856 |                  280.164 |                   273.094 |                  266.023 |             270.506 |                   -0.513  |                       277.175 |                  -27.157  |                269.993 | -0.383  |                       -0.7986 |                0.0095 |                  0.7973 |          273.009 |              272.859 |         0.2857 |           -0.5171 |                            2.3046 |                   0.7182 |            3.8073 |                0.1676 |
        | 2025-12-29 |                 0.0405 |                       1 |         2.47632e+09 |          6.66789e+07 |                      4.11195e+09 |          2.18213e+07 |   0.4593 |                      0 |               0.0756 |            -4.7955 |                 2736.94 |                 794.003 |            48.1203 |      -48.4089 |              35.7143 |                57.1429 |                  -30.5671 |                -0.1563 |                            -0.1782 |       1.7719e+06  |               57.3149 |                        0.0006 |                        -0.0817 |                                    -0.0126 |                                  -0.0691 |                       0.1103 |                     20.0593 |                     -21.8633 |                    270.612 |              276.183 |                   265.068 |                   250.519 |         51.5911 |         49.5995 |      0.169  |             0.9895 |                   49.9666 |             40.0245 |             37.7306 |                1.4003 |                            5.217  |             0.5323 |              -2.0654 |                 0.9316 |                 1.0955 |                 0.2129 |                -1.7901 |          -0.0149 |            54.4742 |            0.0118 |                        0.2104 |         -0.0496 |                       273.546 |                            273.268 |                                    272.416 | 0.0003 |                     272.918 |                         272.591 |                     271.326 |                                  272.877 |                                273.16  |         275.741 |       272.926 |                    273.939 |                 271.438 |                       272.6   |                       271.048 |                284.195 |                 275.729 |                267.264 |         0.3276 |                0.0614 |       2.003  |               4.2884 |      23.8779 |      268.465 |                            1 |                273.905 |                 279.736 |                 281.304 |                  273.268 |                 265.233 |                  279.177 |                   272.6   |                  266.023 |             270.682 |                   -0.4407 |                       276.411 |                  -23.7843 |                270.241 | -0.294  |                       -0.7244 |                0.0103 |                  0.854  |          272.834 |              272.81  |         0.4286 |           -0.2895 |                            2.0477 |                   0.6802 |            3.8073 |                0.158  |
        | 2025-12-30 |                -0.0195 |                      -1 |         2.45418e+09 |          6.66239e+07 |                      4.10949e+09 |          1.83379e+07 |   1.5877 |                      0 |               0.0249 |            -5.2836 |                 2730.14 |                 794.003 |            43.4593 |      -53.5607 |              28.5714 |                50      |                  -35.5284 |                -0.1288 |                            -0.1608 | -464807           |               54.1042 |                        0.0003 |                        -0.1122 |                                    -0.0325 |                                  -0.0797 |                      -2.3582 |                     19.2762 |                     -21.7389 |                    270.612 |              276.183 |                   265.068 |                   250.519 |         46.4393 |         48.9647 |      0.0788 |             0.8074 |                   48.4209 |             29.5551 |             34.7338 |                0.6843 |                            4.0838 |             0.15   |              -1.9448 |                 0.9145 |                 1.1096 |                 0.0857 |                -1.708  |          -0.0148 |            53.5704 |           -0.0182 |                        0.1612 |         -0.0465 |                       273.254 |                            273.117 |                                    272.247 | 0.0002 |                     272.531 |                         272.403 |                     271.798 |                                  272.849 |                                272.911 |         274.769 |       272.541 |                    273.676 |                 271.673 |                       272.6   |                       271.048 |                283.299 |                 275.23  |                267.162 |         0.308  |                0.0586 |       1.7937 |               4.1102 |      23.5003 |      268.465 |                            1 |                274.284 |                 279.357 |                 280.707 |                  273.117 |                 265.526 |                  279.177 |                   272.6   |                  266.023 |             270.7   |                   -0.3931 |                       275.809 |                  -21.4583 |                270.306 | -0.2918 |                       -0.6718 |                0.0069 |                  0.7814 |          272.765 |              272.72  |         0.2857 |           -0.4587 |                            1.8328 |                   0.6541 |            3.8073 |                0.1582 |
        | 2025-12-31 |                -0.0707 |                      -1 |         2.42688e+09 |          6.65019e+07 |                      4.08531e+09 |          7.5513e+06  | nan      |                      0 |              -0.0357 |            -5.2468 |                 2730.14 |                 790.456 |            38.2003 |      -62.8029 |              21.4286 |                42.8571 |                  -56.9136 |                -0.0887 |                            -0.1381 |      -4.82705e+06 |               46.4079 |                       -0.0003 |                        -0.1258 |                                    -0.0512 |                                  -0.0746 |                      -1.5169 |                     18.7176 |                     -37.4456 |                    270.248 |              276.821 |                   265.6   |                   250.867 |         37.1971 |         45.0758 |     -0.0897 |             0.628  |                   45.6902 |             12.0094 |             27.1964 |               -0.6702 |                            2.8953 |            -0.6218 |              -1.6049 |                 0.9113 |                 1.1148 |                -0.0091 |                -1.9324 |          -0.0248 |            52.5944 |           -0.0452 |                        0.1157 |         -0.0431 |                       272.762 |                            272.823 |                                    271.815 | 0.0001 |                     272.2   |                         272.091 |                     271.883 |                                  272.686 |                                272.594 |         273.895 |       272.198 |                    273.062 |                 271.268 |                       272.6   |                       271.048 |                281.366 |                 274.516 |                267.666 |         0.2372 |                0.0499 |       1.9233 |               3.954  |      23.1557 |      268.465 |                            1 |                274.628 |                 279.013 |                 280.039 |                  272.823 |                 265.607 |                  279.177 |                   272.6   |                  266.023 |             270.749 |                   -0.3096 |                       274.774 |                  -17.2037 |                270.44  | -0.4926 |                       -0.6071 |                0.0023 |                  0.6255 |          272.585 |              272.451 |         0.1429 |           -0.8651 |                            1.4702 |                   0.5702 |            3.8073 |                0.1177 |
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

        collections = [
            self.collect_breadth_indicators(period=period, close_column=close_column),
            self.collect_momentum_indicators(
                period=period, close_column=close_column, window=window
            ),
            self.collect_overlap_indicators(
                period=period, close_column=close_column, window=window
            ),
            self.collect_volatility_indicators(
                period=period, close_column=close_column, window=window
            ),
        ]

        # A topic without indicators returns an empty DataFrame, which is left out so it does
        # not change the dates or the columns of the others.
        for collect_indicators in (
            self.collect_statistics_indicators,
            self.collect_cycle_indicators,
        ):
            indicators = collect_indicators(
                period=period, close_column=close_column, window=window
            )
            if not indicators.empty:
                collections.append(indicators)

        self._all_indicators = pd.concat(collections, axis=1)

        self._all_indicators = apply_rounding(
            self._all_indicators, rounding if rounding is not None else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            self._all_indicators_growth = calculate_growth(
                dataset=self._all_indicators,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="index",
            )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                self._all_indicators_growth = calculate_standardization(
                    dataset=self._all_indicators_growth,
                    rounding=standardize_rounding,
                    axis="rows",
                )
            else:
                self._all_indicators = calculate_standardization(
                    dataset=self._all_indicators,
                    rounding=standardize_rounding,
                    axis="rows",
                )

        return self._all_indicators_growth if growth else self._all_indicators

    def _get_historical_data(self, period: str, close_column: str) -> pd.DataFrame:
        """
        The historical data of a period with the open, high and low on the same basis as the
        close an indicator uses. With the dividend-adjusted close (the default) they are
        adjusted by the same factor, Adj Close / Close, so that a bounded indicator stays within
        its bounds and a range is not distorted around ex-dividend days.

        The factor changes on every ex-dividend date, so the open, high and low of a week,
        month, quarter or year are taken from the adjusted daily prices (the first open, the
        highest high and the lowest low) rather than adjusted with the factor at the end of
        the period. The latter would overstate an open before an ex-dividend date within the
        period by that dividend's yield.

        Args:
            period (str): The period, e.g. "daily".
            close_column (str): The close the indicator uses.

        Returns:
            pd.DataFrame: The historical data, with an adjusted open, high and low when the close
            is the adjusted close.
        """
        historical_data = self._historical_data[period]
        columns = historical_data.columns.get_level_values(0)
        if close_column != "Adj Close" or "Close" not in columns:
            return historical_data

        key = ("consistent", period)
        if key not in self._consistent_data:
            factor = historical_data["Adj Close"] / historical_data["Close"]
            adjusted = {
                column: historical_data[column] * factor
                for column in ("Open", "High", "Low")
                if column in columns
            }

            daily_data = self._historical_data.get("daily", pd.DataFrame())
            if (
                period in PERIOD_TRANSLATION
                and not daily_data.empty
                and isinstance(daily_data.index, pd.PeriodIndex)
                and "Close" in daily_data.columns.get_level_values(0)
            ):
                daily_adjusted = self._get_historical_data("daily", close_column)
                dates = daily_adjusted.index.asfreq(PERIOD_TRANSLATION[period])
                # The flat bars of the dates a ticker did not trade while another did are left out, as in the period data itself, so a stock's week does not open or bottom at a holiday's carried-forward close.  # noqa: E501
                non_trading_days = get_non_trading_days(daily_adjusted, "Close")
                for column, aggregation in (
                    ("Open", "first"),
                    ("High", "max"),
                    ("Low", "min"),
                ):
                    if column in adjusted:
                        # A period the daily data does not cover, or without a trading day of the ticker, keeps the end-of-period factor.  # noqa: E501
                        adjusted[column] = (
                            daily_adjusted[column]
                            .mask(
                                non_trading_days.reindex(
                                    columns=daily_adjusted[column].columns,
                                    fill_value=False,
                                )
                            )
                            .groupby(dates)
                            .agg(aggregation)
                            .reindex(
                                index=historical_data.index,
                                columns=adjusted[column].columns,
                            )
                            .combine_first(adjusted[column])
                        )

            self._consistent_data[key] = pd.concat(
                {
                    column: adjusted.get(column, historical_data[column])
                    for column in columns.unique()
                },
                axis=1,
            )
        return self._consistent_data[key]

    def _close_prices(
        self, period: str, close_column: str, traded_only: bool = False
    ) -> pd.DataFrame:
        """
        The closes of a period, with the intraday check.

        Args:
            period (str): The period, e.g. "daily".
            close_column (str): The close column, "Close" for intraday data.
            traded_only (bool, optional): Whether the closes are NaN on the dates a ticker
                did not trade while another did (see `get_non_trading_days`) rather than the
                carried-forward close. Defaults to False.

        Returns:
            pd.DataFrame: The closes, a column per ticker.
        """
        if period not in PERIODS:
            raise ValueError(
                "Period must be intraday, daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"
        closes = self._historical_data[period][close_column]
        if traded_only:
            closes = closes.mask(
                get_non_trading_days(self._historical_data[period], close_column)
            )
        return closes

    @handle_errors
    def get_factor_analysis(
        self,
        metrics: str | list[str] = "momentum",
        period: str = "monthly",
        horizons: list[int] | int | None = None,
        quantiles: int = 5,
        periods_apart: int = 1,
        detail: str | None = None,
        column: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Factor analysis (predictive power) of the technical indicators: whether the tickers
        that score high on an indicator go on to return more than those that score low. Per
        period, the information coefficient (IC) is the Spearman rank correlation between the
        indicator of the tickers at the end of the period and their return over the following
        horizon: 1 means the indicator ranks the later returns perfectly, -1 in reverse and 0
        that it says nothing about them. The IC measures skill in Grinold and Kahn's
        fundamental law of active management, IR ≈ IC × √breadth: a small but consistent IC
        across many tickers and periods gives a high information ratio.

        The summary has a row per indicator and horizon with the mean IC, its standard
        deviation, the IC information ratio (mean divided by standard deviation), the
        t-statistic of the mean with its p-value, the share of periods with a positive IC, the
        number of periods, the top minus bottom spread (the mean forward return of the highest
        quantile minus that of the lowest), the mean turnover of the top and bottom quantiles
        and the mean rank autocorrelation of the indicator from one period to the next. With
        detail, it returns one of those parts per period (or per quantile) for a single
        indicator.

        The indicators are calculated daily and the value on the last trading day of each
        period is compared with the returns from that day's close onwards, using the dividend
        and split adjusted close. An indicator in price units does not compare across tickers
        with different share prices, so a price line (a moving average, band, channel, stop or
        cloud line) is used as the distance of the close from it, close / line - 1, and a price
        difference (the true range, ATR, MACD, awesome oscillator, elder ray and detrended
        price oscillator) as a share of the close. Each ticker keeps its own history: a ticker
        without a value or a return in a period is left out of that period only.

        Args:
            metrics (str | list[str], optional): An indicator method name, e.g.
                "get_relative_strength_index", a list of them, or a collection: "momentum",
                "overlap", "volatility", "breadth" or "all". Defaults to "momentum", the
                oscillators and momentum indicators. The collections leave out the market-wide
                breadth measures, the volume-scaled indicators and the pivot points and
                Fibonacci levels.
            period (str, optional): The frequency at which the indicators are evaluated:
                daily, weekly, monthly, quarterly or yearly. Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            quantiles (int, optional): The number of quantiles the tickers are sorted into per
                period. Defaults to 5.
            periods_apart (int, optional): The number of periods between the dates the
                quantile turnover and the rank autocorrelation compare. Defaults to 1.
            detail (str | None, optional): Returns one part of the analysis of a single
                indicator instead of the summary: "information_coefficient" (per period and
                horizon), "quantile_returns" (the mean forward return per quantile and
                horizon), "quantile_turnover" (per period and quantile) or
                "rank_autocorrelation" (per period). Defaults to None.
            column (str | None, optional): The output to use of an indicator with several,
                e.g. "Signal Line" of the MACD or "Upper Band" of the Bollinger Bands.
                Defaults to None, which uses the first output in alphabetical order other
                than a price, e.g. "MACD Line" or "Lower Band".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            **kwargs: Parameters passed to the indicators that take them, e.g. window=14 or
                close_column="Close".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The summary per indicator and horizon, or with detail that part of
            the analysis of the indicator.

        Notes:
            - The IC is reported as measured: an overbought RSI that goes on to underperform
              gives a negative IC. Read the sign as the direction of the relationship.
            - The cross-section is the tickers of the Toolkit, so use at least ten and
              preferably twenty or more for a meaningful rank correlation and quantiles.
            - A horizon longer than one period makes consecutive returns overlap, which
              inflates the t-statistic; judge those horizons by their mean IC and IC
              information ratio.
            - Based on alphalens (Quantopian) and on Grinold and Kahn (2000), Active Portfolio
              Management.

        Also known as: predictive power, information coefficient, factor testing, alpha signal

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ",
             "XOM", "PG", "KO", "WMT", "V", "UNH", "HD", "CVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        # The 200-day rate of change and the distance of the close from its 200-day average.
        toolkit.technicals.get_factor_analysis(
            ["get_rate_of_change", "get_moving_average"], horizons=[1, 12], window=200
        )
        ```

        Which returns:
        |                                     |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   p-Value |   Positive Share |   Observations |   Top Minus Bottom |   Quantile Turnover |   Rank Autocorrelation |
        |:------------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|-----------------:|---------------:|-------------------:|--------------------:|-----------------------:|
        | ('get_rate_of_change', '1 Month')   |    0.0477 |                  0.3456 |                 0.1381 |        1.5063 |    0.1347 |           0.5798 |            119 |             0.0092 |              0.2626 |                 0.8582 |
        | ('get_rate_of_change', '12 Months') |    0.1181 |                  0.3986 |                 0.2964 |        3.0802 |    0.0026 |           0.6389 |            108 |             0.08   |              0.2626 |                 0.8582 |
        | ('get_moving_average', '1 Month')   |    0.0383 |                  0.3458 |                 0.1107 |        1.2071 |    0.2298 |           0.563  |            119 |             0.0125 |              0.3074 |                 0.8031 |
        | ('get_moving_average', '12 Months') |    0.1018 |                  0.3848 |                 0.2646 |        2.7503 |    0.007  |           0.5833 |            108 |             0.0853 |              0.3074 |                 0.8031 |
        """
        tickers = self._tickers
        factors = factor_model.get_daily_factors(
            self,
            factor_model.select_metrics(metrics, FACTOR_COLLECTIONS),
            kwargs,
            {},
            tickers,
            column,
        )

        close = factor_model.to_daily_factor(
            self._historical_data["daily"][kwargs.get("close_column", "Adj Close")],
            tickers,
        )
        for name, factor in factors.items():
            # The Supertrend's direction is +1 or -1 and stays as it is.
            if name in FACTOR_PRICE_LINES and column != "Trend Direction":
                factors[name] = close.reindex_like(factor) / factor - 1
            elif name in FACTOR_PRICE_DIFFERENCES:
                factors[name] = factor / close.reindex_like(factor)

        result = factor_model.get_factor_analysis(
            factors,
            self._historical_data["daily"]["Adj Close"],
            period,
            horizons,
            quantiles,
            periods_apart,
            detail,
            self._start_date,
            self._end_date,
            returns=self._historical_data["daily"].get("Return"),
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )


@filter_shown_columns
@track_sources
class Technicals(
    BreadthTechnicals,
    MomentumTechnicals,
    OverlapTechnicals,
    VolatilityTechnicals,
    PatternTechnicals,
    SeasonalityTechnicals,
    StatisticsTechnicals,
    CycleTechnicals,
    TechnicalsBase,
):
    """
    The Technicals Module contains 50+ Technical Indicators that can
    be used to analyse companies. These ratios are divided into 4
    categories which are breadth, momentum, overlap and volatility.
    Each indicator is calculated using the data from the Toolkit module.
    """
