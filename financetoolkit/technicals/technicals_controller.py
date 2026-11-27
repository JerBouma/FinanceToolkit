"""Technicals Module"""

__docformat__ = "google"

import inspect
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
    is_factor_listed,
    report_missing_values,
)
from financetoolkit.technicals.labeling.labeling_controller import LabelingTechnicals
from financetoolkit.technicals.momentum.momentum_controller import MomentumTechnicals
from financetoolkit.technicals.overlap.overlap_controller import OverlapTechnicals
from financetoolkit.technicals.pattern.pattern_controller import PatternTechnicals
from financetoolkit.technicals.seasonality.seasonality_controller import (
    SeasonalityTechnicals,
)
from financetoolkit.technicals.volatility.volatility_controller import (
    VolatilityTechnicals,
)
from financetoolkit.utilities import factor_model, logger_model
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
    PERIOD_TRANSLATION,
    apply_rounding,
    calculate_growth,
    calculate_standardization,
)

logger = logger_model.get_logger()

# The examples' output tables are wider than the line length.
# ruff: noqa: E501

# The columns of the historical data an indicator can be calculated on.
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close")


@report_missing_values
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
        profile: pd.DataFrame | Callable[[], pd.DataFrame] | None = None,
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
            profile (pd.DataFrame | Callable | None, optional): The company profiles, with
                the tickers as columns and the profile items (such as "Sector" and
                "Industry") as rows, or a function that retrieves them when first needed,
                for the groups of get_factor_analysis. Defaults to None, which leaves
                group="sector" and group="industry" unavailable.

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
        self._profile_source = profile
        self._historical_data = historical_data
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._portfolio_weights: dict | None = None
        # The historical data with the open, high and low adjusted like the close, per period.
        self._consistent_data: dict[tuple, pd.DataFrame] = {}
        # The state of report_missing_values (technicals helpers): the depth of the calls
        # and collections running, whether the window of the current indicator was reported
        # as longer than the data, and the long windows and the indicators without values a
        # collection reports in one message.
        self._collecting = 0
        self._depth = 0
        self._window_too_long = False
        self._long_windows: set[int] = set()
        self._without_values: dict[str, None] = {}

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
        Collect all the technical indicators of every ticker in one table: the breadth, momentum, overlap,
        volatility and cycle indicators.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            window (int, optional): The number of periods to use for the calculation, at least 2 as
                the Commodity Channel Index and the Choppiness Index need two periods. Defaults to 14.
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

        toolkit.technicals.collect_all_indicators().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   McClellan Oscillator |   Advancers - Decliners |   On-Balance Volume |   Price Volume Trend |   Accumulation/Distribution Line |   Chaikin Oscillator |     TRIN |   New Highs - New Lows |   Chaikin Money Flow |   Ease of Movement |   Negative Volume Index |   Positive Volume Index |   Klinger Volume Oscillator |   Klinger Volume Oscillator Signal Line |   Market Facilitation Index |   Williams Accumulation/Distribution |   Relative Volume |   Volume Flow Indicator |   Finite Volume Elements |   Twiggs Money Flow |   Volume Zone Oscillator |   Cumulative Force Index |   Price Volume |   Price Volume Rank |   Buying Pressure |   Selling Pressure |   Archer On-Balance Volume Fast Average |   Archer On-Balance Volume Slow Average |   Archer On-Balance Volume Minimum |   Archer On-Balance Volume Maximum |   Archer On-Balance Volume Long Run |   Archer On-Balance Volume Short Run |   Money Flow Index |   Williams %R |   Aroon Indicator Up |   Aroon Indicator Down |   Aroon Oscillator |   Commodity Channel Index |   Relative Vigor Index |   Relative Vigor Index Signal Line |       Force Index |   Ultimate Oscillator |   Percentage Price Oscillator |   Percentage Price Oscillator Signal Line |   Percentage Price Oscillator Histogram |   Absolute Price Oscillator |   Percentage Volume Oscillator |   Percentage Volume Oscillator Signal Line |   Percentage Volume Oscillator Histogram |   Detrended Price Oscillator |   Average Directional Index |   Plus Directional Movement |   Minus Directional Movement |   Plus Directional Indicator |   Minus Directional Indicator |   Directional Movement Index |   Average Directional Index Rating |   Chande Momentum Oscillator |   Intraday Momentum Index |   Ichimoku Conversion Line |   Ichimoku Base Line |   Ichimoku Leading Span A |   Ichimoku Leading Span B |   Stochastic %K |   Stochastic %D |   MACD Line |   MACD Signal Line |   MACD Histogram |   Relative Strength Index |   Stochastic RSI %K |   Stochastic RSI %D |   True Strength Index |   True Strength Index Signal Line |   Balance of Power |   Internal Bar Strength |   Qstick |   Awesome Oscillator |   Accelerator Oscillator |   Vortex Indicator VI+ |   Vortex Indicator VI- |   Elder Ray Bull Power |   Elder Ray Bear Power |   Rate of Change |   Momentum |   Choppiness Index |   Efficiency Ratio |   Vertical Horizontal Filter |   Know Sure Thing |   Know Sure Thing Signal Line |   Coppock Curve |   Fisher Transform |   Fisher Transform Trigger |   Inverse Fisher Transform RSI |   KDJ K |   KDJ D |   KDJ J |   Premier Stochastic Oscillator |   Stochastic Momentum Index |   Stochastic Momentum Index Signal Line |   SMI Ergodic Indicator |   SMI Ergodic Signal Line |   SMI Ergodic Oscillator |   Schaff Trend Cycle |   WaveTrend |   WaveTrend Signal Line |   Derivative Oscillator |   Swing Index |   Accumulative Swing Index |     QQE |   QQE RSI Moving Average |   Dynamic Momentum Index |   Price Zone Oscillator |   Connors RSI |   Inertia |    Bias |   BRAR AR |   BRAR BR |   Pretty Good Oscillator |   Psychological Line |   Volatility-Based Momentum |   AMAT Long Run |   AMAT Short Run |   TTM Trend |   Correlation Trend Indicator |   Forecast Oscillator |   Projection Oscillator |   Simple Moving Average (SMA) |   Exponential Moving Average (EMA) |   Double Exponential Moving Average (DEMA) |   TRIX |   Triangular Moving Average |   Weighted Moving Average (WMA) |   Hull Moving Average (HMA) |   Triple Exponential Moving Average (TEMA) |   Zero Lag Exponential Moving Average (ZLEMA) |   Wilder's Smoothed Moving Average (RMA) |   Sine Weighted Moving Average (SINWMA) |   Fibonacci Weighted Moving Average (FWMA) |   Pascal's Weighted Moving Average (PWMA) |   Kaufman Adaptive Moving Average (KAMA) |   Tillson T3 Moving Average (T3) |   Arnaud Legoux Moving Average (ALMA) |   Volume Weighted Moving Average (VWMA) |   McGinley Dynamic |   Variable Index Dynamic Average (VIDYA) |   Fractal Adaptive Moving Average (FRAMA) |   Holt-Winters Moving Average (HWMA) |   Holt-Winters Channel Upper Band |   Holt-Winters Channel Lower Band |   Elastic Volume Weighted Moving Average (EVWMA) |   Volume Adjusted Moving Average (VAMA) |   Midpoint |   Midpoint Price |   Average Price |   Median Price |   Typical Price |   Weighted Close Price |   Volume Weighted Average Price (VWAP) |   Parabolic SAR |   Gann HiLo Activator |   Zig Zag |   Pivot Point |   Pivot Point Resistance 1 |   Pivot Point Support 1 |   Fibonacci Retracement 50.0% |   Fibonacci Retracement 61.8% |   Bollinger Band Upper |   Bollinger Band Middle |   Bollinger Band Lower |   Bollinger %B |   Bollinger BandWidth |   True Range |   Average True Range |   Mass Index |   Supertrend |   Supertrend Trend Direction |   Chandelier Exit Long |   Chandelier Exit Short |   Keltner Channel Upper |   Keltner Channel Middle |   Keltner Channel Lower |   Donchian Channel Upper |   Donchian Channel Middle |   Donchian Channel Lower |   Normalized Average True Range |   Average Day Range |   Chaikin Volatility |   Relative Volatility Index |   Acceleration Band Upper |   Acceleration Band Middle |   Acceleration Band Lower |   Chande Kroll High Stop |   Chande Kroll Low Stop |   Adaptive Price Zone Upper |   Adaptive Price Zone Middle |   Adaptive Price Zone Lower |   Aberration Upper |   Aberration Middle |   Aberration Lower |    PMax |   PMax Trend Direction |   Price Distance |   Market Thermometer |   Market Thermometer Average |   Squeeze Momentum |   Squeeze |   Hilbert Dominant Cycle Period |   Hilbert Dominant Cycle Phase |   Hilbert In Phase |   Hilbert Quadrature |   Hilbert Sine |   Hilbert Lead Sine |   Hilbert Trend Mode |   Hilbert Instantaneous Trendline |   MESA Adaptive Moving Average (MAMA) |   Following Adaptive Moving Average (FAMA) |   Swiss Army Knife Filter |   SuperSmoother Filter |   Even Better Sinewave |   Detrended Synthetic Price |   MESA Sine Wave Sine |   MESA Sine Wave Lead Sine |   Center of Gravity Oscillator |   Center of Gravity Trigger |   Laguerre RSI |
        |:-----------|-----------------------:|------------------------:|--------------------:|---------------------:|---------------------------------:|---------------------:|---------:|-----------------------:|---------------------:|-------------------:|------------------------:|------------------------:|----------------------------:|----------------------------------------:|----------------------------:|-------------------------------------:|------------------:|------------------------:|-------------------------:|--------------------:|-------------------------:|-------------------------:|---------------:|--------------------:|------------------:|-------------------:|----------------------------------------:|----------------------------------------:|-----------------------------------:|-----------------------------------:|------------------------------------:|-------------------------------------:|-------------------:|--------------:|---------------------:|-----------------------:|-------------------:|--------------------------:|-----------------------:|-----------------------------------:|------------------:|----------------------:|------------------------------:|------------------------------------------:|----------------------------------------:|----------------------------:|-------------------------------:|-------------------------------------------:|-----------------------------------------:|-----------------------------:|----------------------------:|----------------------------:|-----------------------------:|-----------------------------:|------------------------------:|-----------------------------:|-----------------------------------:|-----------------------------:|--------------------------:|---------------------------:|---------------------:|--------------------------:|--------------------------:|----------------:|----------------:|------------:|-------------------:|-----------------:|--------------------------:|--------------------:|--------------------:|----------------------:|----------------------------------:|-------------------:|------------------------:|---------:|---------------------:|-------------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------:|-----------:|-------------------:|-------------------:|-----------------------------:|------------------:|------------------------------:|----------------:|-------------------:|---------------------------:|-------------------------------:|--------:|--------:|--------:|--------------------------------:|----------------------------:|----------------------------------------:|------------------------:|--------------------------:|-------------------------:|---------------------:|------------:|------------------------:|------------------------:|--------------:|---------------------------:|--------:|-------------------------:|-------------------------:|------------------------:|--------------:|----------:|--------:|----------:|----------:|-------------------------:|---------------------:|----------------------------:|----------------:|-----------------:|------------:|------------------------------:|----------------------:|------------------------:|------------------------------:|-----------------------------------:|-------------------------------------------:|-------:|----------------------------:|--------------------------------:|----------------------------:|-------------------------------------------:|----------------------------------------------:|-----------------------------------------:|----------------------------------------:|-------------------------------------------:|------------------------------------------:|-----------------------------------------:|---------------------------------:|--------------------------------------:|----------------------------------------:|-------------------:|-----------------------------------------:|------------------------------------------:|-------------------------------------:|----------------------------------:|----------------------------------:|-------------------------------------------------:|----------------------------------------:|-----------:|-----------------:|----------------:|---------------:|----------------:|-----------------------:|---------------------------------------:|----------------:|----------------------:|----------:|--------------:|---------------------------:|------------------------:|------------------------------:|------------------------------:|-----------------------:|------------------------:|-----------------------:|---------------:|----------------------:|-------------:|---------------------:|-------------:|-------------:|-----------------------------:|-----------------------:|------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|-------------------------:|--------------------------------:|--------------------:|---------------------:|----------------------------:|--------------------------:|---------------------------:|--------------------------:|-------------------------:|------------------------:|----------------------------:|-----------------------------:|----------------------------:|-------------------:|--------------------:|-------------------:|--------:|-----------------------:|-----------------:|---------------------:|-----------------------------:|-------------------:|----------:|--------------------------------:|-------------------------------:|-------------------:|---------------------:|---------------:|--------------------:|---------------------:|----------------------------------:|--------------------------------------:|-------------------------------------------:|--------------------------:|-----------------------:|-----------------------:|----------------------------:|----------------------:|---------------------------:|-------------------------------:|----------------------------:|---------------:|
        | 2025-12-17 |                -0.0981 |                      -1 |         2.26688e+09 |          6.59406e+07 |                      3.99771e+09 |         -1.74849e+07 | nan      |                      0 |              -0.0358 |            -4.7166 |                 2735.53 |                 788.839 |                -2.03391e+08 |                            -4.28732e+07 |                      8.9837 |                              165.336 |            1.171  |                 25.8886 |                  -0.0337 |             -0.0429 |                  -0.194  |              1.01185e+10 |    1.36297e+10 |                   3 |            0.7602 |             0.9939 |                             2.29242e+09 |                             2.31813e+09 |                        2.26688e+09 |                        2.31702e+09 |                                   0 |                                    1 |            39.7393 |      -98.8222 |              28.5714 |               100      |           -71.4286 |                 -139.905  |                -0.2262 |                            -0.0758 |      -3.27416e+07 |               44.5331 |                        0.0063 |                                    0.0116 |                                 -0.0053 |                      1.7262 |                        -0.0329 |                                    -0.0509 |                                   0.018  |                       0.0306 |                     24.6713 |                     12.185  |                      13.7577 |                      17.868  |                       20.1742 |                       6.0622 |                            27.6362 |                     -20.6209 |                   41.1264 |                    275.43  |              276.008 |                   265.041 |                   250.519 |          1.1778 |          8.6605 |      1.7262 |             3.1596 |          -1.4334 |                   45.1866 |              0      |              1.2037 |               12.3225 |                           19.7487 |            -0.7013 |                  0.0442 |  -0.4015 |               1.2653 |                  -1.836  |                 0.9211 |                 1.0533 |                -0.1328 |                -4.6371 |          -0.0206 |    -5.6902 |            49.8427 |             0.2062 |                       0.5182 |            0.3638 |                        0.5107 |          0.0029 |            -2.2385 |                    -1.8906 |                        -0.8927 | 15.8397 | 25.2454 | -2.9718 |                         -0.84   |                      9.1395 |                                 26.1148 |                 -6.2668 |                    4.4051 |                 -10.6719 |               1.3165 |     -2.5415 |                 12.0151 |                 -6.9465 |      -38.8013 |                    5877.92 | 57.4732 |                  50.8734 |                  13.8158 |                -14.5643 |       20.1105 |   51.7715 | -0.014  |  131.997  |  121.214  |                  -1.3452 |              33.3333 |                     -2.243  |               0 |                1 |          -1 |                       -0.8216 |               -0.013  |                  0.2466 |                       277.781 |                            275.314 |                                    275.767 | 0.0016 |                     277.982 |                         276.128 |                     273.116 |                                    273.419 |                                       274.067 |                                  273.378 |                                 278.063 |                                    273.419 |                                   277.589 |                                  274.218 |                          279.323 |                               274.905 |                                 277.896 |            271.978 |                                  270.616 |                                   275.481 |                              276.12  |                           280.057 |                           272.184 |                                          273.073 |                                 277.765 |    278.046 |          279.158 |         272.712 |        272.949 |         272.265 |                271.923 |                                277.974 |         285.493 |               280.207 |   270.697 |       273.015 |                    275.185 |                 271.488 |                       279.158 |                       277.161 |                285.548 |                 276     |                266.452 |         0.2327 |                0.0692 |       4.5043 |               4.871  |      24.5798 |      268.465 |                            1 |                272.67  |                 279.347 |                 284.794 |                  275.314 |                 265.833 |                  287.618 |                   279.232 |                  270.846 |                          0.018  |              4.5036 |              -0.1546 |                     49.6227 |                   288.841 |                    276     |                   263.609 |                  282.999 |                 270.062 |                     284.262 |                      275.342 |                     266.422 |            279.563 |             274.669 |            269.775 | 263.457 |                      1 |           6.2482 |               0.6577 |                       1.7931 |             0.179  |         1 |                         18.3902 |                       282.339  |            -2.1962 |               4.3124 |        -0.9769 |             -0.5397 |                    0 |                           276.437 |                               276.673 |                                    273.841 |                   -2.8349 |                274.241 |                -0.9054 |                     -1.3767 |               -0.7739 |                    -0.0994 |                        -5.5219 |                     -5.5285 |         0      |
        | 2025-12-18 |                -0.0378 |                       1 |         2.31851e+09 |          6.6007e+07  |                      4.02708e+09 |         -1.16789e+07 | nan      |                      0 |               0.0024 |            -7.4094 |                 2735.53 |                 789.855 |                -2.50907e+08 |                            -7.25923e+07 |                     12.8931 |                              170.558 |            1.1996 |                 25.9203 |                   0.0135 |             -0.0099 |                  -0.0055 |              1.0083e+10  |    1.40534e+10 |                   1 |            0.9191 |             0.961  |                             2.30286e+09 |                             2.31819e+09 |                        2.26688e+09 |                        2.31851e+09 |                                   0 |                                    1 |            34.4002 |      -75.8192 |              21.4286 |               100      |           -78.5714 |                 -162.428  |                -0.2832 |                            -0.1659 |      -2.59752e+07 |               50.9808 |                        0.0047 |                                    0.0102 |                                 -0.0055 |                      1.2731 |                        -0.0132 |                                    -0.0434 |                                   0.0302 |                      -0.3822 |                     24.4324 |                     11.3147 |                      17.4488 |                      16.1683 |                       24.9338 |                      21.3261 |                            28.0928 |                     -24.9064 |                   36.3175 |                    272.6   |              276.008 |                   265.041 |                   250.519 |         24.1808 |         14.0381 |      1.2731 |             2.7823 |          -1.5092 |                   45.9182 |              2.4336 |              2.0149 |                9.3441 |                           17.1476 |            -0.2126 |                  0.7844 |  -0.6157 |               0.0099 |                  -2.1487 |                 0.865  |                 1.0802 |                -2.0699 |                -8.7267 |          -0.0239 |    -6.6369 |            42.7375 |             0.2491 |                       0.5366 |            0.311  |                        0.4834 |         -0.014  |            -2.6249 |                    -2.2385 |                        -0.9368 | 23.7921 | 24.761  | 21.8543 |                         -0.8426 |                      3.2693 |                                 21.5457 |                -10.1477 |                   -0.4458 |                  -9.7019 |               0.6582 |    -12.4226 |                  2.2173 |                 -7.2145 |      -12.5266 |                    5865.39 | 55.8771 |                  49.2217 |                  15.5532 |                  0.5149 |       46.5574 |   50.526  | -0.0123 |  108.702  |  105.088  |                  -1.1882 |              33.3333 |                     -2.7539 |               0 |                1 |          -1 |                       -0.9297 |               -0.003  |                  0.5333 |                       277.308 |                            274.771 |                                    274.694 | 0.0014 |                     277.238 |                         275.257 |                     271.901 |                                    272.199 |                                       272.815 |                                  273.226 |                                 277.267 |                                    272.587 |                                   277.094 |                                  274.081 |                          279.146 |                               273.813 |                                 277.325 |            271.925 |                                  270.663 |                                   273.495 |                              274.766 |                           278.713 |                           270.819 |                                          272.916 |                                 277.208 |    278.046 |          276.821 |         270.652 |        269.352 |         269.983 |                270.298 |                                277.321 |         284.309 |               279.456 |   266.023 |       272.265 |                    273.833 |                 269.328 |                       276.821 |                       274.272 |                285.211 |                 276.181 |                267.151 |         0.2267 |                0.0654 |       6.6568 |               4.9986 |      24.6224 |      268.465 |                            1 |                272.441 |                 279.575 |                 284.635 |                  274.771 |                 264.907 |                  287.618 |                   279.158 |                  270.697 |                          0.0184 |              4.7648 |              -0.1054 |                     52.2511 |                   288.909 |                    276.181 |                   263.684 |                  282.999 |                 270.955 |                     283.909 |                      274.378 |                     264.847 |            278.441 |             273.429 |            268.418 | 263.457 |                      1 |          13.6624 |               4.6738 |                       2.0436 |            -1.5338 |         1 |                         18.2309 |                       303.757  |            -0.8123 |               1.9646 |        -0.8314 |             -0.195  |                    1 |                           276.799 |                               276.073 |                                    273.964 |                   -2.5273 |                272.939 |                -0.9568 |                     -1.5143 |               -0.6032 |                     0.1375 |                        -5.5252 |                     -5.5219 |         0      |
        | 2025-12-19 |                 0.0139 |                       1 |         2.46314e+09 |          6.67935e+07 |                      4.11448e+09 |          1.95656e+07 | nan      |                      0 |               0.1361 |            -9.4904 |                 2735.53 |                 794.15  |                -1.89072e+08 |                            -8.92323e+07 |                      3.2384 |                              174.315 |            3.3172 |                 27.1858 |                   0.149  |              0.0742 |                   0.3343 |              1.00928e+10 |    3.95814e+10 |                   1 |            1.2421 |             0.9738 |                             2.36697e+09 |                             2.34049e+09 |                        2.31851e+09 |                        2.46314e+09 |                                   1 |                                    0 |            43.3299 |      -68.989  |              14.2857 |                92.8571 |           -78.5714 |                 -103.239  |                -0.2694 |                            -0.2349 |       5.93143e+06 |               51.5414 |                        0.0037 |                                    0.0089 |                                 -0.0052 |                      1.0212 |                         0.1453 |                                    -0.0056 |                                   0.1509 |                      -0.4186 |                     23.9078 |                     11.4732 |                      16.2024 |                      16.469  |                       23.2575 |                      17.0882 |                            28.6043 |                     -39.3401 |                   28.2097 |                    272.6   |              276.008 |                   265.041 |                   250.519 |         31.011  |         18.7898 |      1.0212 |             2.4301 |          -1.4089 |                   49.0174 |             12.7427 |              5.0588 |                7.6467 |                           14.7723 |             0.3234 |                  0.8021 |  -0.8698 |              -1.1188 |                  -2.2692 |                 0.903  |                 1.137  |                -0.8132 |                -5.4969 |          -0.0333 |    -9.3972 |            41.2482 |             0.3934 |                       0.5987 |            0.2645 |                        0.4504 |         -0.0269 |            -2.3555 |                    -2.6249 |                        -0.9292 | 32.8313 | 27.4511 | 43.5917 |                         -0.7977 |                     -0.9462 |                                 17.0473 |                 -9.7979 |                   -3.5632 |                  -6.2347 |               0.3291 |    -16.4302 |                 -6.2154 |                 -6.63   |       14.9273 |                    5880.32 | 55.8245 |                  49.1536 |                  57.0396 |                 13.6936 |       64.4303 |   49.4718 | -0.007  |  113.263  |  110.127  |                  -0.7701 |              41.6667 |                     -2.0674 |               0 |                1 |          -1 |                       -0.9192 |                0.0092 |                  0.7123 |                       276.636 |                            274.498 |                                    274.194 | 0.0012 |                     276.433 |                         274.645 |                     271.147 |                                    271.835 |                                       272.223 |                                  273.19  |                                 276.385 |                                    272.635 |                                   276.649 |                                  273.957 |                          278.852 |                               273.059 |                                 276.053 |            271.981 |                                  270.755 |                                   272.783 |                              273.865 |                           277.958 |                           269.773 |                                          272.876 |                                 276.066 |    278.046 |          276.821 |         271.634 |        271.305 |         271.777 |                272.012 |                                276.002 |         282.481 |               278.475 |   266.023 |       269.983 |                    273.942 |                 267.286 |                       276.821 |                       274.272 |                284.285 |                 276.55  |                268.816 |         0.2524 |                0.0559 |       4.6837 |               4.9761 |      24.6659 |      268.465 |                            1 |                272.493 |                 279.704 |                 284.312 |                  274.498 |                 264.683 |                  287.618 |                   276.821 |                  266.023 |                          0.0182 |              4.5812 |              -0.036  |                     54.7332 |                   288.375 |                    276.55  |                   264.349 |                  282.999 |                 270.93  |                     283.524 |                      273.768 |                     264.013 |            277.346 |             272.356 |            267.367 | 263.457 |                      1 |           7.8924 |               0.9667 |                       1.95   |            -3.209  |         1 |                         18.064  |                       -35.1903 |            -0.6121 |              -1.6203 |        -0.5763 |              0.1704 |                    1 |                           276.956 |                               274.397 |                                    274.072 |                   -2.0487 |                272.161 |                -0.9881 |                     -1.274  |               -0.2922 |                     0.4696 |                        -5.5254 |                     -5.5252 |         0      |
        | 2025-12-22 |                -0.0419 |                      -1 |         2.42657e+09 |          6.64327e+07 |                      4.08789e+09 |          2.21089e+07 | nan      |                      0 |               0.079  |           -12.8267 |                 2708.55 |                 794.15  |                -1.99306e+08 |                            -1.04957e+08 |                      9.1827 |                              171.415 |            0.7534 |                 27.6286 |                   0.1149 |              0.0461 |                   0.2145 |              1.00828e+10 |    9.90986e+09 |                   4 |            1.1372 |             0.9639 |                             2.39081e+09 |                             2.35373e+09 |                        2.42657e+09 |                        2.46314e+09 |                                   1 |                                    0 |            36.4312 |      -81.4487 |               7.1429 |                85.7143 |           -78.5714 |                 -104.733  |                -0.2367 |                            -0.2614 |      -7.97965e+06 |               45.7604 |                        0.0022 |                                    0.0076 |                                 -0.0054 |                      0.5976 |                         0.1018 |                                     0.0158 |                                   0.0859 |                       2.2593 |                     23.4207 |                     10.6537 |                      15.0451 |                      15.6562 |                       22.1097 |                      17.0882 |                            29.1456 |                     -64.5455 |                   17.6515 |                    272.6   |              276.008 |                   264.954 |                   250.519 |         18.5513 |         24.581  |      0.5976 |             2.0636 |          -1.466  |                   44.0574 |              0      |              5.0588 |                4.6682 |                           12.2463 |            -0.5608 |                  0.1365 |  -1.2314 |              -2.0902 |                  -2.1525 |                 0.8075 |                 1.2147 |                -0.8978 |                -4.2561 |          -0.0532 |   -15.1671 |            40.4149 |             0.6455 |                       0.5589 |            0.2055 |                        0.4111 |         -0.0418 |            -2.0419 |                    -2.3555 |                        -0.9498 | 32.0392 | 28.9805 | 38.1567 |                         -0.7692 |                     -4.8407 |                                 12.6697 |                -13.7773 |                   -6.9679 |                  -6.8094 |               0.1646 |    -20.9318 |                -13.0815 |                 -6.456  |      -45.6961 |                    5834.63 | 54.1419 |                  47.4549 |                  37.2576 |                 -1.2512 |       20.8244 |   48.1033 | -0.0165 |  110.181  |  102.622  |                  -1.0926 |              41.6667 |                     -1.9446 |               0 |                1 |          -1 |                       -0.9246 |                0.0011 |                  0.5069 |                       275.553 |                            273.902 |                                    273.122 | 0.0009 |                     275.578 |                         273.764 |                     270.399 |                                    270.666 |                                       270.96  |                                  272.964 |                                 275.489 |                                    271.634 |                                   275.978 |                                  273.151 |                          278.443 |                               272.238 |                                 275.011 |            271.838 |                                  270.713 |                                   272.504 |                              272.454 |                           276.493 |                           268.414 |                                          272.723 |                                 275.059 |    276.596 |          276.821 |         271.111 |        271.25  |         270.843 |                270.64  |                                275.063 |         280.835 |               277.345 |   266.023 |       271.777 |                    274.59  |                 269.906 |                       276.821 |                       274.272 |                284.343 |                 276.525 |                268.707 |         0.0846 |                0.0565 |       3.3583 |               4.8605 |      24.609  |      268.465 |                            1 |                272.722 |                 279.644 |                 283.406 |                  273.902 |                 264.397 |                  287.618 |                   276.821 |                  266.023 |                          0.018  |              4.4815 |              -0.0369 |                     51.5807 |                   287.967 |                    276.525 |                   265.009 |                  282.999 |                 270.776 |                     282.464 |                      273.023 |                     263.583 |            276.457 |             271.577 |            266.696 | 263.457 |                      1 |           5.6404 |               0      |                       1.7804 |            -5.0602 |         1 |                         17.9853 |                       -17.9254 |            -2.3868 |              -3.8682 |        -0.3078 |              0.4552 |                    1 |                           276.873 |                               274.178 |                                    274.075 |                   -1.347  |                271.642 |                -0.9995 |                     -1.4735 |                0.1789 |                     0.8222 |                        -5.5263 |                     -5.5254 |         0      |
        | 2025-12-23 |                 0.0106 |                       1 |         2.45621e+09 |          6.65847e+07 |                      4.11471e+09 |          2.96727e+07 | nan      |                      0 |               0.0651 |           -14.4654 |                 2722.44 |                 794.15  |                -2.24884e+08 |                            -1.22089e+08 |                      9.8839 |                              174.205 |            0.6251 |                 27.8548 |                   0.1894 |              0.071  |                   0.2753 |              1.00822e+10 |    8.07329e+09 |                   2 |            1.0956 |             0.8755 |                             2.41697e+09 |                             2.3695e+09  |                        2.42657e+09 |                        2.45621e+09 |                                   1 |                                    0 |            37.2984 |      -69.5725 |               0      |                78.5714 |           -78.5714 |                  -97.9344 |                -0.205  |                            -0.2501 |      -1.44128e+06 |               49.1967 |                        0.0014 |                                    0.0063 |                                 -0.005  |                      0.3694 |                         0.0548 |                                     0.0236 |                                   0.0311 |                       2.351  |                     23.1944 |                      9.8927 |                      14.9172 |                      14.9624 |                       22.5619 |                      20.2521 |                            28.9735 |                     -51.4169 |                   23.8077 |                    272.6   |              276.008 |                   265.083 |                   250.519 |         30.4275 |         26.6632 |      0.3694 |             1.7248 |          -1.3553 |                   47.029  |             15.5282 |              9.4236 |                3.0048 |                            9.9359 |             0.517  |                  0.9524 |  -0.9773 |              -2.6623 |                  -1.7431 |                 0.7732 |                 1.2661 |                -1.9284 |                -4.8582 |          -0.0415 |   -11.7491 |            46.4518 |             0.5142 |                       0.4243 |            0.1522 |                        0.3663 |         -0.0501 |            -1.9612 |                    -2.0419 |                        -0.9406 | 35.0211 | 30.994  | 43.0753 |                         -0.727  |                     -6.7949 |                                  8.7768 |                -13.4496 |                   -9.1285 |                  -4.3211 |               0.0823 |    -24.6243 |                -18.6022 |                 -5.7342 |       12.4911 |                    5847.12 | 53.9827 |                  47.3129 |                  56.8024 |                 12.1426 |       57.2709 |   47.6333 | -0.0114 |  105.311  |  103.415  |                  -0.6584 |              50      |                     -1.3037 |               0 |                1 |          -1 |                       -0.9141 |                0.0076 |                  0.7178 |                       274.714 |                            273.57  |                                    272.607 | 0.0007 |                     274.764 |                         273.212 |                     269.985 |                                    270.319 |                                       270.788 |                                  272.853 |                                 274.716 |                                    271.545 |                                   275.004 |                                  272.908 |                          277.949 |                               271.772 |                                 274.32  |            271.808 |                                  270.755 |                                   272.185 |                              271.5   |                           275.349 |                           267.651 |                                          272.665 |                                 274.38  |    274.877 |          274.882 |         270.373 |        270.089 |         270.531 |                270.752 |                                274.256 |         279.354 |               276.408 |   266.023 |       270.843 |                    272.115 |                 268.757 |                       274.882 |                       272.792 |                284.455 |                 276.347 |                268.24  |         0.1958 |                0.0587 |       2.9298 |               4.7226 |      24.5236 |      268.465 |                            1 |                273     |                 279.366 |                 282.71  |                  273.57  |                 264.43  |                  287.618 |                   276.821 |                  266.023 |                          0.0174 |              4.3121 |              -0.0391 |                     54.401  |                   287.422 |                    276.347 |                   265.252 |                  282.999 |                 270.593 |                     281.343 |                      272.509 |                     263.674 |            275.83  |             271.08  |            266.329 | 263.457 |                      1 |           4.4745 |               0.9467 |                       1.7079 |            -6.1264 |         1 |                         18.0561 |                        -3.6741 |            -3.886  |              -1.9042 |        -0.0641 |              0.6603 |                    1 |                           276.637 |                               274.04  |                                    274.074 |                   -0.5972 |                271.196 |                -0.9999 |                     -1.2831 |                0.8774 |                     0.9597 |                        -5.5259 |                     -5.5263 |         0      |
        | 2025-12-24 |                 0.0554 |                       1 |         2.47412e+09 |          6.66801e+07 |                      4.11466e+09 |          3.00518e+07 | nan      |                      0 |               0.0846 |            -6.6454 |                 2736.93 |                 794.15  |                -2.1628e+08  |                            -1.35545e+08 |                     17.9713 |                              175.81  |            0.3926 |                 27.2474 |                   0.156  |              0.0697 |                   0.3125 |              1.00864e+10 |    4.9041e+09  |                   2 |            1.0156 |             0.8174 |                             2.43983e+09 |                             2.38559e+09 |                        2.45621e+09 |                        2.47412e+09 |                                   1 |                                    0 |            41.6581 |      -51.6561 |               0      |                71.4286 |           -71.4286 |                  -34.6705 |                -0.1841 |                            -0.2228 |       2.20161e+06 |               54.4448 |                        0.0011 |                                    0.0053 |                                 -0.0042 |                      0.3017 |                        -0.0045 |                                     0.018  |                                  -0.0225 |                       3.0907 |                     22.018  |                     12.1059 |                      13.8517 |                      18.736  |                       21.4379 |                       6.7256 |                            28.2487 |                     -32.919  |                   31.7545 |                    272.6   |              276.008 |                   265.063 |                   250.519 |         48.3439 |         32.4409 |      0.3017 |             1.4402 |          -1.1385 |                   50.0121 |             38.5455 |             18.0246 |                2.4525 |                            8.0651 |             0.4551 |                  0.4985 |  -0.6307 |              -2.8188 |                  -1.0827 |                 0.8892 |                 1.1613 |                 1.0804 |                -2.1383 |          -0.0245 |    -6.8661 |            53.1324 |             0.3292 |                       0.3731 |            0.1033 |                        0.3168 |         -0.0518 |            -1.5593 |                    -1.9612 |                        -0.8894 | 40.6707 | 34.2196 | 53.573  |                         -0.6448 |                     -7.0164 |                                  5.6181 |                -10.3613 |                   -9.5394 |                  -0.8219 |               0.0411 |    -22.4462 |                -21.1081 |                 -4.3189 |       36.2295 |                    5883.35 | 53.9827 |                  48.2126 |                  42.5356 |                 23.8305 |       67.2632 |   47.9699 | -0.0071 |  112.863  |  120.363  |                  -0.2751 |              58.3333 |                     -0.8397 |               0 |                1 |           1 |                       -0.852  |                0.0123 |                  0.8331 |                       274.223 |                            273.476 |                                    272.559 | 0.0006 |                     274.049 |                         272.965 |                     270.161 |                                    270.616 |                                       270.958 |                                  272.854 |                                 274.088 |                                    272.044 |                                   273.884 |                                  272.904 |                          277.412 |                               271.704 |                                 273.909 |            271.882 |                                  270.834 |                                   272.353 |                              270.973 |                           274.704 |                           267.242 |                                          272.67  |                                 273.974 |    273.921 |          273.094 |         272.496 |        272.864 |         272.863 |                272.862 |                                273.798 |         278.021 |               275.97  |   266.023 |       270.531 |                    272.438 |                 269.508 |                       273.094 |                       271.425 |                284.438 |                 276.19  |                267.941 |         0.2981 |                0.0597 |       3.2188 |               4.6152 |      24.4118 |      268.465 |                            1 |                273.225 |                 280.416 |                 282.345 |                  273.476 |                 264.606 |                  283.741 |                   274.882 |                  266.023 |                          0.0169 |              4.105  |              -0.037  |                     56.7182 |                   286.983 |                    276.19  |                   265.287 |                  282.999 |                 270.458 |                     280.619 |                      272.348 |                     264.077 |            275.848 |             271.199 |            266.551 | 263.457 |                      1 |           4.9926 |               2.9198 |                       1.8133 |            -6.7609 |         1 |                         18.3028 |                         9.1699 |            -4.0014 |              -0.3151 |         0.1594 |              0.8108 |                    1 |                           276.361 |                               273.981 |                                    274.072 |                    0.2127 |                271.251 |                -0.9962 |                     -0.8825 |                0.6824 |                    -0.0344 |                        -5.519  |                     -5.5259 |         0.1145 |
        | 2025-12-26 |                -0.0066 |                      -1 |         2.4526e+09  |          6.66479e+07 |                      4.10239e+09 |          2.35736e+07 | nan      |                      0 |               0.0857 |            -5.4396 |                 2736.93 |                 792.961 |                -1.90374e+08 |                            -1.43378e+08 |                     11.6221 |                              173.846 |            0.4872 |                 26.5611 |                   0.0943 |              0.0554 |                   0.2254 |              1.00882e+10 |    5.88406e+09 |                   3 |            0.9294 |             0.7778 |                             2.44494e+09 |                             2.3959e+09  |                        2.4526e+09  |                        2.47412e+09 |                                   1 |                                    0 |            46.9441 |      -51.1363 |               0      |                64.2857 |           -64.2857 |                  -25.6932 |                -0.1697 |                            -0.1974 |  735552           |               51.2159 |                        0.0008 |                                    0.0044 |                                 -0.0036 |                      0.2126 |                        -0.0488 |                                     0.0046 |                                  -0.0534 |                      -0.6819 |                     20.9257 |                     11.2412 |                      12.8623 |                      17.9861 |                       20.58   |                       6.7256 |                            27.2889 |                     -27.7033 |                   33.1185 |                    272.6   |              276.098 |                   264.961 |                   250.519 |         48.8637 |         42.545  |      0.2126 |             1.1946 |          -0.982  |                   49.1689 |             34.6217 |             29.5651 |                1.7619 |                            6.4893 |            -0.3028 |                  0.2151 |  -0.5595 |              -2.1658 |                   0.0054 |                 0.9091 |                 1.1011 |                 1.1553 |                -1.346  |          -0.0193 |    -5.3614 |            55.49   |             0.277  |                       0.4022 |            0.0564 |                        0.2631 |         -0.0522 |            -1.1484 |                    -1.5593 |                        -0.8224 | 43.4017 | 37.2803 | 55.6445 |                         -0.4625 |                     -6.9074 |                                  3.113  |                 -9.0825 |                   -9.3871 |                   0.3046 |               0.0206 |    -20.284  |                -22.0716 |                 -2.8918 |       -3.8045 |                    5879.54 | 53.9827 |                  48.5314 |                  33.7903 |                  7.3734 |       39.8837 |   48.2486 | -0.0094 |  120.453  |  119.909  |                  -0.284  |              58.3333 |                     -0.7927 |               1 |                0 |           1 |                       -0.7986 |                0.0095 |                  0.7973 |                       273.84  |                            273.339 |                                    272.426 | 0.0004 |                     273.429 |                         272.729 |                     270.664 |                                    270.745 |                                       271.364 |                                  272.825 |                                 273.546 |                                    272.197 |                                   272.872 |                                  272.88  |                          276.858 |                               271.784 |                                 273.559 |            271.922 |                                  270.903 |                                   272.475 |                              270.423 |                           273.963 |                           266.884 |                                          272.663 |                                 273.628 |    273.921 |          272.6   |         272.996 |        273.163 |         272.926 |                272.807 |                                273.418 |         276.821 |               275.64  |   266.023 |       272.863 |                    274.471 |                 271.252 |                       272.6   |                       271.048 |                284.387 |                 275.983 |                267.579 |         0.2899 |                0.0609 |       2.5013 |               4.4642 |      24.2185 |      268.465 |                            1 |                273.538 |                 280.103 |                 281.822 |                  273.339 |                 264.856 |                  280.164 |                   273.094 |                  266.023 |                          0.0164 |              4.0637 |              -0.1593 |                     54.2874 |                   286.737 |                    275.983 |                   265.137 |                  282.851 |                 270.265 |                     279.904 |                      272.297 |                     264.69  |            276.293 |             271.788 |            267.282 | 263.457 |                      1 |           4.594  |               0      |                       1.6556 |            -7.2886 |         1 |                         18.6492 |                        13.3467 |            -4.0726 |               2.8793 |         0.2308 |              0.8512 |                    0 |                           276.11  |                               273.904 |                                    274.067 |                    1.0468 |                271.629 |                -0.9816 |                     -0.6725 |                0.9729 |                     0.5244 |                        -5.5128 |                     -5.519  |         0.1222 |
        | 2025-12-29 |                 0.0405 |                       1 |         2.47632e+09 |          6.66791e+07 |                      4.11195e+09 |          2.18213e+07 |   0.4593 |                      0 |               0.0756 |            -4.7955 |                 2736.93 |                 794.005 |                -1.915e+08   |                            -1.50253e+08 |                      8.4461 |                              175.251 |            0.5442 |                 26.2841 |                   0.0461 |              0.0642 |                   0.2857 |              1.00912e+10 |    6.49227e+09 |                   1 |            0.8723 |             0.7174 |                             2.45749e+09 |                             2.40827e+09 |                        2.4526e+09  |                        2.47632e+09 |                                   1 |                                    0 |            48.1203 |      -48.4089 |              35.7143 |                57.1429 |           -21.4286 |                  -30.5671 |                -0.1563 |                            -0.1782 |       1.7719e+06  |               57.3149 |                        0.0006 |                                    0.0036 |                                 -0.003  |                      0.169  |                        -0.0817 |                                    -0.0126 |                                  -0.0691 |                       0.1103 |                     20.0593 |                     10.4382 |                      12.4518 |                      17.3861 |                       20.7398 |                       8.7965 |                            26.5057 |                     -21.8633 |                   36.4164 |                    270.612 |              276.183 |                   265.068 |                   250.519 |         51.5911 |         49.5995 |      0.169  |             0.9895 |          -0.8205 |                   49.9666 |             40.0245 |             37.7306 |                1.4003 |                            5.217  |             0.5323 |                  0.7015 |  -0.4662 |              -2.0654 |                   0.2951 |                 0.9316 |                 1.0955 |                 0.2129 |                -1.7901 |          -0.0149 |    -4.1157 |            54.4742 |             0.2186 |                       0.4134 |            0.0118 |                        0.2104 |         -0.0496 |            -0.8195 |                    -1.1484 |                        -0.6882 | 53.5816 | 42.714  | 75.3166 |                         -0.1913 |                     -6.4822 |                                  1.194  |                 -7.5042 |                   -8.7594 |                   1.2553 |               0.0103 |    -19.6328 |                -21.7468 |                 -1.5792 |        5.5886 |                    5885.13 | 53.9827 |                  49.0098 |                  34.5775 |                 19.7029 |       57.1631 |   49.2247 | -0.0088 |  110.988  |  114.995  |                  -0.1531 |              58.3333 |                     -1.0766 |               1 |                0 |           1 |                       -0.7244 |                0.0103 |                  0.854  |                       273.546 |                            273.268 |                                    272.416 | 0.0003 |                     272.918 |                         272.591 |                     271.326 |                                    271.012 |                                       271.765 |                                  272.824 |                                 273.082 |                                    272.429 |                                   272.145 |                                  272.877 |                          276.31  |                               272.011 |                                 273.315 |            271.984 |                                  270.911 |                                   272.462 |                              270.03  |                           273.441 |                           266.62  |                                          272.668 |                                 273.386 |    273.921 |          272.6   |         272.341 |        272.406 |         272.541 |                272.608 |                                273.16  |         275.741 |               275.206 |   266.023 |       272.926 |                    273.939 |                 271.438 |                       272.6   |                       271.048 |                284.195 |                 275.729 |                267.264 |         0.3276 |                0.0614 |       2.003  |               4.2884 |      23.8779 |      268.465 |                            1 |                273.905 |                 279.736 |                 281.304 |                  273.268 |                 265.233 |                  279.177 |                   272.6   |                  266.023 |                          0.0157 |              3.9562 |              -0.1596 |                     55.4689 |                   286.405 |                    275.729 |                   265.055 |                  282.851 |                 270.041 |                     279.204 |                      272.345 |                     265.486 |            276.279 |             271.941 |            267.602 | 263.457 |                      1 |           3.6472 |               0.5082 |                       1.5559 |            -7.4802 |         1 |                         18.9903 |                        23.5748 |            -1.6754 |               4.2075 |         0.3999 |              0.9309 |                    0 |                           275.772 |                               273.85  |                                    274.062 |                    1.7167 |                272.038 |                -0.9626 |                     -0.4628 |                0.3714 |                    -0.3939 |                        -5.5043 |                     -5.5128 |         0.203  |
        | 2025-12-30 |                -0.0195 |                      -1 |         2.45418e+09 |          6.66241e+07 |                      4.10949e+09 |          1.83379e+07 |   1.5877 |                      0 |               0.0249 |            -5.2836 |                 2730.14 |                 794.005 |                -2.06929e+08 |                            -1.58349e+08 |                      8.1019 |                              174.255 |            0.506  |                 25.8195 |                   0.0467 |              0.06   |                   0.1863 |              1.0091e+10  |    6.04588e+09 |                   4 |            0.8069 |             0.6714 |                             2.45616e+09 |                             2.41534e+09 |                        2.45418e+09 |                        2.47632e+09 |                                   1 |                                    0 |            43.4593 |      -53.5607 |              28.5714 |                50      |           -21.4286 |                  -35.5284 |                -0.1288 |                            -0.1608 | -464807           |               54.1042 |                        0.0003 |                                    0.003  |                                 -0.0027 |                      0.0788 |                        -0.1122 |                                    -0.0325 |                                  -0.0797 |                      -2.3582 |                     19.2762 |                      9.6926 |                      11.6321 |                      16.8441 |                       20.2146 |                       9.095  |                            25.712  |                     -21.7389 |                   38.6752 |                    270.612 |              276.183 |                   265.068 |                   250.519 |         46.4393 |         48.9647 |      0.0788 |             0.8074 |          -0.7285 |                   48.4209 |             29.5551 |             34.7338 |                0.6843 |                            4.0838 |             0.15   |                  0.4444 |  -0.3773 |              -1.9448 |                   0.3866 |                 0.9145 |                 1.1096 |                 0.0857 |                -1.708  |          -0.0148 |    -4.0858 |            53.5704 |             0.2174 |                       0.4141 |           -0.0182 |                        0.1612 |         -0.0465 |            -0.4022 |                    -0.8195 |                        -0.6065 | 57.9069 | 47.7783 | 78.1641 |                          0.0877 |                     -6.3714 |                                 -0.3191 |                 -7.7102 |                   -8.4097 |                   0.6995 |               0.0051 |    -20.2725 |                -20.6589 |                 -0.7486 |       -3.2998 |                    5881.83 | 53.9827 |                  48.8135 |                  50.5959 |                  3.8001 |       36.7617 |   50.4639 | -0.0122 |  113.905  |  105.452  |                  -0.2379 |              58.3333 |                     -1.0873 |               0 |                1 |           1 |                       -0.6718 |                0.0069 |                  0.7814 |                       273.254 |                            273.117 |                                    272.247 | 0.0002 |                     272.531 |                         272.403 |                     271.798 |                                    271.015 |                                       271.736 |                                  272.774 |                                 272.681 |                                    272.315 |                                   271.738 |                                  272.849 |                          275.78  |                               272.173 |                                 273.113 |            271.995 |                                  270.932 |                                   272.442 |                              269.554 |                           272.853 |                           266.256 |                                          272.648 |                                 273.181 |    273.921 |          272.6   |         272.115 |        272.232 |         272.198 |                272.182 |                                272.911 |         274.769 |               274.771 |   266.023 |       272.541 |                    273.676 |                 271.673 |                       272.6   |                       271.048 |                283.299 |                 275.23  |                267.162 |         0.308  |                0.0586 |       1.7937 |               4.1102 |      23.5003 |      268.465 |                            1 |                274.284 |                 279.357 |                 280.707 |                  273.117 |                 265.526 |                  279.177 |                   272.6   |                  266.023 |                          0.0151 |              3.863  |              -0.3267 |                     54.0398 |                   285.385 |                    275.23  |                   265.399 |                  279.001 |                 269.818 |                     278.457 |                      272.336 |                     266.215 |            276.381 |             272.212 |            268.043 | 263.457 |                      1 |           4.2652 |               0.0698 |                       1.4266 |            -7.2914 |         0 |                         19.3292 |                        29.9642 |            -1.2127 |               1.7906 |         0.4995 |              0.9658 |                    1 |                           275.284 |                               272.991 |                                    273.794 |                    2.1545 |                272.316 |                -0.9522 |                     -0.423  |               -0.6187 |                    -0.993  |                        -5.4973 |                     -5.5043 |         0.2628 |
        | 2025-12-31 |                -0.0707 |                      -1 |         2.42688e+09 |          6.65022e+07 |                      4.08531e+09 |          7.5513e+06  | nan      |                      0 |              -0.0357 |            -5.2468 |                 2730.14 |                 790.458 |                -2.31239e+08 |                            -1.68762e+08 |                      7.0467 |                              172.441 |            0.6417 |                 25.8501 |                  -0.005  |              0.0314 |                   0.0688 |              1.00847e+10 |    7.42004e+09 |                   3 |            0.7332 |             0.659  |                             2.44445e+09 |                             2.41711e+09 |                        2.42688e+09 |                        2.45418e+09 |                                   0 |                                    1 |            38.2003 |      -62.8029 |              21.4286 |                42.8571 |           -21.4286 |                  -56.9136 |                -0.0887 |                            -0.1381 |      -4.82705e+06 |               46.4079 |                       -0.0003 |                                    0.0023 |                                 -0.0026 |                     -0.0897 |                        -0.1258 |                                    -0.0512 |                                  -0.0746 |                      -1.5169 |                     18.7176 |                      9.0003 |                      11.3294 |                      16.2589 |                       20.4663 |                      11.4565 |                            24.6539 |                     -37.4456 |                   34.0262 |                    270.248 |              276.821 |                   265.6   |                   250.867 |         37.1971 |         45.0758 |     -0.0897 |             0.628  |          -0.7176 |                   45.6902 |             12.0094 |             27.1964 |               -0.6702 |                            2.8953 |            -0.6218 |                  0.057  |  -0.536  |              -1.6049 |                   0.515  |                 0.9113 |                 1.1148 |                -0.0091 |                -1.9324 |          -0.0248 |    -6.896  |            52.5944 |             0.3745 |                       0.3956 |           -0.0452 |                        0.1157 |         -0.0431 |            -0.1073 |                    -0.4022 |                        -0.6548 | 57.905  | 51.1539 | 71.4071 |                          0.1423 |                     -7.1079 |                                 -1.6769 |                -10.1732 |                   -8.9975 |                  -1.1757 |               0.0026 |    -23.361  |                -20.8876 |                 -0.6816 |      -19.3794 |                    5862.45 | 53.9827 |                  47.7724 |                  43.3308 |                 -9.9432 |       24.882  |   52.0466 | -0.0166 |   99.0847 |   92.3301 |                  -0.3998 |              50      |                     -1.4421 |               0 |                1 |          -1 |                       -0.6071 |                0.0023 |                  0.6255 |                       272.762 |                            272.823 |                                    271.815 | 0.0001 |                     272.2   |                         272.091 |                     271.883 |                                    270.627 |                                       271.745 |                                  272.642 |                                 272.319 |                                    271.777 |                                   271.608 |                                  272.686 |                          275.262 |                               272.102 |                                 272.746 |            271.917 |                                  270.932 |                                   272.383 |                              268.893 |                           272.144 |                           265.643 |                                          272.568 |                                 272.8   |    273.672 |          272.6   |         271.641 |        271.768 |         271.484 |                271.342 |                                272.594 |         273.895 |               274.318 |   266.023 |       272.198 |                    273.062 |                 271.268 |                       272.6   |                       271.048 |                281.366 |                 274.516 |                267.666 |         0.2372 |                0.0499 |       1.9233 |               3.954  |      23.1557 |      268.465 |                            1 |                274.628 |                 279.013 |                 280.039 |                  272.823 |                 265.607 |                  279.177 |                   272.6   |                  266.023 |                          0.0146 |              3.7647 |              -0.3494 |                     52.7331 |                   284.415 |                    274.516 |                   265.137 |                  275.232 |                 269.631 |                     277.662 |                      272.143 |                     266.623 |            276.421 |             272.402 |            268.383 | 263.457 |                      1 |           2.6706 |               0.5281 |                       1.3485 |            -6.9011 |         1 |                         19.6066 |                        33.1912 |            -0.9892 |               3.5114 |         0.5474 |              0.9788 |                    1 |                           274.885 |                               272.887 |                                    273.772 |                    2.3417 |                272.229 |                -0.9772 |                     -0.5599 |               -0.7431 |                    -0.0523 |                        -5.4957 |                     -5.4973 |         0.3273 |
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
        cycle_indicators = self.collect_cycle_indicators(
            period=period, close_column=close_column, window=window
        )
        if not cycle_indicators.empty:
            collections.append(cycle_indicators)

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

    def _check_arguments(
        self, period: str, close_column: str | None = None, **windows: int | None
    ) -> None:
        """
        Checks the arguments the indicators share before one is calculated: every window is a
        whole number of at least one period and the close column is a price column of the
        historical data. A window longer than the data of the period leaves the indicator
        without values, which is logged once per call: the window parameter when the
        indicator has one, else its longest window. Within a collection the window is kept for
        the one message of the collection instead.

        Args:
            period (str): The period of the prices, e.g. "daily".
            close_column (str | None, optional): The close column the indicator uses, or None for
                an indicator without one. Defaults to None.
            **windows (int | None): The windows of the indicator by parameter name, None for a
                window the indicator chooses itself.

        Raises:
            InvalidArgumentError: For a window below one period or a close column the historical
                data does not have (a ValueError).
        """
        for name, window in windows.items():
            validate_window(name, window, 1)

        # Intraday indicators always use the traded close; an unknown period is reported by the indicator.
        if period not in PERIODS or period == "intraday":
            return

        historical_data = self._historical_data[period]
        if close_column is not None and not historical_data.empty:
            columns = historical_data.columns.get_level_values(0)
            check_option(
                "close_column",
                close_column,
                [column for column in PRICE_COLUMNS if column in columns],
            )

        # The window parameter is the indicator's own; the other windows of an indicator with
        # variants (such as the rank window of Connors' RSI) only matter to the variant using them.
        relevant = [windows["window"]] if "window" in windows else windows.values()
        longest = max((window for window in relevant if window), default=0)
        if longest <= len(historical_data):
            return

        self._window_too_long = True
        if self._collecting:
            self._long_windows.add(longest)
        else:
            # An indicator built from others checks the same window again; the call logs it once (report_missing_values).  # noqa: E501
            logger.warning(
                "A window of %s periods is longer than the %s %s periods of data, which leaves the "
                "indicator without values. Please choose a shorter window or a longer date range.",
                longest,
                len(historical_data),
                period,
            )

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
        check_period(period, PERIODS)
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
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
        group: str | dict | None = None,
        demeaned: bool = False,
        binning: str = "quantile",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Factor analysis (predictive power) of the technical indicators: whether the tickers that score high on an
        indicator go on to return more than those that score low.

        Per period, the information coefficient (IC) is the Spearman rank correlation between the indicator of the
        tickers at the end of the period and their return over the following horizon: 1 means the indicator ranks
        the later returns perfectly, -1 in reverse and 0 that it says nothing about them. The IC measures skill in
        Grinold and Kahn's fundamental law of active management, IR ≈ IC × √breadth: a small but consistent IC
        across many tickers and periods gives a high information ratio.

        The summary has a row per indicator and horizon with the mean IC, its standard deviation,
        the IC information ratio (mean divided by standard deviation), the t-statistic of the mean
        with its p-value, the skewness and excess kurtosis of the IC, the share of periods with a
        positive IC, the number of periods, the top minus bottom spread (the mean forward return of
        the highest quantile minus that of the lowest), the annualized alpha and the beta of the
        factor-weighted long-short portfolio against the equally weighted tickers, the mean turnover
        of the top and bottom quantiles and the mean rank autocorrelation of the indicator from one
        period to the next. With detail, it returns one part of the analysis per period, quantile or
        group for a single indicator.

        The formula is as follows:

        - IC(t) = Spearman rank correlation of f(i, t) and R(i, t, t + h) over the tickers i,
          with f the factor and R the forward return over the horizon h
        - Demeaned return: R(i, t) − mean R(j, t) over the tickers j (of the group of i)
        - Portfolio weight: w(i, t) = (f(i, t) − mean f(t)) / Σ_j |f(j, t) − mean f(t)|
        - Portfolio return: R_F(t) = Σ_i w(i, t) × R(i, t, t + h)
        - Alpha and beta: R_F(t) = α + β × R_U(t) + ε(t), with R_U(t) = mean R(i, t, t + h)
        - Annualized Alpha = (1 + α) ^ (periods per year / h) − 1
        - Per period: (1 + mean return) ^ (1 / h) − 1 and standard error / √h
        - Standard error of a quantile = standard deviation over the periods / √periods

        The indicators are calculated daily and the value on the last trading day of each
        period is compared with the returns from that day's close onwards, using the dividend
        and split adjusted close. An indicator in price units does not compare across tickers
        with different share prices, so a price line (a moving average, band, channel, stop or
        cloud line, and the Butterworth and Gaussian filters of the Swiss Army Knife) is used as
        the distance of the close from it, close / line - 1, and a price difference as a share
        of the close: the true range, the ATR and average day range, the price distance, the
        market thermometer, the squeeze momentum, the MACD, the momentum (absolute rate of
        change), the Qstick, the awesome and accelerator oscillators, the elder ray, the
        detrended price oscillator, the force index, the Hilbert phasor components, the
        high-pass and band-pass filters of the Swiss Army Knife and the detrended synthetic
        price. The collections evaluate every variant of an indicator with variants, such as
        each method of `get_moving_average`, labelled with the method name and its parameters.
        Each ticker keeps its own history: a ticker without
        a value or a return in a period is left out of that period only.

        Referenced against:
        - alphalens, factor_information_coefficient: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L28
        - alphalens, mean_information_coefficient: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L80
        - alphalens, mean_return_by_quantile: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L461
        - alphalens, factor_weights: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L131
        - alphalens, factor_returns: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L210
        - alphalens, factor_alpha_beta: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L265
        - alphalens, quantile_turnover: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L575
        - alphalens, factor_rank_autocorrelation: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L620

        Shares similar logic:
        - alphalens, average_cumulative_return_by_quantile: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L758

        Args:
            metrics (str | list[str], optional): An indicator method name, e.g.
                "get_relative_strength_index", a variant of one by its label, the method with
                its parameters as keyword arguments, e.g. "get_relative_strength_index(method='connors')"
                or "get_moving_average(method='hull', window=10)", a list of them, or a collection: "momentum",
                "overlap", "volatility", "breadth", "cycle" or "all". Defaults to "momentum",
                the oscillators and momentum indicators. The collections leave out the market-wide
                breadth measures, the volume-scaled indicators, the pivot points and
                Fibonacci levels and the Hilbert dominant cycle phase and trend mode.
            period (str, optional): The frequency at which the indicators are evaluated. Options are:
                "daily" (daily prices), "weekly" (weekly prices), "monthly" (monthly prices),
                "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            quantiles (int, optional): The number of quantiles the tickers are sorted into per
                period. Defaults to 5.
            periods_apart (int, optional): The number of periods between the dates the
                quantile turnover and the rank autocorrelation compare. Defaults to 1.
            detail (str | None, optional): Returns one part of the analysis of a single indicator instead of the
                summary. Options are: "information_coefficient" (the IC per period and horizon),
                "information_coefficient_by_month" (the mean IC per calendar month),
                "information_coefficient_by_year" (the mean IC per calendar year),
                "group_information_coefficient" (the IC statistics within each group, needs group),
                "quantile_returns" (the mean forward return per quantile and horizon),
                "quantile_returns_by_date" (the mean forward return per quantile and horizon, per period),
                "quantile_return_statistics" (the mean forward return, its standard error and both per single
                period), "quantile_statistics" (the range, mean and count of the factor values in each
                quantile), "quantile_turnover" (the turnover per period and quantile),
                "rank_autocorrelation" (the rank autocorrelation per period), "factor_weights" (the weights of
                the factor-weighted long-short portfolio per period and ticker), "factor_returns" (its return
                per period and horizon), "cumulative_returns" (the growth of 1 in it and in each quantile),
                "event_returns" (the mean price path of each quantile from the longest horizon before to the
                longest horizon after the periods it is formed). Defaults to None.
            column (str | None, optional): The output to use of an indicator with several,
                e.g. "Signal Line" of the MACD or "Upper Band" of the Bollinger Bands.
                Defaults to None, which uses the first output in alphabetical order other
                than a price, a histogram or a signal line, e.g. "MACD Line" or "Lower Band".
            group (str | dict | None, optional): The group of each ticker for the group options:
                "sector" or "industry" (or another company profile item) from the company profiles
                of Financial Modeling Prep, or a dictionary of tickers and their groups. Defaults to
                None.
            demeaned (bool, optional): Whether to compare the forward returns with the mean of
                the tickers in each period, or with group with the mean of the ticker's group:
                a group-neutral analysis, in which the IC is group-adjusted and the
                factor-weighted portfolio neutral within each group. Without group the IC
                stays the same, as subtracting one mean from every ticker keeps their ranks.
                Defaults to False.
            binning (str, optional): How the tickers are sorted into quantiles per period. Options are:
                "quantile" (equally sized), "equal_width" (bins of equal width between the lowest and highest
                value), "zero_aware" (half of the quantiles for the negative and half for the zero and positive
                values, for a signal centred on zero; needs an even number of quantiles), "group" (equally sized
                within each group, needs group). Defaults to "quantile".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            **kwargs: Parameters passed to the indicators that take them, e.g. window=14,
                close_column="Close" or method="connors" for a single indicator. A variant in a
                collection keeps its own method.
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
            - The groups from the company profiles are today's sector or industry, applied
              to every period.
            - The factor-weighted portfolio is long-short: its weights sum to zero and their
              absolute values to one. A horizon longer than one period makes its returns
              overlap too, so its alpha and beta are averages rather than tests.
            - The cumulative returns split the capital over as many portfolios as the horizon
              has periods, started on consecutive periods and each rebalanced once per
              horizon; a value shows when its holding period ends, so the series starts at 1.
            - The IC is that of Grinold, R.C. and Kahn, R.N. (2000). "Active Portfolio
              Management." 2nd ed. McGraw-Hill; the factor-weighted portfolio, the demeaned
              and group-neutral returns that of Qian, E., Hua, R. and Sorensen, E. (2007).
              "Quantitative Equity Portfolio Management." Chapman & Hall/CRC; the alpha that
              of Jensen, M.C. (1968). "The Performance of Mutual Funds in the Period
              1945-1964." Journal of Finance 23(2); the event returns follow MacKinlay, A.C.
              (1997). "Event Studies in Economics and Finance." Journal of Economic Literature
              35(1).
            - Unlike alphalens, the top minus bottom spread is twice alphalens' (a gross
              exposure of two rather than one), each horizon uses its own returns where alphalens
              drops a period and ticker without every horizon's return, and the event returns are
              the price paths rebased to the period itself where alphalens averages the
              single-period returns at each offset.

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

        |                                     |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   P-Value |   IC Skewness |   IC Kurtosis |   Positive Share |   Observations |   Top Minus Bottom |   Annualized Alpha |    Beta |   Quantile Turnover |   Rank Autocorrelation |
        |:------------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|--------------:|--------------:|-----------------:|---------------:|-------------------:|-------------------:|--------:|--------------------:|-----------------------:|
        | ('get_rate_of_change', '1 Month')   |    0.0468 |                  0.3467 |                 0.1349 |        1.4713 |    0.1439 |       -0.3812 |       -0.3944 |           0.5798 |            119 |             0.0091 |             0.1115 |  0.0444 |              0.2637 |                 0.8582 |
        | ('get_rate_of_change', '12 Months') |    0.1184 |                  0.3988 |                 0.2968 |        3.0848 |    0.0026 |       -0.4661 |       -0.3225 |           0.6389 |            108 |             0.079  |            -0.0334 |  0.6091 |              0.2637 |                 0.8582 |
        | ('get_moving_average', '1 Month')   |    0.0382 |                  0.346  |                 0.1104 |        1.204  |    0.231  |       -0.2546 |       -0.7156 |           0.563  |            119 |             0.0125 |             0.1154 | -0.0278 |              0.3074 |                 0.8036 |
        | ('get_moving_average', '12 Months') |    0.102  |                  0.3847 |                 0.2651 |        2.7545 |    0.0069 |       -0.3393 |       -0.5643 |           0.5833 |            108 |             0.0853 |            -0.0393 |  0.5512 |              0.3074 |                 0.8036 |
        """
        tickers = self._tickers
        selected = factor_model.select_metrics(metrics, FACTOR_COLLECTIONS)
        factors = factor_model.get_daily_factors(
            self, selected, kwargs, {}, tickers, column
        )

        close = factor_model.to_daily_factor(
            self._historical_data["daily"][kwargs.get("close_column", "Adj Close")],
            tickers,
        )
        for metric in selected:
            label = factor_model.get_metric_label(metric)
            if label not in factors:
                continue
            factor = factors[label]

            # The unit follows the method and, for a method with variants, the variant
            # calculated: per parameter the value of the metric, the one given or the
            # method's default, e.g. the method of a moving average or the filter of the
            # Swiss Army Knife.
            name, variant = (metric, {}) if isinstance(metric, str) else metric
            arguments = {
                key: variant.get(key, kwargs.get(key, parameter.default))
                for key, parameter in inspect.signature(
                    getattr(self, name)
                ).parameters.items()
            }

            # The Supertrend's direction is +1 or -1 and stays as it is.
            if (
                is_factor_listed(FACTOR_PRICE_LINES, name, arguments)
                and column != "Trend Direction"
            ):
                factors[label] = close.reindex_like(factor) / factor - 1
            elif is_factor_listed(FACTOR_PRICE_DIFFERENCES, name, arguments):
                factors[label] = factor / close.reindex_like(factor)

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
            binning=binning,
            demeaned=demeaned,
            groups=factor_model.get_groups(group, self._profile_source, self._tickers),
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
    CycleTechnicals,
    LabelingTechnicals,
    TechnicalsBase,
):
    """
    The Technicals Module contains the technical indicators of the Toolkit, grouped by topic
    (breadth, momentum, overlap, volatility, patterns, cycles and more) and calculated from the
    historical data of the Toolkit, with the labels and sample weights of financial machine
    learning (López de Prado).

    The indicators are those of the technical analysis literature, such as Wilder, Appel, Lane,
    Bollinger, Chande, Ehlers and Nison, whose work each method cites. Their calculation is
    referenced against the established libraries: TA-Lib (Mario Fortier), pandas-ta-classic (the
    continuation of Kevin Johnson's pandas-ta), ta (Dario Lopez Padial), finta (Peerchemist) and
    Tulip Indicators (Lewis Van Winkle). Where an indicator is defined by one of these libraries,
    or where a method deliberately differs from them (such as the seeding of an average), its
    docstring says so.
    """
