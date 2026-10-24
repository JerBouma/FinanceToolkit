"""Technicals Module"""

__docformat__ = "google"

import warnings

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.technicals import (
    breadth_model,
    momentum_model,
    overlap_model,
    volatility_model,
)
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    PERIODS_PER_YEAR,
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-public-methods,too-many-locals,eval-used
# pylint: disable=too-many-boolean-expressions
# The examples' output tables are wider than the line length.
# ruff: noqa: E501

# The default number of periods the Stochastic Oscillator's %K line is smoothed over to obtain the %D signal line, named so the deprecated `smooth_widow` alias can tell an explicitly passed `smooth_window` apart from the untouched default.  # noqa: E501
DEFAULT_STOCHASTIC_SMOOTH_WINDOW = 3


class Technicals:
    """
    The Technicals Module contains 50+ Technical Indicators that can
    be used to analyse companies. These ratios are divided into 4
    categories which are breadth, momentum, overlap and volatility.
    Each indicator is calculated using the data from the Toolkit module.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: dict[str, pd.DataFrame],
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Technicals Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the calculation.
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
        self._tickers = tickers
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates all Technical Indicators based on the data provided.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "daily".
            window (int, optional): The number of days to use for the calculation. Defaults to 14.
            close_column (str, optional): The column to use for the calculation. Defaults to "Adj Close".
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        | Date       |   McClellan Oscillator |   Advancers - Decliners |   On-Balance Volume |   Accumulation/Distribution Line |   Chaikin Oscillator |     TRIN |   New Highs - New Lows |   Chaikin Money Flow |   Ease of Movement |   Negative Volume Index |   Positive Volume Index |   Money Flow Index |   Williams %R |   Aroon Indicator Up |   Aroon Indicator Down |   Commodity Channel Index |   Relative Vigor Index |   Relative Vigor Index Signal Line |       Force Index |   Ultimate Oscillator |   Percentage Price Oscillator |   Detrended Price Oscillator |   Average Directional Index |   Chande Momentum Oscillator |   Ichimoku Conversion Line |   Ichimoku Base Line |   Ichimoku Leading Span A |   Ichimoku Leading Span B |   Stochastic %K |   Stochastic %D |   MACD Line |   MACD Signal Line |   Relative Strength Index |   Balance of Power |   Awesome Oscillator |   Vortex Indicator VI+ |   Vortex Indicator VI- |   Elder Ray Bull Power |   Elder Ray Bear Power |   Rate of Change |   Choppiness Index |   Know Sure Thing |   Know Sure Thing Signal Line |   Simple Moving Average (SMA) |   Exponential Moving Average (EMA) |   Double Exponential Moving Average (DEMA) |   TRIX |   Triangular Moving Average |   Weighted Moving Average (WMA) |   Hull Moving Average (HMA) |   Kaufman Adaptive Moving Average (KAMA) |   Volume Weighted Average Price (VWAP) |   Parabolic SAR |   Pivot Point |   Pivot Point Resistance 1 |   Pivot Point Support 1 |   Fibonacci Retracement 50.0% |   Fibonacci Retracement 61.8% |   Bollinger Band Upper |   Bollinger Band Middle |   Bollinger Band Lower |   True Range |   Average True Range |   Supertrend |   Supertrend Trend Direction |   Keltner Channel Upper |   Keltner Channel Middle |   Keltner Channel Lower |   Donchian Channel Upper |   Donchian Channel Middle |   Donchian Channel Lower |
        |:-----------|-----------------------:|------------------------:|--------------------:|---------------------------------:|---------------------:|---------:|-----------------------:|---------------------:|-------------------:|------------------------:|------------------------:|-------------------:|--------------:|---------------------:|-----------------------:|--------------------------:|-----------------------:|-----------------------------------:|------------------:|----------------------:|------------------------------:|-----------------------------:|----------------------------:|-----------------------------:|---------------------------:|---------------------:|--------------------------:|--------------------------:|----------------:|----------------:|------------:|-------------------:|--------------------------:|-------------------:|---------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------:|-------------------:|------------------:|------------------------------:|------------------------------:|-----------------------------------:|-------------------------------------------:|-------:|----------------------------:|--------------------------------:|----------------------------:|-----------------------------------------:|---------------------------------------:|----------------:|--------------:|---------------------------:|------------------------:|------------------------------:|------------------------------:|-----------------------:|------------------------:|-----------------------:|-------------:|---------------------:|-------------:|-----------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|-------------------------:|
        | 2025-12-17 |                -0.0981 |                      -1 |         2.26688e+09 |                      3.99772e+09 |         -1.74845e+07 | nan      |                      0 |              -0.0358 |            -4.7237 |                 2735.56 |                 788.833 |            39.7393 |      -98.8221 |              28.5714 |               100      |                 -139.905  |                -0.2262 |                            -0.0758 |      -3.27663e+07 |               44.5333 |                        0.6309 |                       0.0306 |                     24.6712 |                     -20.6214 |                    275.64  |              276.218 |                   265.243 |                   250.709 |          1.1779 |          8.6606 |      1.7276 |             3.162  |                   45.1866 |            -0.7013 |               1.2663 |                 0.9211 |                 1.0533 |                -0.1329 |                -4.6407 |          -2.0573 |            49.8428 |           36.3832 |                       51.073  |                       277.993 |                            275.523 |                                    275.977 | 0.1587 |                     278.193 |                         276.338 |                     273.323 |                                  274.426 |                                278.186 |         285.71  |       273.223 |                    275.394 |                 271.694 |                       279.37  |                       277.371 |                285.486 |                 277.993 |                270.499 |       4.5077 |               4.8747 |      268.669 |                            1 |                 285.011 |                  275.523 |                 266.035 |                  287.837 |                   279.444 |                  271.052 |
        | 2025-12-18 |                -0.0378 |                       1 |         2.31851e+09 |                      4.02709e+09 |         -1.16785e+07 | nan      |                      0 |               0.0024 |            -7.4205 |                 2735.56 |                 789.848 |            34.4002 |      -75.8191 |              21.4286 |               100      |                 -162.429  |                -0.2832 |                            -0.1659 |      -2.59949e+07 |               50.981  |                        0.4656 |                      -0.3825 |                     24.4323 |                     -24.9067 |                    272.807 |              276.218 |                   265.243 |                   250.709 |         24.1809 |         14.0382 |      1.2741 |             2.7844 |                   45.9182 |            -0.2126 |               0.01   |                 0.865  |                 1.0802 |                -2.0715 |                -8.7334 |          -2.3884 |            42.7376 |           31.0981 |                       48.3448 |                       277.518 |                            274.98  |                                    274.903 | 0.1367 |                     277.449 |                         275.466 |                     272.107 |                                  274.289 |                                277.532 |         284.525 |       272.472 |                    274.041 |                 269.533 |                       277.031 |                       274.481 |                285.733 |                 277.518 |                269.304 |       6.6619 |               5.0024 |      268.669 |                            1 |                 284.851 |                  274.98  |                 265.109 |                  287.837 |                   279.37  |                  270.903 |
        | 2025-12-19 |                 0.0139 |                       1 |         2.46314e+09 |                      4.11449e+09 |          1.95658e+07 | nan      |                      0 |               0.1361 |            -9.5047 |                 2735.56 |                 794.143 |            43.3299 |      -68.9888 |              14.2857 |                92.8571 |                 -103.239  |                -0.2694 |                            -0.2349 |       5.9366e+06  |               51.5415 |                        0.3736 |                      -0.4188 |                     23.9077 |                     -39.3408 |                    272.807 |              276.218 |                   265.243 |                   250.709 |         31.0112 |         18.79   |      1.022  |             2.4319 |                   49.0175 |             0.3234 |              -1.1196 |                 0.903  |                 1.137  |                -0.8138 |                -5.5011 |          -3.3309 |            41.2483 |           26.4483 |                       45.0423 |                       276.846 |                            274.706 |                                    274.402 | 0.116  |                     276.642 |                         274.854 |                     271.353 |                                  274.166 |                                276.211 |         282.695 |       270.188 |                    274.15  |                 267.489 |                       277.031 |                       274.481 |                284.913 |                 276.846 |                268.78  |       4.6872 |               4.9799 |      268.669 |                            1 |                 284.528 |                  274.706 |                 264.885 |                  287.837 |                   277.031 |                  266.225 |
        | 2025-12-22 |                -0.0419 |                      -1 |         2.42657e+09 |                      4.0879e+09  |          2.21091e+07 | nan      |                      0 |               0.079  |           -12.8461 |                 2708.57 |                 794.143 |            36.4312 |      -81.449  |               7.1429 |                85.7143 |                 -104.734  |                -0.2367 |                            -0.2614 |      -7.98568e+06 |               45.7605 |                        0.2188 |                       2.261  |                     23.4205 |                     -64.5465 |                    272.807 |              276.218 |                   265.155 |                   250.709 |         18.551  |         24.581  |      0.5981 |             2.0652 |                   44.0572 |            -0.5608 |              -2.0918 |                 0.8075 |                 1.2147 |                -0.8985 |                -4.2594 |          -5.3182 |            40.4149 |           20.5529 |                       41.1123 |                       275.762 |                            274.11  |                                    273.33  | 0.0946 |                     275.788 |                         273.972 |                     270.604 |                                  273.358 |                                275.272 |         281.048 |       271.983 |                    274.799 |                 270.111 |                       277.031 |                       274.481 |                282.966 |                 275.762 |                268.558 |       3.3609 |               4.8642 |      268.669 |                            1 |                 283.622 |                  274.11  |                 264.598 |                  287.837 |                   277.031 |                  266.225 |
        | 2025-12-23 |                 0.0106 |                       1 |         2.45621e+09 |                      4.11472e+09 |          2.96729e+07 | nan      |                      0 |               0.0651 |           -14.4872 |                 2722.46 |                 794.143 |            37.2984 |      -69.5722 |               0      |                78.5714 |                  -97.9336 |                -0.205  |                            -0.2501 |      -1.4419e+06  |               49.1968 |                        0.1353 |                       2.3529 |                     23.1942 |                     -51.4171 |                    272.807 |              276.218 |                   265.285 |                   250.709 |         30.4278 |         26.6633 |      0.3697 |             1.7261 |                   47.0291 |             0.517  |              -2.6643 |                 0.7732 |                 1.2661 |                -1.9298 |                -4.8618 |          -4.1492 |            46.4518 |           15.2164 |                       36.6324 |                       274.922 |                            273.778 |                                    272.814 | 0.0747 |                     274.973 |                         273.42  |                     270.19  |                                  273.115 |                                274.464 |         279.566 |       271.049 |                    272.322 |                 268.961 |                       275.091 |                       272.999 |                281.038 |                 274.922 |                268.807 |       2.932  |               4.7262 |      268.669 |                            1 |                 282.925 |                  273.778 |                 264.631 |                  287.837 |                   277.031 |                  266.225 |
        | 2025-12-24 |                 0.0554 |                       1 |         2.47412e+09 |                      4.11466e+09 |          3.0052e+07  | nan      |                      0 |               0.0846 |            -6.6554 |                 2736.96 |                 794.143 |            41.6581 |      -51.6561 |               0      |                71.4286 |                  -34.6713 |                -0.1841 |                            -0.2228 |       2.20352e+06 |               54.4449 |                        0.1105 |                       3.093  |                     22.0178 |                     -32.9194 |                    272.807 |              276.218 |                   265.264 |                   250.709 |         48.3439 |         32.4409 |      0.3019 |             1.4413 |                   50.012  |             0.4551 |              -2.8209 |                 0.8892 |                 1.1613 |                 1.0812 |                -2.14   |          -2.4546 |            53.1326 |           10.3327 |                       31.6831 |                       274.432 |                            273.683 |                                    272.766 | 0.0578 |                     274.257 |                         273.173 |                     270.366 |                                  273.112 |                                274.006 |         278.232 |       270.736 |                    272.645 |                 269.713 |                       273.301 |                       271.631 |                279.93  |                 274.432 |                268.933 |       3.2212 |               4.6187 |      268.669 |                            1 |                 282.56  |                  273.683 |                 264.807 |                  283.957 |                   275.091 |                  266.225 |
        | 2025-12-26 |                -0.0066 |                      -1 |         2.4526e+09  |                      4.1024e+09  |          2.35738e+07 | nan      |                      0 |               0.0857 |            -5.4478 |                 2736.96 |                 792.954 |            46.9441 |      -51.1365 |               0      |                64.2857 |                  -25.6937 |                -0.1697 |                            -0.1974 |  736348           |               51.216  |                        0.0779 |                      -0.6824 |                     20.9255 |                     -27.7035 |                    272.807 |              276.308 |                   265.162 |                   250.709 |         48.8635 |         42.5451 |      0.2128 |             1.1956 |                   49.1688 |            -0.3028 |              -2.1675 |                 0.9091 |                 1.1011 |                 1.1561 |                -1.3471 |          -1.9298 |            55.4899 |            5.6441 |                       26.3128 |                       274.048 |                            273.547 |                                    272.633 | 0.0432 |                     273.637 |                         272.936 |                     270.87  |                                  273.087 |                                273.626 |         277.031 |       273.07  |                    274.679 |                 271.458 |                       272.807 |                       271.254 |                279.231 |                 274.048 |                268.866 |       2.5032 |               4.4676 |      268.669 |                            1 |                 282.036 |                  273.547 |                 265.057 |                  280.377 |                   273.301 |                  266.225 |
        | 2025-12-29 |                 0.0405 |                       1 |         2.47632e+09 |                      4.11196e+09 |          2.18213e+07 |   0.4593 |                      0 |               0.0756 |            -4.8028 |                 2736.96 |                 793.999 |            48.1203 |      -48.4087 |              35.7143 |                57.1429 |                  -30.5655 |                -0.1563 |                            -0.1782 |       1.77365e+06 |               57.3149 |                        0.0619 |                       0.1104 |                     20.0591 |                     -21.8634 |                    270.818 |              276.393 |                   265.269 |                   250.709 |         51.5913 |         49.5996 |      0.1691 |             0.9903 |                   49.9667 |             0.5323 |              -2.0669 |                 0.9316 |                 1.0955 |                 0.2132 |                -1.7914 |          -1.4862 |            54.4742 |            1.1838 |                       21.0434 |                       273.754 |                            273.476 |                                    272.623 | 0.031  |                     273.126 |                         272.798 |                     271.532 |                                  273.085 |                                273.368 |         275.951 |       273.133 |                    274.147 |                 271.644 |                       272.807 |                       271.254 |                278.663 |                 273.754 |                268.846 |       2.0045 |               4.2917 |      268.669 |                            1 |                 281.517 |                  273.476 |                 265.435 |                  279.389 |                   272.807 |                  266.225 |
        | 2025-12-30 |                -0.0195 |                      -1 |         2.45418e+09 |                      4.1095e+09  |          1.83379e+07 |   1.5877 |                      0 |               0.0249 |            -5.2916 |                 2730.16 |                 793.999 |            43.4593 |      -53.5605 |              28.5714 |                50      |                  -35.5276 |                -0.1288 |                            -0.1608 | -464845           |               54.1042 |                        0.0289 |                      -2.36   |                     19.2759 |                     -21.7392 |                    270.818 |              276.393 |                   265.269 |                   250.709 |         46.4395 |         48.9648 |      0.0789 |             0.808  |                   48.4209 |             0.15   |              -1.9463 |                 0.9145 |                 1.1096 |                 0.0858 |                -1.7093 |          -1.4792 |            53.5704 |           -1.8152 |                       16.116  |                       273.462 |                            273.324 |                                    272.454 | 0.0204 |                     272.738 |                         272.61  |                     272.005 |                                  273.056 |                                273.119 |         274.978 |       272.748 |                    273.885 |                 271.88  |                       272.807 |                       271.254 |                278.183 |                 273.462 |                268.742 |       1.7951 |               4.1134 |      268.669 |                            1 |                 280.921 |                  273.324 |                 265.728 |                  279.389 |                   272.807 |                  266.225 |
        | 2025-12-31 |                -0.0707 |                      -1 |         2.42688e+09 |                      4.08532e+09 |          7.55142e+06 | nan      |                      0 |              -0.0357 |            -5.2547 |                 2730.16 |                 790.451 |            38.2003 |      -62.803  |              21.4286 |                42.8571 |                  -56.9152 |                -0.0887 |                            -0.1381 |      -4.83062e+06 |               46.408  |                       -0.0329 |                      -1.518  |                     18.7173 |                     -37.4455 |                    270.454 |              277.031 |                   265.802 |                   251.058 |         37.197  |         45.0759 |     -0.0897 |             0.6284 |                   45.6902 |            -0.6218 |              -1.6061 |                 0.9113 |                 1.1148 |                -0.0091 |                -1.9339 |          -2.4822 |            52.5944 |           -4.5221 |                       11.571  |                       272.969 |                            273.031 |                                    272.022 | 0.0101 |                     272.407 |                         272.298 |                     272.09  |                                  272.894 |                                272.801 |         274.103 |       272.405 |                    273.269 |                 271.474 |                       272.807 |                       271.254 |                277.084 |                 272.969 |                268.854 |       1.9248 |               3.957  |      268.669 |                            1 |                 280.252 |                  273.031 |                 265.809 |                  279.389 |                   272.807 |                  266.225 |
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

        self._all_indicators = pd.concat(
            [
                self.collect_breadth_indicators(
                    period=period, close_column=close_column
                ),
                self.collect_momentum_indicators(
                    period=period, close_column=close_column, window=window
                ),
                self.collect_overlap_indicators(
                    period=period, close_column=close_column, window=window
                ),
                self.collect_volatility_indicators(
                    period=period, close_column=close_column, window=window
                ),
            ],
            axis=1,
        )

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

    @handle_errors
    def collect_breadth_indicators(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects various breadth indicators based on the provided data.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
        | 2025-12-17 |                -0.0981 |                      -1 |         2.26688e+09 |                      3.99772e+09 |         -1.74845e+07 | nan      |                      0 |              -0.0358 |            -4.7237 |                 2735.56 |                 788.833 |
        | 2025-12-18 |                -0.0378 |                       1 |         2.31851e+09 |                      4.02709e+09 |         -1.16785e+07 | nan      |                      0 |               0.0024 |            -7.4205 |                 2735.56 |                 789.848 |
        | 2025-12-19 |                 0.0139 |                       1 |         2.46314e+09 |                      4.11449e+09 |          1.95658e+07 | nan      |                      0 |               0.1361 |            -9.5047 |                 2735.56 |                 794.143 |
        | 2025-12-22 |                -0.0419 |                      -1 |         2.42657e+09 |                      4.0879e+09  |          2.21091e+07 | nan      |                      0 |               0.079  |           -12.8461 |                 2708.57 |                 794.143 |
        | 2025-12-23 |                 0.0106 |                       1 |         2.45621e+09 |                      4.11472e+09 |          2.96729e+07 | nan      |                      0 |               0.0651 |           -14.4872 |                 2722.46 |                 794.143 |
        | 2025-12-24 |                 0.0554 |                       1 |         2.47412e+09 |                      4.11466e+09 |          3.0052e+07  | nan      |                      0 |               0.0846 |            -6.6554 |                 2736.96 |                 794.143 |
        | 2025-12-26 |                -0.0066 |                      -1 |         2.4526e+09  |                      4.1024e+09  |          2.35738e+07 | nan      |                      0 |               0.0857 |            -5.4478 |                 2736.96 |                 792.954 |
        | 2025-12-29 |                 0.0405 |                       1 |         2.47632e+09 |                      4.11196e+09 |          2.18213e+07 |   0.4593 |                      0 |               0.0756 |            -4.8028 |                 2736.96 |                 793.999 |
        | 2025-12-30 |                -0.0195 |                      -1 |         2.45418e+09 |                      4.1095e+09  |          1.83379e+07 |   1.5877 |                      0 |               0.0249 |            -5.2916 |                 2730.16 |                 793.999 |
        | 2025-12-31 |                -0.0707 |                      -1 |         2.42688e+09 |                      4.08532e+09 |          7.55142e+06 | nan      |                      0 |              -0.0357 |            -5.2547 |                 2730.16 |                 790.451 |
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
                for column, aggregation in (
                    ("Open", "first"),
                    ("High", "max"),
                    ("Low", "min"),
                ):
                    if column in adjusted:
                        # A period the daily data does not cover keeps the end-of-period factor.
                        adjusted[column] = (
                            daily_adjusted[column]
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the McClellan Oscillator for a given price series.

        The McClellan Oscillator is the difference between a short-term and a long-term
        exponential moving average of the daily advance/decline signal. Traditionally it is
        calculated from the number of advancing minus declining stocks across a market; here
        it is applied to each ticker's own signal (+1 on a day it closed higher, -1 on a day it
        closed lower, 0 otherwise), so it measures how persistently that ticker has been rising
        or falling.

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_mcclellan_oscillator()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | -0.0981 | -0.0175 |     -0.1682 |
        | 2025-12-18 | -0.0378 |  0.0326 |     -0.1047 |
        | 2025-12-19 |  0.0139 |  0.0753 |     -0.0499 |
        | 2025-12-22 | -0.0419 |  0.0115 |     -0.0027 |
        | 2025-12-23 |  0.0106 |  0.0568 |      0.0376 |
        | 2025-12-24 |  0.0554 |  0.0953 |      0.0718 |
        | 2025-12-26 | -0.0066 |  0.0277 |      0.0008 |
        | 2025-12-29 |  0.0405 | -0.0302 |     -0.06   |
        | 2025-12-30 | -0.0195 |  0.0204 |     -0.1116 |
        | 2025-12-31 | -0.0707 | -0.0364 |     -0.1552 |
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

        mcclellan_oscillator = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            mcclellan_oscillator[ticker] = breadth_model.get_mcclellan_oscillator(
                historical_data[close_column][ticker], short_ema_window, long_ema_window
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the daily advance/decline signal for a given price series.

        Traditionally advancers versus decliners is a market-wide count of rising against
        falling stocks; applied to a single ticker it becomes a directional signal: +1 on a
        day the ticker closed higher than the day before, -1 on a day it closed lower and 0
        on an unchanged day. It feeds the McClellan Oscillator; get_trin and
        get_new_highs_new_lows measure breadth across all tickers.

        The formula is a follows:

        - Signal = +1 if Close(t) > Close(t-1), -1 if Close(t) < Close(t-1), else 0

        Also known as: advance decline ratio, market breadth.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_advancers_decliners()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-17 |     -1 |     -1 |          -1 |
        | 2025-12-18 |      1 |      1 |           1 |
        | 2025-12-19 |      1 |      1 |           1 |
        | 2025-12-22 |     -1 |     -1 |           1 |
        | 2025-12-23 |      1 |      1 |           1 |
        | 2025-12-24 |      1 |      1 |           1 |
        | 2025-12-26 |     -1 |     -1 |          -1 |
        | 2025-12-29 |      1 |     -1 |          -1 |
        | 2025-12-30 |     -1 |      1 |          -1 |
        | 2025-12-31 |     -1 |     -1 |          -1 |
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

        advancers_decliners = breadth_model.get_advancers_decliners(
            historical_data[close_column],
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the On-Balance Volume (OBV) for a given price series.

        The On-Balance Volume (OBV) is a technical indicator that uses volume flow to predict changes in stock price.
        It accumulates the volume on up days and subtracts the volume on down days. The resulting OBV line provides
        insights into the buying and selling pressure behind price movements.

        The formula is a follows:

        - OBV = Previous OBV + Volume if Close > Previous Close, Previous OBV — Volume if Close < Previous Close,
          and Previous OBV otherwise.

        The level of OBV has no meaning by itself, since it depends on the first date of the data; only its
        direction and its divergences from price matter. A rising OBV while price moves sideways is read as
        accumulation ahead of a breakout, a falling OBV as distribution. A single day of very high volume, such
        as an index rebalance or an earnings day, can move the line for a long time.

        Also known as: OBV, volume momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the OBV.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: On-Balance Volume values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates On-Balance Volume
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the OBV using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_on_balance_volume()
        ```

        Which returns:

        | Date       |        AAPL |        MSFT |   Benchmark |
        |:-----------|------------:|------------:|------------:|
        | 2025-12-17 | 2.26688e+09 | 8.29319e+08 | 1.14122e+09 |
        | 2025-12-18 | 2.31851e+09 | 8.57893e+08 | 1.24987e+09 |
        | 2025-12-19 | 2.46314e+09 | 9.28729e+08 | 1.35347e+09 |
        | 2025-12-22 | 2.42657e+09 | 9.11766e+08 | 1.42302e+09 |
        | 2025-12-23 | 2.45621e+09 | 9.26449e+08 | 1.48786e+09 |
        | 2025-12-24 | 2.47412e+09 | 9.32305e+08 | 1.52731e+09 |
        | 2025-12-26 | 2.4526e+09  | 9.23463e+08 | 1.48569e+09 |
        | 2025-12-29 | 2.47632e+09 | 9.1257e+08  | 1.42314e+09 |
        | 2025-12-30 | 2.45418e+09 | 9.26514e+08 | 1.37597e+09 |
        | 2025-12-31 | 2.42688e+09 | 9.10912e+08 | 1.30183e+09 |
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

        on_balance_volume = breadth_model.get_on_balance_volume(
            historical_data[close_column],
            historical_data["Volume"],
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
    def get_accumulation_distribution_line(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Accumulation/Distribution Line for a given price series.

        The Accumulation/Distribution Line is a technical indicator that evaluates the flow of money
        into or out of an asset. It takes into account both price and volume information to identify
        whether an asset is being accumulated (bought) or distributed (sold) by investors.

        The formula is a follows:

        - Money Flow Multiplier = ((Close — Low) — (High — Close)) / (High — Low)
        - Money Flow Volume = Money Flow Multiplier * Volume
        - ADL = Previous ADL + Money Flow Volume

        Also known as: ADL, Chaikin ADL, volume-price trend.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Accumulation/Distribution Line.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Accumulation/Distribution Line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Accumulation/Distribution Line for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Accumulation/Distribution Line
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_accumulation_distribution_line()
        ```

        Which returns:

        | Date       |        AAPL |        MSFT |   Benchmark |
        |:-----------|------------:|------------:|------------:|
        | 2025-12-17 | 3.99772e+09 | 1.60643e+09 | 7.84069e+09 |
        | 2025-12-18 | 4.02709e+09 | 1.60757e+09 | 7.79046e+09 |
        | 2025-12-19 | 4.11449e+09 | 1.6274e+09  | 7.87163e+09 |
        | 2025-12-22 | 4.0879e+09  | 1.62296e+09 | 7.92573e+09 |
        | 2025-12-23 | 4.11472e+09 | 1.62833e+09 | 7.98338e+09 |
        | 2025-12-24 | 4.11466e+09 | 1.6311e+09  | 8.01111e+09 |
        | 2025-12-26 | 4.1024e+09  | 1.63659e+09 | 8.00572e+09 |
        | 2025-12-29 | 4.11196e+09 | 1.64095e+09 | 8.01431e+09 |
        | 2025-12-30 | 4.1095e+09  | 1.64021e+09 | 7.98763e+09 |
        | 2025-12-31 | 4.08532e+09 | 1.62668e+09 | 7.919e+09   |
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

        accumulation_distribution_line = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            accumulation_distribution_line[ticker] = (
                breadth_model.get_accumulation_distribution_line(
                    historical_data["High"][ticker],
                    historical_data["Low"][ticker],
                    historical_data[close_column][ticker],
                    historical_data["Volume"][ticker],
                ).loc[self._start_date : self._end_date]
            )

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chaikin Oscillator for a given price series.

        The Chaikin Oscillator is a momentum-based indicator that combines price and volume
        to help identify potential trends and reversals in the market. It is calculated as the
        difference between the 3-day and 10-day Accumulation/Distribution Line.

        The formula is a follows:

        - Chaikin Oscillator = EMA(short-window ADL) — EMA(long-window ADL)

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Chaikin Oscillator.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Chaikin Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Chaikin Oscillator for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Chaikin Oscillator
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chaikin_oscillator()
        ```

        Which returns:

        | Date       |         AAPL |        MSFT |    Benchmark |
        |:-----------|-------------:|------------:|-------------:|
        | 2025-12-17 | -1.74845e+07 | 7.01681e+06 | -2.28053e+07 |
        | 2025-12-18 | -1.16785e+07 | 4.50813e+06 | -5.08337e+07 |
        | 2025-12-19 |  1.95658e+07 | 9.37952e+06 | -3.185e+07   |
        | 2025-12-22 |  2.21091e+07 | 9.10779e+06 | -3.97501e+06 |
        | 2025-12-23 |  2.96729e+07 | 9.8772e+06  |  2.61336e+07 |
        | 2025-12-24 |  3.0052e+07  | 1.01762e+07 |  4.48979e+07 |
        | 2025-12-26 |  2.35738e+07 | 1.11187e+07 |  4.67752e+07 |
        | 2025-12-29 |  2.18213e+07 | 1.18816e+07 |  4.60255e+07 |
        | 2025-12-30 |  1.83379e+07 | 1.08801e+07 |  3.30466e+07 |
        | 2025-12-31 |  7.55142e+06 | 5.17349e+06 |  2.89497e+06 |
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

        chaikin_oscillator = breadth_model.get_chaikin_oscillator(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            historical_data["Volume"],
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chaikin Money Flow (CMF) for a given price series.

        The Chaikin Money Flow sums the same Money Flow Volume used by the Accumulation/
        Distribution Line over a rolling window and normalizes it by the window's total
        volume, turning the running (unbounded) Accumulation/Distribution Line into a
        bounded oscillator. Sustained readings above zero indicate buying pressure
        (accumulation) is dominating over the window, while sustained readings below zero
        indicate selling pressure (distribution).

        The formula is a follows:

        - Money Flow Multiplier = ((Close — Low) — (High — Close)) / (High — Low)
        - Money Flow Volume = Money Flow Multiplier * Volume
        - CMF = Sum(Money Flow Volume, window) / Sum(Volume, window)

        Also known as: CMF, Chaikin Money Flow.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods to sum the Money Flow Volume and
                volume over. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Chaikin Money Flow.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Chaikin Money Flow values, bounded between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Chaikin Money Flow for each asset in the Toolkit instance.
        - There is no formal journal citation for the Chaikin Money Flow; the standard
          textbook treatment is Murphy, J.J. (1999). "Technical Analysis of the Financial
          Markets." New York Institute of Finance.
        - If `growth` is set to True, the method calculates the growth of the Chaikin Money
          Flow using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_chaikin_money_flow()
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        chaikin_money_flow = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            chaikin_money_flow[ticker] = breadth_model.get_chaikin_money_flow(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                historical_data["Volume"][ticker],
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

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Ease of Movement.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Ease of Movement values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ease of Movement for each asset in the Toolkit instance.
        - Reference: Arms, R.W. (1989). "The Arms Index (TRIN): An Introduction to the
          Volume Analysis of Stock and Bond Markets." Business One Irwin.
        - If `growth` is set to True, the method calculates the growth of the Ease of
          Movement using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_ease_of_movement()
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        ease_of_movement = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            ease_of_movement[ticker] = breadth_model.get_ease_of_movement(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data["Volume"][ticker],
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

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Negative Volume Index.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Negative Volume Index values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Negative Volume Index for each asset in the Toolkit instance.
        - Reference: Fosback, N.G. (1976). "Stock Market Logic: A Sophisticated Approach to
          Profits on Wall Street." The Institute for Econometric Research.
        - If `growth` is set to True, the method calculates the growth of the Negative Volume
          Index using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_negative_volume_index()
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        negative_volume_index = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            negative_volume_index[ticker] = breadth_model.get_negative_volume_index(
                historical_data[close_column][ticker],
                historical_data["Volume"][ticker],
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

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Positive Volume Index.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Positive Volume Index values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Positive Volume Index for each asset in the Toolkit instance.
        - Reference: Fosback, N.G. (1976). "Stock Market Logic: A Sophisticated Approach to
          Profits on Wall Street." The Institute for Econometric Research.
        - If `growth` is set to True, the method calculates the growth of the Positive Volume
          Index using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_positive_volume_index()
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        positive_volume_index = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            positive_volume_index[ticker] = breadth_model.get_positive_volume_index(
                historical_data[close_column][ticker],
                historical_data["Volume"][ticker],
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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        | Date       |   Money Flow Index |   Williams %R |   Aroon Indicator Up |   Aroon Indicator Down |   Commodity Channel Index |   Relative Vigor Index |   Relative Vigor Index Signal Line |       Force Index |   Ultimate Oscillator |   Percentage Price Oscillator |   Detrended Price Oscillator |   Average Directional Index |   Chande Momentum Oscillator |   Ichimoku Conversion Line |   Ichimoku Base Line |   Ichimoku Leading Span A |   Ichimoku Leading Span B |   Stochastic %K |   Stochastic %D |   MACD Line |   MACD Signal Line |   Relative Strength Index |   Balance of Power |   Awesome Oscillator |   Vortex Indicator VI+ |   Vortex Indicator VI- |   Elder Ray Bull Power |   Elder Ray Bear Power |   Rate of Change |   Choppiness Index |   Know Sure Thing |   Know Sure Thing Signal Line |
        |:-----------|-------------------:|--------------:|---------------------:|-----------------------:|--------------------------:|-----------------------:|-----------------------------------:|------------------:|----------------------:|------------------------------:|-----------------------------:|----------------------------:|-----------------------------:|---------------------------:|---------------------:|--------------------------:|--------------------------:|----------------:|----------------:|------------:|-------------------:|--------------------------:|-------------------:|---------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------:|-------------------:|------------------:|------------------------------:|
        | 2025-12-17 |            39.7393 |      -98.8221 |              28.5714 |               100      |                 -139.905  |                -0.2262 |                            -0.0758 |      -3.27663e+07 |               44.5333 |                        0.6309 |                       0.0306 |                     24.6712 |                     -20.6214 |                    275.64  |              276.218 |                   265.243 |                   250.709 |          1.1779 |          8.6606 |      1.7276 |             3.162  |                   45.1866 |            -0.7013 |               1.2663 |                 0.9211 |                 1.0533 |                -0.1329 |                -4.6407 |          -2.0573 |            49.8428 |           36.3832 |                       51.073  |
        | 2025-12-18 |            34.4002 |      -75.8191 |              21.4286 |               100      |                 -162.429  |                -0.2832 |                            -0.1659 |      -2.59949e+07 |               50.981  |                        0.4656 |                      -0.3825 |                     24.4323 |                     -24.9067 |                    272.807 |              276.218 |                   265.243 |                   250.709 |         24.1809 |         14.0382 |      1.2741 |             2.7844 |                   45.9182 |            -0.2126 |               0.01   |                 0.865  |                 1.0802 |                -2.0715 |                -8.7334 |          -2.3884 |            42.7376 |           31.0981 |                       48.3448 |
        | 2025-12-19 |            43.3299 |      -68.9888 |              14.2857 |                92.8571 |                 -103.239  |                -0.2694 |                            -0.2349 |       5.9366e+06  |               51.5415 |                        0.3736 |                      -0.4188 |                     23.9077 |                     -39.3408 |                    272.807 |              276.218 |                   265.243 |                   250.709 |         31.0112 |         18.79   |      1.022  |             2.4319 |                   49.0175 |             0.3234 |              -1.1196 |                 0.903  |                 1.137  |                -0.8138 |                -5.5011 |          -3.3309 |            41.2483 |           26.4483 |                       45.0423 |
        | 2025-12-22 |            36.4312 |      -81.449  |               7.1429 |                85.7143 |                 -104.734  |                -0.2367 |                            -0.2614 |      -7.98568e+06 |               45.7605 |                        0.2188 |                       2.261  |                     23.4205 |                     -64.5465 |                    272.807 |              276.218 |                   265.155 |                   250.709 |         18.551  |         24.581  |      0.5981 |             2.0652 |                   44.0572 |            -0.5608 |              -2.0918 |                 0.8075 |                 1.2147 |                -0.8985 |                -4.2594 |          -5.3182 |            40.4149 |           20.5529 |                       41.1123 |
        | 2025-12-23 |            37.2984 |      -69.5722 |               0      |                78.5714 |                  -97.9336 |                -0.205  |                            -0.2501 |      -1.4419e+06  |               49.1968 |                        0.1353 |                       2.3529 |                     23.1942 |                     -51.4171 |                    272.807 |              276.218 |                   265.285 |                   250.709 |         30.4278 |         26.6633 |      0.3697 |             1.7261 |                   47.0291 |             0.517  |              -2.6643 |                 0.7732 |                 1.2661 |                -1.9298 |                -4.8618 |          -4.1492 |            46.4518 |           15.2164 |                       36.6324 |
        | 2025-12-24 |            41.6581 |      -51.6561 |               0      |                71.4286 |                  -34.6713 |                -0.1841 |                            -0.2228 |       2.20352e+06 |               54.4449 |                        0.1105 |                       3.093  |                     22.0178 |                     -32.9194 |                    272.807 |              276.218 |                   265.264 |                   250.709 |         48.3439 |         32.4409 |      0.3019 |             1.4413 |                   50.012  |             0.4551 |              -2.8209 |                 0.8892 |                 1.1613 |                 1.0812 |                -2.14   |          -2.4546 |            53.1326 |           10.3327 |                       31.6831 |
        | 2025-12-26 |            46.9441 |      -51.1365 |               0      |                64.2857 |                  -25.6937 |                -0.1697 |                            -0.1974 |  736348           |               51.216  |                        0.0779 |                      -0.6824 |                     20.9255 |                     -27.7035 |                    272.807 |              276.308 |                   265.162 |                   250.709 |         48.8635 |         42.5451 |      0.2128 |             1.1956 |                   49.1688 |            -0.3028 |              -2.1675 |                 0.9091 |                 1.1011 |                 1.1561 |                -1.3471 |          -1.9298 |            55.4899 |            5.6441 |                       26.3128 |
        | 2025-12-29 |            48.1203 |      -48.4087 |              35.7143 |                57.1429 |                  -30.5655 |                -0.1563 |                            -0.1782 |       1.77365e+06 |               57.3149 |                        0.0619 |                       0.1104 |                     20.0591 |                     -21.8634 |                    270.818 |              276.393 |                   265.269 |                   250.709 |         51.5913 |         49.5996 |      0.1691 |             0.9903 |                   49.9667 |             0.5323 |              -2.0669 |                 0.9316 |                 1.0955 |                 0.2132 |                -1.7914 |          -1.4862 |            54.4742 |            1.1838 |                       21.0434 |
        | 2025-12-30 |            43.4593 |      -53.5605 |              28.5714 |                50      |                  -35.5276 |                -0.1288 |                            -0.1608 | -464845           |               54.1042 |                        0.0289 |                      -2.36   |                     19.2759 |                     -21.7392 |                    270.818 |              276.393 |                   265.269 |                   250.709 |         46.4395 |         48.9648 |      0.0789 |             0.808  |                   48.4209 |             0.15   |              -1.9463 |                 0.9145 |                 1.1096 |                 0.0858 |                -1.7093 |          -1.4792 |            53.5704 |           -1.8152 |                       16.116  |
        | 2025-12-31 |            38.2003 |      -62.803  |              21.4286 |                42.8571 |                  -56.9152 |                -0.0887 |                            -0.1381 |      -4.83062e+06 |               46.408  |                       -0.0329 |                      -1.518  |                     18.7173 |                     -37.4455 |                    270.454 |              277.031 |                   265.802 |                   251.058 |         37.197  |         45.0759 |     -0.0897 |             0.6284 |                   45.6902 |            -0.6218 |              -1.6061 |                 0.9113 |                 1.1148 |                -0.0091 |                -1.9339 |          -2.4822 |            52.5944 |           -4.5221 |                       11.571  |
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
        momentum_indicators["Percentage Price Oscillator"] = (
            self.get_percentage_price_oscillator(
                period=period, close_column=close_column
            )
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
        momentum_indicators["Chande Momentum Oscillator"] = (
            self.get_chande_momentum_oscillator(
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

        momentum_indicators["Relative Strength Index"] = (
            self.get_relative_strength_index(
                period=period, close_column=close_column, window=window
            )
        )
        momentum_indicators["Balance of Power"] = self.get_balance_of_power(
            period=period, close_column=close_column
        )

        momentum_indicators["Awesome Oscillator"] = self.get_awesome_oscillator(
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

        momentum_indicators["Choppiness Index"] = self.get_choppiness_index(
            period=period, close_column=close_column, window=window
        )

        know_sure_thing = self.get_know_sure_thing(
            period=period, close_column=close_column
        )

        momentum_indicators["Know Sure Thing"] = know_sure_thing["KST"]
        momentum_indicators["Know Sure Thing Signal Line"] = know_sure_thing[
            "Signal Line"
        ]

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Money Flow Index (MFI) for a given price series.

        The Money Flow Index is a momentum indicator that measures the strength and
        direction of money flowing in and out of a security by considering both price
        and volume.

        The formula is a follows:

        - MFI = 100 — (100 / (1 + (positive_money_flow / negative_money_flow)))

        Also known as: MFI, volume-weighted RSI.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for calculating the MFI.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Money Flow Index (MFI) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the MFI values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_money_flow_index()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 39.7393 | 50.9835 |     44.2618 |
        | 2025-12-18 | 34.4002 | 53.0158 |     47.0982 |
        | 2025-12-19 | 43.3299 | 65.0307 |     54.0865 |
        | 2025-12-22 | 36.4312 | 64.775  |     54.3543 |
        | 2025-12-23 | 37.2984 | 72.3431 |     54.6624 |
        | 2025-12-24 | 41.6581 | 71.0458 |     53.8185 |
        | 2025-12-26 | 46.9441 | 67.1794 |     52.3144 |
        | 2025-12-29 | 48.1203 | 62.6155 |     51.9586 |
        | 2025-12-30 | 43.4593 | 62.5153 |     52.4466 |
        | 2025-12-31 | 38.2003 | 66.6327 |     45.3997 |
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

        money_flow_index = momentum_model.get_money_flow_index(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            historical_data["Volume"],
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Williams Percent R (Williams %R) for a given price series.

        The Williams %R is a momentum indicator that measures the level of the close price
        relative to the high-low range over a certain number of periods.

        The formula is a follows:

        - Williams %R = (Highest High — Close) / (Highest High — Lowest Low) * —100

        Also known as: Williams percent R, overbought oversold oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for calculating the Williams %R.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Williams %R values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Williams %R values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_williams_percent_r()
        ```

        Which returns:

        | Date       |     AAPL |     MSFT |   Benchmark |
        |:-----------|---------:|---------:|------------:|
        | 2025-12-17 | -98.8221 | -76.8347 |    -98.892  |
        | 2025-12-18 | -75.8191 | -42.0864 |    -70.8032 |
        | 2025-12-19 | -68.9888 | -33.5098 |    -36.8356 |
        | 2025-12-22 | -81.449  | -34.4535 |    -13.2761 |
        | 2025-12-23 | -69.5722 | -25.4434 |     -1.2647 |
        | 2025-12-24 | -51.6561 | -19.9814 |     -2.0826 |
        | 2025-12-26 | -51.1365 | -21.4284 |     -6.0167 |
        | 2025-12-29 | -48.4087 | -23.6347 |    -16.9808 |
        | 2025-12-30 | -53.5605 | -11.7021 |    -20.7246 |
        | 2025-12-31 | -62.803  | -32.234  |    -43.4096 |
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

        williams_percent_r = momentum_model.get_williams_percent_r(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Aroon Indicator for a given price series.

        The Aroon Indicator is an oscillator that measures the strength of a trend and the
        likelihood of its continuation or reversal.

        The formula is a follows:

        - Aroon Up = ((Number of periods) — (Number of periods since highest high)) / (Number of periods) * 100

        Also known as: Aroon Up, Aroon Down, trend strength.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for calculating the Aroon Indicator.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame:
            Aroon Indicator values for the upward and downward trends.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Aroon Indicator values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_aroon_indicator().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Aroon Down |   Aroon Up |
        |:-----------|-------------:|-----------:|
        | 2025-12-17 |     100      |    28.5714 |
        | 2025-12-18 |     100      |    21.4286 |
        | 2025-12-19 |      92.8571 |    14.2857 |
        | 2025-12-22 |      85.7143 |     7.1429 |
        | 2025-12-23 |      78.5714 |     0      |
        | 2025-12-24 |      71.4286 |     0      |
        | 2025-12-26 |      64.2857 |     0      |
        | 2025-12-29 |      57.1429 |    35.7143 |
        | 2025-12-30 |      50      |    28.5714 |
        | 2025-12-31 |      42.8571 |    21.4286 |
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

        aroon_indicator_dict = {}

        for ticker in historical_data[close_column].columns:
            aroon_indicator_dict[ticker] = momentum_model.get_aroon_indicator(
                historical_data["High"][ticker], historical_data["Low"][ticker], window
            )

        aroon_indicator = (
            pd.concat(aroon_indicator_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Commodity Channel Index (CCI) for a given price series.

        The Commodity Channel Index is an oscillator that measures the current price level
        relative to an average price level over a specified period.

        The formula is a follows:

        - CCI = (Typical Price — SMA(Typical Price)) / (constant * Mean Deviation)

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Commodity Channel Index (CCI) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the CCI values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_commodity_channel_index()
        ```

        Which returns:

        | Date       |      AAPL |     MSFT |   Benchmark |
        |:-----------|----------:|---------:|------------:|
        | 2025-12-17 | -139.905  | -74.8721 |   -239.211  |
        | 2025-12-18 | -162.429  |  26.1797 |   -118.59   |
        | 2025-12-19 | -103.239  |  54.4683 |    -21.3412 |
        | 2025-12-22 | -104.734  |  65.9749 |     74.5493 |
        | 2025-12-23 |  -97.9336 |  71.0128 |    125.5    |
        | 2025-12-24 |  -34.6713 |  72.9192 |    155.834  |
        | 2025-12-26 |  -25.6937 |  62.5804 |    144.031  |
        | 2025-12-29 |  -30.5655 |  56.6706 |     82.3319 |
        | 2025-12-30 |  -35.5276 |  80.307  |     67.171  |
        | 2025-12-31 |  -56.9152 |  32.9915 |     13.312  |
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

        commodity_channel_index = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )

        for ticker in historical_data[close_column].columns:
            commodity_channel_index[ticker] = (
                momentum_model.get_commodity_channel_index(
                    historical_data["High"][ticker],
                    historical_data["Low"][ticker],
                    historical_data[close_column][ticker],
                    window,
                    constant,
                ).loc[self._start_date : self._end_date]
            )

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Vigor Index (RVI) for a given price series.

        The Relative Vigor Index measures the conviction behind a move by comparing where each
        period closes relative to its open with the period's full high-low range: in an uptrend
        prices tend to close above their open, in a downtrend below it.

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_relative_vigor_index().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Relative Vigor Index |   Signal Line |
        |:-----------|-----------------------:|--------------:|
        | 2025-12-17 |                -0.2262 |       -0.0758 |
        | 2025-12-18 |                -0.2832 |       -0.1659 |
        | 2025-12-19 |                -0.2694 |       -0.2349 |
        | 2025-12-22 |                -0.2367 |       -0.2614 |
        | 2025-12-23 |                -0.205  |       -0.2501 |
        | 2025-12-24 |                -0.1841 |       -0.2228 |
        | 2025-12-26 |                -0.1697 |       -0.1974 |
        | 2025-12-29 |                -0.1563 |       -0.1782 |
        | 2025-12-30 |                -0.1288 |       -0.1608 |
        | 2025-12-31 |                -0.0887 |       -0.1381 |
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

        relative_vigor_index_dict = {}

        for ticker in historical_data[close_column].columns:
            relative_vigor_index_dict[ticker] = momentum_model.get_relative_vigor_index(
                historical_data["Open"][ticker],
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
            ).loc[self._start_date : self._end_date]

        relative_vigor_index = (
            pd.concat(relative_vigor_index_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Force Index for a given price series.

        The Force Index is an indicator that measures the strength behind price movements.

        The formula is a follows:

        - Raw Force Index = (Close — Close(1)) * Volume
        - Force Index = EMA(Raw Force Index, window)

        Also known as: FI, Elder's Force Index.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods of the EMA that smooths the raw
                Force Index. Defaults to 13, Elder's setting for the trend (2 is his setting
                for timing entries).
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Force Index values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Force Index values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_force_index()
        ```

        Which returns:

        | Date       |              AAPL |         MSFT |    Benchmark |
        |:-----------|------------------:|-------------:|-------------:|
        | 2025-12-17 |      -3.59981e+07 | -3.95103e+07 | -1.83097e+08 |
        | 2025-12-18 |      -2.82814e+07 | -1.98405e+06 | -7.90865e+07 |
        | 2025-12-19 |       6.2576e+06  |  1.78077e+07 |  2.19835e+07 |
        | 2025-12-22 |      -8.70499e+06 |  1.28557e+07 |  6.06476e+07 |
        | 2025-12-23 |      -1.59104e+06 |  1.5042e+07  |  8.07513e+07 |
        | 2025-12-24 |       2.33607e+06 |  1.38658e+07 |  8.27469e+07 |
        | 2025-12-26 |  745167           |  1.14959e+07 |  7.05128e+07 |
        | 2025-12-29 |       1.8553e+06  |  8.91024e+06 |  3.86241e+07 |
        | 2025-12-30 | -554750           |  8.38976e+06 |  2.74909e+07 |
        | 2025-12-31 |      -5.21952e+06 | -1.35779e+06 | -2.99318e+07 |
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

        force_index = momentum_model.get_force_index(
            historical_data[close_column],
            historical_data["Volume"],
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Ultimate Oscillator for a given price series.

        The Ultimate Oscillator is a momentum oscillator that combines short-term, mid-term,
        and long-term price momentum into a single value.

        The formula is a follows:

        - Average(i) = Sum(Buying Pressure, window_i) / Sum(True Range, window_i)
        - Ultimate Oscillator = 100 * [(4 * Average_1) + (2 * Average_2) + Average_3] / 7

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Ultimate Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ultimate Oscillator values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_ultimate_oscillator()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 44.5333 | 47.4263 |     46.921  |
        | 2025-12-18 | 50.981  | 46.9751 |     47.9861 |
        | 2025-12-19 | 51.5415 | 52.5231 |     49.415  |
        | 2025-12-22 | 45.7605 | 48.6534 |     48.2421 |
        | 2025-12-23 | 49.1968 | 52.0442 |     53.3907 |
        | 2025-12-24 | 54.4449 | 55.7706 |     57.7082 |
        | 2025-12-26 | 51.216  | 53.7379 |     56.8785 |
        | 2025-12-29 | 57.3149 | 56.5047 |     64.5078 |
        | 2025-12-30 | 54.1042 | 55.7349 |     66.0175 |
        | 2025-12-31 | 46.408  | 51.1112 |     52.8914 |
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

        ultimate_oscillator = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            ultimate_oscillator[ticker] = momentum_model.get_ultimate_oscillator(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
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
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Percentage Price Oscillator (PPO) for a given price series.

        The Percentage Price Oscillator (PPO) is a momentum oscillator that measures the
        difference between two moving averages as a percentage of the longer moving average.

        The formula is a follows:

        - PPO = ((Short-term EMA — Long-term EMA) / Long-term EMA) * 100

        The PPO is the MACD line as a percentage of the long EMA, so unlike the MACD its level
        can be compared across stocks and over long periods. The usual signal line is the
        9-period EMA of the PPO; it is not returned here and can be calculated from the result,
        for example with `get_exponential_moving_average` of the `overlap_model`.

        Also known as: PPO, price oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            short_window (int, optional): The number of periods for the short-term moving average.
                Defaults to 12, as for MACD.
            long_window (int, optional): The number of periods for the long-term moving average.
                Defaults to 26, as for MACD.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Percentage Price Oscillator (PPO) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the PPO values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_percentage_price_oscillator()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 |  0.6309 | -1.3088 |      0.2828 |
        | 2025-12-18 |  0.4656 | -1.1665 |      0.2205 |
        | 2025-12-19 |  0.3736 | -1.0096 |      0.2412 |
        | 2025-12-22 |  0.2188 | -0.8916 |      0.3044 |
        | 2025-12-23 |  0.1353 | -0.7572 |      0.3871 |
        | 2025-12-24 |  0.1105 | -0.624  |      0.4756 |
        | 2025-12-26 |  0.0779 | -0.5176 |      0.5384 |
        | 2025-12-29 |  0.0619 | -0.4384 |      0.5526 |
        | 2025-12-30 |  0.0289 | -0.3651 |      0.5475 |
        | 2025-12-31 | -0.0329 | -0.3669 |      0.4779 |
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

        percentage_price_oscillator = momentum_model.get_percentage_price_oscillator(
            historical_data[close_column],
            short_window,
            long_window,
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
    def get_detrended_price_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Detrended Price Oscillator (DPO) for a given price series.

        The Detrended Price Oscillator (DPO) is an indicator that helps identify short-term cycles
        by removing longer-term trends from prices.

        The formula is a follows:

        - Displacement = floor(Number of Periods / 2) + 1
        - DPO = Close(t — Displacement) — SMA(Close, Number of Periods)(t)

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Detrended Price Oscillator (DPO) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the DPO values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_detrended_price_oscillator()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 |  0.0306 |  0.2008 |      3.1326 |
        | 2025-12-18 | -0.3825 |  8.5813 |      1.5837 |
        | 2025-12-19 | -0.4188 |  9.6332 |      0.8352 |
        | 2025-12-22 |  2.261  | -3.3814 |      4.9407 |
        | 2025-12-23 |  2.3529 |  0.8503 |      6.0923 |
        | 2025-12-24 |  3.093  | -4.5681 |     -1.8061 |
        | 2025-12-26 | -0.6824 | -8.5777 |     -3.2957 |
        | 2025-12-29 |  0.1104 | -6.7394 |     -5.5777 |
        | 2025-12-30 | -2.36   | -6.6855 |    -13.392  |
        | 2025-12-31 | -1.518  |  0.7659 |     -8.1192 |
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

        detrended_price_oscillator = momentum_model.get_detrended_price_oscillator(
            historical_data[close_column], window
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX) for a given price series.

        The Average Directional Index (ADX) is an indicator that measures the strength of a trend,
        whether it's an uptrend or a downtrend.

        The formula is a follows:

        - ADX = Wilder's Smoothed Moving Average of DX, where DX = 100 * |+DI — -DI| / (+DI + -DI)

        The ADX measures trend strength, not direction: a reading above about 25 is usually taken as a
        trending market and below 20 as a range, whether prices are rising or falling. Traders combine it
        with the +DI and -DI lines it is built from, which this method does not return, or with the price
        itself, to know the direction. With Wilder's smoothing it reacts slowly and needs a few times the
        window of history before its values settle.

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series: Average Directional Index (ADX) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the ADX values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_average_directional_index()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 24.6712 | 25.787  |     14.9613 |
        | 2025-12-18 | 24.4323 | 24.4769 |     14.3719 |
        | 2025-12-19 | 23.9077 | 23.2604 |     13.3525 |
        | 2025-12-22 | 23.4205 | 22.0112 |     13.1626 |
        | 2025-12-23 | 23.1942 | 20.8512 |     13.4436 |
        | 2025-12-24 | 22.0178 | 19.5733 |     14.0997 |
        | 2025-12-26 | 20.9255 | 18.3867 |     14.8316 |
        | 2025-12-29 | 20.0591 | 17.5595 |     14.7622 |
        | 2025-12-30 | 19.2759 | 16.5556 |     14.6978 |
        | 2025-12-31 | 18.7173 | 15.983  |     13.7115 |
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

        average_directional_index = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            average_directional_index[ticker] = (
                momentum_model.get_average_directional_index(
                    historical_data["High"][ticker],
                    historical_data["Low"][ticker],
                    historical_data[close_column][ticker],
                    window,
                ).loc[self._start_date : self._end_date]
            )

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
    def get_chande_momentum_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chande Momentum Oscillator (CMO) for a given price series.

        The Chande Momentum Oscillator is an indicator that measures the momentum of a price
        series and identifies overbought and oversold conditions.

        The formula is a follows:

        - CMO = 100 * ((Sum of Upward Change) — (Sum of Downward Change)) / ((Sum of Upward Change)
            + (Sum of Downward Change)), bounded between -100 and 100.

        Also known as: CMO, Chande momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods to consider for the CMO calculation.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Chande Momentum Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Chande Momentum Oscillator values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_chande_momentum_oscillator()
        ```

        Which returns:

        | Date       |     AAPL |     MSFT |   Benchmark |
        |:-----------|---------:|---------:|------------:|
        | 2025-12-17 | -20.6214 | -13.3126 |    -21.3403 |
        | 2025-12-18 | -24.9067 | -11.1824 |    -17.2311 |
        | 2025-12-19 | -39.3408 |  -1.1974 |      5.3996 |
        | 2025-12-22 | -64.5465 |  -7.6713 |     11.5321 |
        | 2025-12-23 | -51.4171 |  16.3207 |     13.0006 |
        | 2025-12-24 | -32.9194 |  13.3111 |     16.4316 |
        | 2025-12-26 | -27.7035 |   8.762  |     13.9793 |
        | 2025-12-29 | -21.8634 |  -8.7736 |     13.013  |
        | 2025-12-30 | -21.7392 | -10.304  |     12.4222 |
        | 2025-12-31 | -37.4455 |  14.6837 |     -7.4375 |
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

        chande_momentum_oscillator = momentum_model.get_chande_momentum_oscillator(
            historical_data[close_column], window
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Ichimoku Cloud indicator for a given price series.

        The Ichimoku Cloud, also known as the Ichimoku Kinko Hyo, is a versatile indicator that
        defines support and resistance, identifies trend direction, gauges momentum, and provides
        trading signals.

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame:
            Conversion Line, Base Line, Lead Span A, and Lead Span B values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ichimoku Cloud values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        - The leading spans are shifted forward by exactly `base_window` periods, as StockCharts
          defines the cloud. Platforms that count the current period as the first of the 26
          shift by `base_window` - 1 and show the same values one period earlier.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_ichimoku_cloud().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Base Line |   Conversion Line |   Leading Span A |   Leading Span B |
        |:-----------|------------:|------------------:|-----------------:|-----------------:|
        | 2025-12-17 |     276.218 |           275.64  |          265.243 |          250.709 |
        | 2025-12-18 |     276.218 |           272.807 |          265.243 |          250.709 |
        | 2025-12-19 |     276.218 |           272.807 |          265.243 |          250.709 |
        | 2025-12-22 |     276.218 |           272.807 |          265.155 |          250.709 |
        | 2025-12-23 |     276.218 |           272.807 |          265.285 |          250.709 |
        | 2025-12-24 |     276.218 |           272.807 |          265.264 |          250.709 |
        | 2025-12-26 |     276.308 |           272.807 |          265.162 |          250.709 |
        | 2025-12-29 |     276.393 |           270.818 |          265.269 |          250.709 |
        | 2025-12-30 |     276.393 |           270.818 |          265.269 |          250.709 |
        | 2025-12-31 |     277.031 |           270.454 |          265.802 |          251.058 |
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

        ichimoku_cloud_dict = {}

        for ticker in historical_data[close_column].columns:
            ichimoku_cloud_dict[ticker] = momentum_model.get_ichimoku_cloud(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                conversion_window,
                base_window,
                lead_span_b_window,
            ).loc[self._start_date : self._end_date]

        ichimoku_cloud = (
            pd.concat(ichimoku_cloud_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

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
        smooth_window: int = DEFAULT_STOCHASTIC_SMOOTH_WINDOW,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        smooth_widow: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Stochastic Oscillator indicator for a given price series.

        The Stochastic Oscillator is a momentum indicator that shows the location of the close
        relative to the high-low range over a set number of periods. It consists of the %K line
        (fast) and the %D line (slow).

        The formula is a follows:

        - %K = 100 * ((Close — Lowest Low) / (Highest High — Lowest Low))
        - %D = SMA(%K, smooth_window)

        Also known as: stochastic oscillator, percent K, percent D.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods to consider for the %K line calculation.
                Defaults to 14.
            smooth_window (int, optional): The number of periods used to smooth the %K line
                into the %D signal line. Defaults to 3.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the %K and %D values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            smooth_widow (int | None, optional): Deprecated misspelling of `smooth_window`,
                accepted so that existing callers keep working. Passing it emits a
                DeprecationWarning and forwards the value to `smooth_window`. Defaults to None.

        Returns:
            pd.Series or pd.DataFrame: Stochastic Oscillator (%K and %D) values.

        Raises:
            ValueError: If the specified `period` is not one of the valid options, or if both
                `smooth_window` and the deprecated `smooth_widow` are given conflicting values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Stochastic Oscillator values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the %K and %D values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_stochastic_oscillator().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Stochastic %D |   Stochastic %K |
        |:-----------|----------------:|----------------:|
        | 2025-12-17 |          8.6606 |          1.1779 |
        | 2025-12-18 |         14.0382 |         24.1809 |
        | 2025-12-19 |         18.79   |         31.0112 |
        | 2025-12-22 |         24.581  |         18.551  |
        | 2025-12-23 |         26.6633 |         30.4278 |
        | 2025-12-24 |         32.4409 |         48.3439 |
        | 2025-12-26 |         42.5451 |         48.8635 |
        | 2025-12-29 |         49.5996 |         51.5913 |
        | 2025-12-30 |         48.9648 |         46.4395 |
        | 2025-12-31 |         45.0759 |         37.197  |
        """
        if smooth_widow is not None:
            warnings.warn(
                "The 'smooth_widow' parameter is a misspelling and is deprecated, use "
                "'smooth_window' instead. It will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            if smooth_window not in (DEFAULT_STOCHASTIC_SMOOTH_WINDOW, smooth_widow):
                raise ValueError(
                    "Received conflicting values for 'smooth_window' and the deprecated "
                    "'smooth_widow'. Pass only 'smooth_window'."
                )
            smooth_window = smooth_widow

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

        stochastic_oscillator_dict = {}

        for ticker in historical_data[close_column].columns:
            stochastic_oscillator_dict[ticker] = (
                momentum_model.get_stochastic_oscillator(
                    historical_data["High"][ticker],
                    historical_data["Low"][ticker],
                    historical_data[close_column][ticker],
                    window,
                    smooth_window,
                ).loc[self._start_date : self._end_date]
            )

        stochastic_oscillator = (
            pd.concat(stochastic_oscillator_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

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
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Moving Average Convergence Divergence (MACD) indicator for a given price series.

        The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator
        that shows the relationship between two moving averages of a security's price. It consists
        of the MACD line and its signal line; their difference is the MACD histogram.

        The formula is a follows:

        - MACD Line = EMA(Close, short_window) — EMA(Close, long_window)
        - Signal Line = EMA(MACD Line, signal_window)

        The MACD line crossing its signal line is the usual trigger, and the MACD crossing zero means
        the short average has crossed the long one. Because the MACD is a price difference, its level
        scales with the share price and cannot be compared across stocks; the Percentage Price Oscillator
        divides by the long average for that. Both lines lag price, so crossovers come late in fast
        reversals.

        Also known as: MACD, momentum indicator.

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the MACD and signal values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame:
            MACD line and signal line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the MACD and signal line values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the MACD and signal values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_moving_average_convergence_divergence().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   MACD Line |   Signal Line |
        |:-----------|------------:|--------------:|
        | 2025-12-17 |      1.7276 |        3.162  |
        | 2025-12-18 |      1.2741 |        2.7844 |
        | 2025-12-19 |      1.022  |        2.4319 |
        | 2025-12-22 |      0.5981 |        2.0652 |
        | 2025-12-23 |      0.3697 |        1.7261 |
        | 2025-12-24 |      0.3019 |        1.4413 |
        | 2025-12-26 |      0.2128 |        1.1956 |
        | 2025-12-29 |      0.1691 |        0.9903 |
        | 2025-12-30 |      0.0789 |        0.808  |
        | 2025-12-31 |     -0.0897 |        0.6284 |
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

        macd_dict = {}

        for ticker in historical_data[close_column].columns:
            macd_dict[ticker] = (
                momentum_model.get_moving_average_convergence_divergence(
                    historical_data[close_column][ticker],
                    short_window,
                    long_window,
                    signal_window,
                ).loc[self._start_date : self._end_date]
            )

        macd = pd.concat(macd_dict, axis=1).swaplevel(1, 0, axis=1).sort_index(axis=1)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI) indicator for a given price series.

        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and
        change of price movements. It ranges from 0 to 100 and is used to identify overbought or
        oversold conditions in an asset's price.

        The formula is a follows:

        - RSI = 100 — (100 / (1 + RS))

        Readings above 70 are conventionally read as overbought and below 30 as oversold, but in a
        strong trend the RSI can stay above 70 or below 30 for weeks, so a crossing is not a signal on
        its own. Practitioners look more at divergences, a new price high with a lower RSI high, and at
        the 50 line as a trend filter. The value depends on the window: a shorter window swings more and
        reaches the extremes more often.

        Also known as: RSI, momentum oscillator, overbought, oversold.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for RSI calculation. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the RSI.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Relative Strength Index (RSI) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          RSI for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the RSI
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_relative_strength_index()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 45.1866 | 40.7933 |     43.3897 |
        | 2025-12-18 | 45.9182 | 47.2376 |     48.6798 |
        | 2025-12-19 | 49.0175 | 48.7212 |     54.2478 |
        | 2025-12-22 | 44.0572 | 47.9724 |     57.6776 |
        | 2025-12-23 | 47.0291 | 49.5829 |     60.058  |
        | 2025-12-24 | 50.012  | 50.5816 |     61.8449 |
        | 2025-12-26 | 49.1688 | 50.2973 |     61.7588 |
        | 2025-12-29 | 49.9667 | 49.7053 |     58.6688 |
        | 2025-12-30 | 48.4209 | 50.0994 |     57.6088 |
        | 2025-12-31 | 45.6902 | 46.1447 |     51.5335 |
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

        relative_strength_index = momentum_model.get_relative_strength_index(
            historical_data[close_column], window
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
    def get_balance_of_power(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Balance of Power (BOP) indicator for a given price series.

        The Balance of Power (BOP) indicator measures the strength of buyers versus sellers
        in the market. It relates the price change to the change in the asset's trading range.

        The formula is a follows:

        - BOP = (Close — Open) / (High — Low)

        Also known as: BOP, bull bear power.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the BOP.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Balance of Power (BOP) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          BOP for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the BOP
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_balance_of_power()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | -0.7013 | -0.158  |     -0.9188 |
        | 2025-12-18 | -0.2126 |  0.4944 |     -0.1935 |
        | 2025-12-19 |  0.3234 | -0.2687 |      0.8658 |
        | 2025-12-22 | -0.5608 | -0.1987 |      0.1866 |
        | 2025-12-23 |  0.517  |  0.6052 |      0.933  |
        | 2025-12-24 |  0.4551 |  0.5404 |      0.802  |
        | 2025-12-26 | -0.3028 |  0.463  |     -0.1381 |
        | 2025-12-29 |  0.5323 |  0.5372 |      0.099  |
        | 2025-12-30 |  0.15   |  0.3708 |     -0.2222 |
        | 2025-12-31 | -0.6218 | -0.8719 |     -0.9239 |
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

        balance_of_power = momentum_model.get_balance_of_power(
            historical_data["Open"],
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Awesome Oscillator (AO) for a given price series.

        The Awesome Oscillator measures market momentum by comparing a short-term and a
        long-term Simple Moving Average of the median price (the midpoint of each period's high
        and low, rather than the closing price). It was developed by Bill Williams as part of
        his broader "Trading Chaos" collection of momentum indicators.

        The formula is a follows:

        - Median Price = (High + Low) / 2
        - AO = SMA(Median Price, short_window) — SMA(Median Price, long_window)

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the AO.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
        - If `growth` is set to True, the method calculates the growth of the AO
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_awesome_oscillator()
        ```

        Which returns:

        | Date       |    AAPL |     MSFT |   Benchmark |
        |:-----------|--------:|---------:|------------:|
        | 2025-12-17 |  1.2663 | -14.3575 |      4.2885 |
        | 2025-12-18 |  0.01   | -12.5205 |      2.8747 |
        | 2025-12-19 | -1.1196 | -10.3187 |      2.2738 |
        | 2025-12-22 | -2.0918 |  -7.545  |      2.7043 |
        | 2025-12-23 | -2.6643 |  -4.4653 |      4.3753 |
        | 2025-12-24 | -2.8209 |  -1.9132 |      7.0459 |
        | 2025-12-26 | -2.1675 |  -0.8848 |      9.3855 |
        | 2025-12-29 | -2.0669 |  -0.4006 |     10.4583 |
        | 2025-12-30 | -1.9463 |   0.3908 |     11.0502 |
        | 2025-12-31 | -1.6061 |   0.8429 |     10.5952 |
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

        awesome_oscillator = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            awesome_oscillator[ticker] = momentum_model.get_awesome_oscillator(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
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
    def get_vortex_indicator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Vortex Indicator for a given price series.

        The Vortex Indicator quantifies the presence and strength of a directional trend by
        comparing each period's price movement away from the prior period's range to the
        period's overall volatility (True Range). It consists of two lines, VI+ and VI-, whose
        crossovers signal potential trend changes: VI+ above VI- suggests an uptrend is in
        control, VI- above VI+ suggests a downtrend is in control.

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VI+ and VI- values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame:
            VI+ and VI- values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Vortex Indicator values for each asset in the Toolkit instance.
        - Reference: Botes, E., & Siepman, D. (2010). "The Vortex Indicator." Technical Analysis
          of Stocks & Commodities, 28(1), 20-25.
        - If `growth` is set to True, the method calculates the growth of the VI+ and VI-
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_vortex_indicator().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |    VI+ |    VI- |
        |:-----------|-------:|-------:|
        | 2025-12-17 | 0.9211 | 1.0533 |
        | 2025-12-18 | 0.865  | 1.0802 |
        | 2025-12-19 | 0.903  | 1.137  |
        | 2025-12-22 | 0.8075 | 1.2147 |
        | 2025-12-23 | 0.7732 | 1.2661 |
        | 2025-12-24 | 0.8892 | 1.1613 |
        | 2025-12-26 | 0.9091 | 1.1011 |
        | 2025-12-29 | 0.9316 | 1.0955 |
        | 2025-12-30 | 0.9145 | 1.1096 |
        | 2025-12-31 | 0.9113 | 1.1148 |
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

        vortex_indicator_dict = {}

        for ticker in historical_data[close_column].columns:
            vortex_indicator_dict[ticker] = momentum_model.get_vortex_indicator(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
            ).loc[self._start_date : self._end_date]

        vortex_indicator = (
            pd.concat(vortex_indicator_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Elder Ray Index (Bull Power and Bear Power) for a given price series.

        The Elder Ray Index measures buying and selling pressure in the market relative to a
        trend baseline (an Exponential Moving Average of the closing price). Bull Power captures
        how far the high extends above the EMA (buying pressure), while Bear Power captures how
        far the low extends below the EMA (selling pressure).

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Bull and Bear Power.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
        - If `growth` is set to True, the method calculates the growth of the Bull and Bear Power
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_elder_ray_index().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Bear Power |   Bull Power |
        |:-----------|-------------:|-------------:|
        | 2025-12-17 |      -4.6407 |      -0.1329 |
        | 2025-12-18 |      -8.7334 |      -2.0715 |
        | 2025-12-19 |      -5.5011 |      -0.8138 |
        | 2025-12-22 |      -4.2594 |      -0.8985 |
        | 2025-12-23 |      -4.8618 |      -1.9298 |
        | 2025-12-24 |      -2.14   |       1.0812 |
        | 2025-12-26 |      -1.3471 |       1.1561 |
        | 2025-12-29 |      -1.7914 |       0.2132 |
        | 2025-12-30 |      -1.7093 |       0.0858 |
        | 2025-12-31 |      -1.9339 |      -0.0091 |
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

        elder_ray_index_dict = {}

        for ticker in historical_data[close_column].columns:
            elder_ray_index_dict[ticker] = momentum_model.get_elder_ray_index(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
            ).loc[self._start_date : self._end_date]

        elder_ray_index = (
            pd.concat(elder_ray_index_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Rate of Change (ROC) for a given price series.

        The Rate of Change is a pure momentum oscillator that measures the percentage
        change in price between the current period and the price a fixed number of periods
        ago. It oscillates around zero: positive values indicate price is higher than
        `window` periods ago (upward momentum), while negative values indicate price is
        lower (downward momentum).

        The formula is a follows:

        - ROC = (Close(t) / Close(t - window) — 1) * 100

        Also known as: ROC, Price Rate of Change, momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to look back for the rate of change
                calculation. Defaults to 12.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Rate of Change.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Rate of Change values, expressed as a percentage.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Rate of Change for each asset in the Toolkit instance.
        - Reference: Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New
          York Institute of Finance.
        - If `growth` is set to True, the method calculates the growth of the Rate of Change
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_rate_of_change()
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        rate_of_change = momentum_model.get_rate_of_change(
            historical_data[close_column],
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
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        The formula is a follows:

        - CHOP = 100 * log10( Sum(True Range, window) / (Max(High, window) — Min(Low, window)) ) / log10(window)

        Also known as: CHOP, Choppiness Index.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods to consider for the Choppiness Index
                calculation. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Choppiness Index.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
        - If `growth` is set to True, the method calculates the growth of the Choppiness
          Index using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_choppiness_index()
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        choppiness_index = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            choppiness_index[ticker] = momentum_model.get_choppiness_index(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
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
    def get_know_sure_thing(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        roc_windows: list[int] | None = None,
        sma_windows: list[int] | None = None,
        weights: list[int] | None = None,
        signal_window: int = 9,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        The formula is a follows:

        - RCMA(i) = SMA(ROC(Close, roc_windows[i]), sma_windows[i])
        - KST = Sum(RCMA(i) * weights[i]) for i = 1..4
        - Signal Line = SMA(KST, signal_window)

        Also known as: KST, Pring's Know Sure Thing, Summed Rate of Change.

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the KST and Signal Line.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame:
            KST and Signal Line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Know Sure Thing values for each asset in the Toolkit instance.
        - Reference: Pring, M.J. (1992). "The Know Sure Thing (KST)." Technical Analysis of
          Stocks & Commodities, 10(6).
        - If `growth` is set to True, the method calculates the growth of the KST and Signal
          Line using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_know_sure_thing().xs("AAPL", level=1, axis="columns")
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        know_sure_thing_dict = {}

        for ticker in historical_data[close_column].columns:
            know_sure_thing_dict[ticker] = momentum_model.get_know_sure_thing(
                historical_data[close_column][ticker],
                roc_windows,
                sma_windows,
                weights,
                signal_window,
            ).loc[self._start_date : self._end_date]

        know_sure_thing = (
            pd.concat(know_sure_thing_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
        | 2025-12-17 |                       277.993 |                            275.523 |                                    275.977 | 0.1587 |                     278.193 |                         276.338 |                     273.323 |                                  274.426 |                                278.186 |         285.71  |       273.223 |                    275.394 |                 271.694 |                       279.37  |                       277.371 |
        | 2025-12-18 |                       277.518 |                            274.98  |                                    274.903 | 0.1367 |                     277.449 |                         275.466 |                     272.107 |                                  274.289 |                                277.532 |         284.525 |       272.472 |                    274.041 |                 269.533 |                       277.031 |                       274.481 |
        | 2025-12-19 |                       276.846 |                            274.706 |                                    274.402 | 0.116  |                     276.642 |                         274.854 |                     271.353 |                                  274.166 |                                276.211 |         282.695 |       270.188 |                    274.15  |                 267.489 |                       277.031 |                       274.481 |
        | 2025-12-22 |                       275.762 |                            274.11  |                                    273.33  | 0.0946 |                     275.788 |                         273.972 |                     270.604 |                                  273.358 |                                275.272 |         281.048 |       271.983 |                    274.799 |                 270.111 |                       277.031 |                       274.481 |
        | 2025-12-23 |                       274.922 |                            273.778 |                                    272.814 | 0.0747 |                     274.973 |                         273.42  |                     270.19  |                                  273.115 |                                274.464 |         279.566 |       271.049 |                    272.322 |                 268.961 |                       275.091 |                       272.999 |
        | 2025-12-24 |                       274.432 |                            273.683 |                                    272.766 | 0.0578 |                     274.257 |                         273.173 |                     270.366 |                                  273.112 |                                274.006 |         278.232 |       270.736 |                    272.645 |                 269.713 |                       273.301 |                       271.631 |
        | 2025-12-26 |                       274.048 |                            273.547 |                                    272.633 | 0.0432 |                     273.637 |                         272.936 |                     270.87  |                                  273.087 |                                273.626 |         277.031 |       273.07  |                    274.679 |                 271.458 |                       272.807 |                       271.254 |
        | 2025-12-29 |                       273.754 |                            273.476 |                                    272.623 | 0.031  |                     273.126 |                         272.798 |                     271.532 |                                  273.085 |                                273.368 |         275.951 |       273.133 |                    274.147 |                 271.644 |                       272.807 |                       271.254 |
        | 2025-12-30 |                       273.462 |                            273.324 |                                    272.454 | 0.0204 |                     272.738 |                         272.61  |                     272.005 |                                  273.056 |                                273.119 |         274.978 |       272.748 |                    273.885 |                 271.88  |                       272.807 |                       271.254 |
        | 2025-12-31 |                       272.969 |                            273.031 |                                    272.022 | 0.0101 |                     272.407 |                         272.298 |                     272.09  |                                  272.894 |                                272.801 |         274.103 |       272.405 |                    273.269 |                 271.474 |                       272.807 |                       271.254 |
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
            self.get_exponential_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Double Exponential Moving Average (DEMA)"] = (
            self.get_double_exponential_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["TRIX"] = self.get_trix(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators["Triangular Moving Average"] = (
            self.get_triangular_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Weighted Moving Average (WMA)"] = (
            self.get_weighted_moving_average(
                period=period, close_column=close_column, window=window
            )
        )

        overlap_indicators["Hull Moving Average (HMA)"] = self.get_hull_moving_average(
            period=period, close_column=close_column, window=window
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Moving Average (MA) for a given price series.

        The Moving Average (MA) is a commonly used technical indicator that smooths out
        price data by calculating the average price over a specified number of periods.

        The formula is a follows:

        - MA = (Sum of Prices) / (Number of Prices)

        Also known as: SMA, simple moving average, MA.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the moving average.
                The number of periods (time intervals) over which to calculate the MA. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the MA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Moving Average (MA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          MA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the MA
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_moving_average()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 277.993 | 479.913 |     675.239 |
        | 2025-12-18 | 277.518 | 479.343 |     674.75  |
        | 2025-12-19 | 276.846 | 479.285 |     674.914 |
        | 2025-12-22 | 275.762 | 478.925 |     675.291 |
        | 2025-12-23 | 274.922 | 479.572 |     675.722 |
        | 2025-12-24 | 274.432 | 480.081 |     676.289 |
        | 2025-12-26 | 274.048 | 480.404 |     676.76  |
        | 2025-12-29 | 273.754 | 480.126 |     677.202 |
        | 2025-12-30 | 273.462 | 479.804 |     677.626 |
        | 2025-12-31 | 272.969 | 480.163 |     677.369 |
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

        moving_average = overlap_model.get_moving_average(
            historical_data[close_column], window
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
    def get_exponential_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Exponential Moving Average (EMA) for a given price series.

        EMA is a technical indicator that gives more weight to recent price data,
        providing a smoothed moving average that reacts faster to price changes.

        The formula is a follows:

        - First EMA = SMA(Close, Window), on the Window-th period
        - EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA

        Seeding with a simple average, as TA-Lib and StockCharts do, leaves the first Window - 1
        periods of the price history empty. The DEMA, TRIX, MACD, Percentage Price Oscillator,
        Force Index, Elder Ray Index, Keltner Channels and Chaikin and McClellan Oscillators
        use the same EMA and start once their own EMAs are seeded.

        Also known as: EMA.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for EMA calculation.
                The number of periods (time intervals) over which to calculate the EMA. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the EMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Exponential Moving Average (EMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          EMA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the EMA
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_exponential_moving_average()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 275.523 | 479.029 |     672.561 |
        | 2025-12-18 | 274.98  | 479.282 |     672.12  |
        | 2025-12-19 | 274.706 | 479.759 |     672.546 |
        | 2025-12-22 | 274.11  | 480.039 |     673.476 |
        | 2025-12-23 | 273.778 | 480.538 |     674.696 |
        | 2025-12-24 | 273.683 | 481.126 |     676.074 |
        | 2025-12-26 | 273.547 | 481.594 |     677.259 |
        | 2025-12-29 | 273.476 | 481.918 |     677.96  |
        | 2025-12-30 | 273.324 | 482.25  |     678.457 |
        | 2025-12-31 | 273.031 | 482.026 |     678.214 |
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

        exponential_moving_average = overlap_model.get_exponential_moving_average(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=exponential_moving_average,
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
    def get_double_exponential_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Double Exponential Moving Average (DEMA) for a given price series.

        DEMA is a technical indicator that attempts to reduce the lag from traditional
        moving averages by using a combination of two exponential moving averages.

        The formula is a follows:

        - EMA1 = EMA(Close, Window)
        - EMA2 = EMA(EMA1, Window)
        - DEMA = 2 * EMA1 — EMA2

        Also known as: DEMA, double EMA.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for moving average calculation.
                The number of periods (time intervals) over which to calculate the moving average. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the DEMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Double Exponential Moving Average (DEMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          DEMA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the DEMA
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_double_exponential_moving_average()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 275.977 | 473.137 |     673.922 |
        | 2025-12-18 | 274.903 | 474.395 |     672.916 |
        | 2025-12-19 | 274.402 | 475.937 |     673.605 |
        | 2025-12-22 | 273.33  | 476.97  |     675.2   |
        | 2025-12-23 | 272.814 | 478.31  |     677.248 |
        | 2025-12-24 | 272.766 | 479.704 |     679.48  |
        | 2025-12-26 | 272.633 | 480.767 |     681.237 |
        | 2025-12-29 | 272.623 | 481.483 |     682.016 |
        | 2025-12-30 | 272.454 | 482.161 |     682.403 |
        | 2025-12-31 | 272.022 | 481.755 |     681.423 |
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

        double_exponential_moving_average = (
            overlap_model.get_double_exponential_moving_average(
                historical_data[close_column], window
            ).loc[self._start_date : self._end_date]
        )

        return finalize_dataset(
            dataset=double_exponential_moving_average,
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
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Trix (Triple Exponential Moving Average) for a given price series.

        Trix is a momentum oscillator that calculates the percentage rate of change of a triple
        exponentially smoothed moving average. It helps identify overbought and oversold conditions
        in a market.

        The formula is a follows:

        - EMA1 = EMA(Close, Window)
        - EMA2 = EMA(EMA1, Window)
        - EMA3 = EMA(EMA2, Window)
        - TRIX = 100 * ((EMA3 — EMA3[—1]) / EMA3[—1])

        Also known as: triple smoothed EMA, rate of change oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for moving average calculation.
                The number of periods (time intervals) over which to calculate the moving average. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Trix.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Trix values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Trix for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Trix
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_trix()
        ```

        Which returns:

        | Date       |   AAPL |    MSFT |   Benchmark |
        |:-----------|-------:|--------:|------------:|
        | 2025-12-17 | 0.1587 | -0.2001 |      0.0628 |
        | 2025-12-18 | 0.1367 | -0.1941 |      0.0568 |
        | 2025-12-19 | 0.116  | -0.1846 |      0.0524 |
        | 2025-12-22 | 0.0946 | -0.1731 |      0.0507 |
        | 2025-12-23 | 0.0747 | -0.1596 |      0.0518 |
        | 2025-12-24 | 0.0578 | -0.1446 |      0.0553 |
        | 2025-12-26 | 0.0432 | -0.1289 |      0.06   |
        | 2025-12-29 | 0.031  | -0.1137 |      0.0644 |
        | 2025-12-30 | 0.0204 | -0.0991 |      0.0678 |
        | 2025-12-31 | 0.0101 | -0.0871 |      0.0686 |
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

        trix = overlap_model.get_trix(historical_data[close_column], window).loc[
            self._start_date : self._end_date
        ]

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
    def get_bollinger_bands(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        num_std_dev: int = 2,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Bollinger Bands for a given price series.

        Bollinger Bands are a volatility indicator that consists of three lines: an upper band,
        a middle band (simple moving average), and a lower band. The upper and lower bands are
        calculated as the moving average plus and minus a specified number of standard deviations,
        respectively.

        The formula is a follows:

        - Middle Band = SMA(Close, Window)
        - Upper Band = Middle Band + (Num Std Dev * Std Dev)
        - Lower Band = Middle Band — (Num Std Dev * Std Dev)

        The standard deviation is the *population* standard deviation (dividing by n), as
        Bollinger himself specifies and as TA-Lib and StockCharts both implement, not
        pandas' default sample standard deviation.

        The band width tracks volatility: bands that narrow sharply (a squeeze) often precede a
        larger move, without saying in which direction. A close at or beyond a band is not a sell
        or buy signal by itself, since prices can walk along a band in a trend. Bollinger's own
        setting is a 20-period average with 2 standard deviations; this method defaults to a
        window of 14, so pass window=20 to match most charting platforms.

        Also known as: Bollinger Bands, BB, volatility bands, price channels.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for moving average calculation.
                The number of periods (time intervals) over which to calculate the moving average. Defaults to 14.
            num_std_dev (int, optional): Number of standard deviations for the bands.
                Defaults to 2.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the bands.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame:
            Bollinger Bands (upper, middle, lower).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Bollinger Bands for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Bollinger Bands
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_bollinger_bands().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Close |   Lower Band |   Middle Band |   Upper Band |
        |:-----------|--------:|-------------:|--------------:|-------------:|
        | 2025-12-17 | 271.102 |      270.499 |       277.993 |      285.486 |
        | 2025-12-18 | 271.451 |      269.304 |       277.518 |      285.733 |
        | 2025-12-19 | 272.927 |      268.78  |       276.846 |      284.913 |
        | 2025-12-22 | 270.234 |      268.558 |       275.762 |      282.966 |
        | 2025-12-23 | 271.621 |      268.807 |       274.922 |      281.038 |
        | 2025-12-24 | 273.067 |      268.933 |       274.432 |      279.93  |
        | 2025-12-26 | 272.658 |      268.866 |       274.048 |      279.231 |
        | 2025-12-29 | 273.017 |      268.846 |       273.754 |      278.663 |
        | 2025-12-30 | 272.339 |      268.742 |       273.462 |      278.183 |
        | 2025-12-31 | 271.122 |      268.854 |       272.969 |      277.084 |
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

        bollinger_bands_dict = {}

        for ticker in historical_data[close_column].columns:
            bollinger_bands_dict[ticker] = volatility_model.get_bollinger_bands(
                historical_data[close_column][ticker], window, num_std_dev
            ).loc[self._start_date : self._end_date]

        bollinger_bands = (
            pd.concat(bollinger_bands_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

        return finalize_dataset(
            dataset=bollinger_bands,
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
    def get_triangular_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Triangular Moving Average (TMA) for a given price series.

        The Triangular Moving Average (TMA) is a smoothed version of the Simple Moving Average (SMA)
        that uses multiple SMAs to reduce noise and provide a smoother trendline.

        The formula is a follows:

        - For an odd window: Sub-window Length = (Window + 1) / 2, applied for both passes.
        - For an even window: the two passes use different sub-window lengths, Window / 2
          and Window / 2 + 1 (matching TA-Lib's TRIMA convention).
        - TMA = SMA(SMA(Close, Sub-window Length 1), Sub-window Length 2)

        Also known as: TMA, triangular MA.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for TMA calculation.
                The number of periods (time intervals) over which to calculate the TMA. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the TMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Triangular Moving Average values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Triangular Moving Average for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Triangular Moving Average
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_triangular_moving_average()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 278.193 | 480.648 |     676.567 |
        | 2025-12-18 | 277.449 | 479.767 |     676.27  |
        | 2025-12-19 | 276.642 | 479.017 |     675.891 |
        | 2025-12-22 | 275.788 | 478.438 |     675.342 |
        | 2025-12-23 | 274.973 | 478.122 |     674.804 |
        | 2025-12-24 | 274.257 | 478.026 |     674.51  |
        | 2025-12-26 | 273.637 | 478.238 |     674.519 |
        | 2025-12-29 | 273.126 | 478.765 |     674.975 |
        | 2025-12-30 | 272.738 | 479.618 |     675.869 |
        | 2025-12-31 | 272.407 | 480.573 |     676.902 |
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

        triangular_moving_average = overlap_model.get_triangular_moving_average(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=triangular_moving_average,
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
    def get_weighted_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Weighted Moving Average (WMA) for a given price series.

        The Weighted Moving Average (WMA) is a moving average that assigns a linearly
        increasing weight to more recent prices, making it more responsive to recent
        price changes than a Simple Moving Average.

        The formula is a follows:

        - WMA = (Sum of (Price * Weight)) / (Sum of Weights)

        Also known as: WMA, linearly weighted moving average.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the WMA.
                The number of periods (time intervals) over which to calculate the WMA. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the WMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Weighted Moving Average (WMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          WMA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the WMA
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_weighted_moving_average()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 276.338 | 477.779 |     674.464 |
        | 2025-12-18 | 275.466 | 477.914 |     673.665 |
        | 2025-12-19 | 274.854 | 478.383 |     673.741 |
        | 2025-12-22 | 273.972 | 478.726 |     674.355 |
        | 2025-12-23 | 273.42  | 479.374 |     675.334 |
        | 2025-12-24 | 273.173 | 480.09  |     676.575 |
        | 2025-12-26 | 272.936 | 480.697 |     677.731 |
        | 2025-12-29 | 272.798 | 481.18  |     678.499 |
        | 2025-12-30 | 272.61  | 481.751 |     679.096 |
        | 2025-12-31 | 272.298 | 481.854 |     678.964 |
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

        weighted_moving_average = overlap_model.get_weighted_moving_average(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=weighted_moving_average,
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
    def get_hull_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Hull Moving Average (HMA) for a given price series.

        The Hull Moving Average (HMA) reduces the lag typically associated with moving
        averages while improving smoothing, by combining a Weighted Moving Average (WMA)
        of half the window length, a WMA of the full window length, and a further WMA
        over the square root of the window length.

        The formula is a follows:

        - Raw HMA = (2 * WMA(Close, Window / 2)) — WMA(Close, Window)
        - HMA = WMA(Raw HMA, sqrt(Window))

        Also known as: HMA, Hull MA.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the HMA.
                The number of periods (time intervals) over which to calculate the HMA. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the HMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Hull Moving Average (HMA) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          HMA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the HMA
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_hull_moving_average()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 273.323 | 473.222 |     672.879 |
        | 2025-12-18 | 272.107 | 473.091 |     670.097 |
        | 2025-12-19 | 271.353 | 474.901 |     669.138 |
        | 2025-12-22 | 270.604 | 477.304 |     670.447 |
        | 2025-12-23 | 270.19  | 479.884 |     673.51  |
        | 2025-12-24 | 270.366 | 482.461 |     677.52  |
        | 2025-12-26 | 270.87  | 484.613 |     681.486 |
        | 2025-12-29 | 271.532 | 485.933 |     684.361 |
        | 2025-12-30 | 272.005 | 486.423 |     685.736 |
        | 2025-12-31 | 272.09  | 485.674 |     684.953 |
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

        hull_moving_average = overlap_model.get_hull_moving_average(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=hull_moving_average,
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

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the KAMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: KAMA values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the KAMA for each asset in the Toolkit instance.
        - Reference: Kaufman, P.J. (1998). "Trading Systems and Methods." 3rd ed. Wiley.
        - If `growth` is set to True, the method calculates the growth of the KAMA using the
          specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_kaufman_adaptive_moving_average()
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        kaufman_adaptive_moving_average = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            kaufman_adaptive_moving_average[ticker] = (
                overlap_model.get_kaufman_adaptive_moving_average(
                    historical_data[close_column][ticker],
                    window,
                    fast_window,
                    slow_window,
                ).loc[self._start_date : self._end_date]
            )

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Volume Weighted Average Price (VWAP) for a given price series.

        The Volume Weighted Average Price (VWAP) weighs the typical price of each period
        by its traded volume over a rolling window, giving a more volume-informed view of
        the average price than a plain moving average.

        The formula is a follows:

        - Typical Price = (High + Low + Close) / 3
        - VWAP = Sum(Typical Price * Volume, Window) / Sum(Volume, Window)

        The VWAP that trading desks use as an execution benchmark is calculated from the trades
        of a single session and resets every day. This method computes a rolling VWAP over several
        periods from the high, low, close and volume of each period, which works as a
        volume-weighted moving average, not as the intraday benchmark against which fills are
        measured.

        Also known as: VWAP.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods to consider for the VWAP.
                The number of periods (time intervals) over which to calculate the VWAP. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VWAP.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Volume Weighted Average Price (VWAP) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          VWAP for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the VWAP
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_volume_weighted_average_price()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 278.186 | 478.827 |     674.906 |
        | 2025-12-18 | 277.532 | 478.63  |     674.443 |
        | 2025-12-19 | 276.211 | 478.981 |     674.468 |
        | 2025-12-22 | 275.272 | 478.73  |     674.699 |
        | 2025-12-23 | 274.464 | 479.175 |     675.016 |
        | 2025-12-24 | 274.006 | 479.432 |     675.253 |
        | 2025-12-26 | 273.626 | 479.603 |     675.357 |
        | 2025-12-29 | 273.368 | 479.287 |     675.675 |
        | 2025-12-30 | 273.119 | 479.124 |     675.902 |
        | 2025-12-31 | 272.801 | 479.609 |     675.857 |
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

        volume_weighted_average_price = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            volume_weighted_average_price[ticker] = (
                overlap_model.get_volume_weighted_average_price(
                    historical_data["High"][ticker],
                    historical_data["Low"][ticker],
                    historical_data[close_column][ticker],
                    historical_data["Volume"][ticker],
                    window,
                ).loc[self._start_date : self._end_date]
            )

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Parabolic Stop and Reverse (SAR) for a given price series.

        The Parabolic SAR is a trend-following indicator that trails price action,
        flipping from below to above price (and vice versa) whenever the trend reverses.
        The acceleration factor increases as the trend extends, causing the SAR to
        converge towards price over time.

        The formula is a follows:

        - Uptrend SAR = Prior SAR + AF * (Extreme Point — Prior SAR)
        - Downtrend SAR = Prior SAR — AF * (Prior SAR — Extreme Point)

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Parabolic SAR.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame or pd.Series:
            Parabolic SAR values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Parabolic SAR for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Parabolic SAR
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_parabolic_sar()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-17 | 285.71  | 487.285 |     681.611 |
        | 2025-12-18 | 284.525 | 467.911 |     680.908 |
        | 2025-12-19 | 282.695 | 468.284 |     680.233 |
        | 2025-12-22 | 281.048 | 468.648 |     664.036 |
        | 2025-12-23 | 279.566 | 469.005 |     664.356 |
        | 2025-12-24 | 278.232 | 469.356 |     665.097 |
        | 2025-12-26 | 277.031 | 469.699 |     666.319 |
        | 2025-12-29 | 275.951 | 470.035 |     667.918 |
        | 2025-12-30 | 274.978 | 470.365 |     669.388 |
        | 2025-12-31 | 274.103 | 471.014 |     670.741 |
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

        parabolic_sar = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            parabolic_sar[ticker] = overlap_model.get_parabolic_sar(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Pivot Points for a given price series.

        Pivot Points are calculated from the previous period's high, low and close
        prices and are used to identify potential support and resistance levels for
        the current period.

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Pivot Points.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_pivot_points().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Pivot Point |   Resistance 1 |   Resistance 2 |   Resistance 3 |   Support 1 |   Support 2 |   Support 3 |
        |:-----------|--------------:|---------------:|---------------:|---------------:|------------:|------------:|------------:|
        | 2025-12-17 |       273.223 |        275.394 |        276.923 |        279.094 |     271.694 |     269.523 |     267.994 |
        | 2025-12-18 |       272.472 |        274.041 |        276.979 |        278.548 |     269.533 |     267.964 |     265.025 |
        | 2025-12-19 |       270.188 |        274.15  |        276.85  |        280.812 |     267.489 |     263.526 |     260.827 |
        | 2025-12-22 |       271.983 |        274.799 |        276.67  |        279.486 |     270.111 |     267.296 |     265.424 |
        | 2025-12-23 |       271.049 |        272.322 |        274.41  |        275.683 |     268.961 |     267.688 |     265.6   |
        | 2025-12-24 |       270.736 |        272.645 |        273.668 |        275.577 |     269.713 |     267.804 |     266.781 |
        | 2025-12-26 |       273.07  |        274.679 |        276.291 |        277.9   |     271.458 |     269.849 |     268.236 |
        | 2025-12-29 |       273.133 |        274.147 |        275.636 |        276.65  |     271.644 |     270.63  |     269.141 |
        | 2025-12-30 |       272.748 |        273.885 |        274.752 |        275.889 |     271.88  |     270.743 |     269.875 |
        | 2025-12-31 |       272.405 |        273.269 |        274.2   |        275.065 |     271.474 |     270.61  |     269.679 |
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

        pivot_points_dict = {}

        for ticker in historical_data[close_column].columns:
            pivot_points_dict[ticker] = overlap_model.get_pivot_points(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
            ).loc[self._start_date : self._end_date]

        pivot_points = (
            pd.concat(pivot_points_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

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
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        support_resistance_levels = toolkit.technicals.get_support_resistance_levels()

        support_resistance_levels.xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Resistance |   Support |
        |:-----------|-------------:|----------:|
        | 2025-12-22 |      285.413 |   265.527 |
        | 2025-12-23 |      285.413 |   265.527 |
        | 2025-12-24 |      285.413 |   265.527 |
        | 2025-12-26 |      285.413 |   265.527 |
        | 2025-12-29 |      285.413 |   265.527 |
        | 2025-12-30 |      285.413 |   265.527 |
        | 2025-12-31 |      285.413 |   265.527 |
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

        support_resistance_levels = {}

        for ticker in historical_data[close_column].columns:
            support_resistance_levels[ticker] = (
                overlap_model.get_support_resistance_levels(
                    prices=historical_data[close_column][ticker],
                    window=window,
                    sensitivity=sensitivity,
                )
            )

        support_resistance_levels_df = (
            pd.concat(support_resistance_levels, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Fibonacci Retracement Levels for a given price series.

        Fibonacci Retracement Levels are horizontal price levels, derived from ratios found in
        the Fibonacci sequence, that traders watch as potential support (during a pullback
        within an uptrend) or resistance (during a bounce within a downtrend) zones. For every
        date, the swing high and swing low are taken as the rolling maximum high and rolling
        minimum low over the specified `window`, and the retracement levels are derived from
        that high/low pair.

        The formula is a follows:

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
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the retracement levels.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_fibonacci_retracement_levels().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |    0.0% |   100.0% |   23.6% |   38.2% |   50.0% |   61.8% |   78.6% |
        |:-----------|--------:|---------:|--------:|--------:|--------:|--------:|--------:|
        | 2025-12-17 | 287.837 |  270.903 | 283.84  | 281.368 | 279.37  | 277.371 | 274.526 |
        | 2025-12-18 | 287.837 |  266.225 | 282.736 | 279.581 | 277.031 | 274.481 | 270.85  |
        | 2025-12-19 | 287.837 |  266.225 | 282.736 | 279.581 | 277.031 | 274.481 | 270.85  |
        | 2025-12-22 | 287.837 |  266.225 | 282.736 | 279.581 | 277.031 | 274.481 | 270.85  |
        | 2025-12-23 | 283.957 |  266.225 | 279.772 | 277.183 | 275.091 | 272.999 | 270.02  |
        | 2025-12-24 | 280.377 |  266.225 | 277.037 | 274.971 | 273.301 | 271.631 | 269.254 |
        | 2025-12-26 | 279.389 |  266.225 | 276.283 | 274.361 | 272.807 | 271.254 | 269.043 |
        | 2025-12-29 | 279.389 |  266.225 | 276.283 | 274.361 | 272.807 | 271.254 | 269.043 |
        | 2025-12-30 | 279.389 |  266.225 | 276.283 | 274.361 | 272.807 | 271.254 | 269.043 |
        | 2025-12-31 | 279.389 |  266.225 | 276.283 | 274.361 | 272.807 | 271.254 | 269.043 |
        Note that the columns sort lexicographically by their label (so "100.0%" sorts right
        after "0.0%", ahead of "23.6%"), matching the sorting convention used by every other
        multi-column indicator in this module (e.g. Pivot Points' Resistance/Support levels).

        | Date       |    0.0% |   100.0% |   23.6% |   38.2% |   50.0% |   61.8% |   78.6% |
        |:-----------|--------:|---------:|--------:|--------:|--------:|--------:|--------:|
        | 2025-12-17 | 287.837 |  270.903 | 283.84  | 281.368 | 279.37  | 277.371 | 274.526 |
        | 2025-12-18 | 287.837 |  266.225 | 282.736 | 279.581 | 277.031 | 274.481 | 270.85  |
        | 2025-12-19 | 287.837 |  266.225 | 282.736 | 279.581 | 277.031 | 274.481 | 270.85  |
        | 2025-12-22 | 287.837 |  266.225 | 282.736 | 279.581 | 277.031 | 274.481 | 270.85  |
        | 2025-12-23 | 283.957 |  266.225 | 279.772 | 277.183 | 275.091 | 272.999 | 270.02  |
        | 2025-12-24 | 280.377 |  266.225 | 277.037 | 274.971 | 273.301 | 271.631 | 269.254 |
        | 2025-12-26 | 279.389 |  266.225 | 276.283 | 274.361 | 272.807 | 271.254 | 269.043 |
        | 2025-12-29 | 279.389 |  266.225 | 276.283 | 274.361 | 272.807 | 271.254 | 269.043 |
        | 2025-12-30 | 279.389 |  266.225 | 276.283 | 274.361 | 272.807 | 271.254 | 269.043 |
        | 2025-12-31 | 279.389 |  266.225 | 276.283 | 274.361 | 272.807 | 271.254 | 269.043 |
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

        fibonacci_retracement_levels_dict = {}

        for ticker in historical_data[close_column].columns:
            rolling_high = historical_data["High"][ticker].rolling(window=window).max()
            rolling_low = historical_data["Low"][ticker].rolling(window=window).min()

            fibonacci_retracement_levels_dict[ticker] = (
                overlap_model.get_fibonacci_retracement_levels(
                    rolling_high, rolling_low, levels=levels, trend=trend
                ).loc[self._start_date : self._end_date]
            )

        fibonacci_retracement_levels = (
            pd.concat(fibonacci_retracement_levels_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

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

    @handle_errors
    def collect_volatility_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects various volatility indicators based on the provided data.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            window (int, optional): The window size for calculating indicators.
                Defaults to 14.
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Volatility indicators calculated based on the specified parameters.

        Notes:
        - The method calculates several volatility-based indicators for each asset in the Toolkit instance.
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

        toolkit.technicals.collect_volatility_indicators().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Bollinger Band Upper |   Bollinger Band Middle |   Bollinger Band Lower |   True Range |   Average True Range |   Supertrend |   Supertrend Trend Direction |   Keltner Channel Upper |   Keltner Channel Middle |   Keltner Channel Lower |   Donchian Channel Upper |   Donchian Channel Middle |   Donchian Channel Lower |
        |:-----------|-----------------------:|------------------------:|-----------------------:|-------------:|---------------------:|-------------:|-----------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|-------------------------:|
        | 2025-12-17 |                285.486 |                 277.993 |                270.499 |       4.5077 |               4.8747 |      268.669 |                            1 |                 285.011 |                  275.523 |                 266.035 |                  287.837 |                   279.444 |                  271.052 |
        | 2025-12-18 |                285.733 |                 277.518 |                269.304 |       6.6619 |               5.0024 |      268.669 |                            1 |                 284.851 |                  274.98  |                 265.109 |                  287.837 |                   279.37  |                  270.903 |
        | 2025-12-19 |                284.913 |                 276.846 |                268.78  |       4.6872 |               4.9799 |      268.669 |                            1 |                 284.528 |                  274.706 |                 264.885 |                  287.837 |                   277.031 |                  266.225 |
        | 2025-12-22 |                282.966 |                 275.762 |                268.558 |       3.3609 |               4.8642 |      268.669 |                            1 |                 283.622 |                  274.11  |                 264.598 |                  287.837 |                   277.031 |                  266.225 |
        | 2025-12-23 |                281.038 |                 274.922 |                268.807 |       2.932  |               4.7262 |      268.669 |                            1 |                 282.925 |                  273.778 |                 264.631 |                  287.837 |                   277.031 |                  266.225 |
        | 2025-12-24 |                279.93  |                 274.432 |                268.933 |       3.2212 |               4.6187 |      268.669 |                            1 |                 282.56  |                  273.683 |                 264.807 |                  283.957 |                   275.091 |                  266.225 |
        | 2025-12-26 |                279.231 |                 274.048 |                268.866 |       2.5032 |               4.4676 |      268.669 |                            1 |                 282.036 |                  273.547 |                 265.057 |                  280.377 |                   273.301 |                  266.225 |
        | 2025-12-29 |                278.663 |                 273.754 |                268.846 |       2.0045 |               4.2917 |      268.669 |                            1 |                 281.517 |                  273.476 |                 265.435 |                  279.389 |                   272.807 |                  266.225 |
        | 2025-12-30 |                278.183 |                 273.462 |                268.742 |       1.7951 |               4.1134 |      268.669 |                            1 |                 280.921 |                  273.324 |                 265.728 |                  279.389 |                   272.807 |                  266.225 |
        | 2025-12-31 |                277.084 |                 272.969 |                268.854 |       1.9248 |               3.957  |      268.669 |                            1 |                 280.252 |                  273.031 |                 265.809 |                  279.389 |                   272.807 |                  266.225 |
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

        volatility_indicators: dict = {}

        bollinger_bands = self.get_bollinger_bands(
            period=period, close_column=close_column, window=window
        )

        volatility_indicators["Bollinger Band Upper"] = bollinger_bands["Upper Band"]
        volatility_indicators["Bollinger Band Middle"] = bollinger_bands["Middle Band"]
        volatility_indicators["Bollinger Band Lower"] = bollinger_bands["Lower Band"]

        volatility_indicators["True Range"] = self.get_true_range(
            period=period, close_column=close_column
        )

        volatility_indicators["Average True Range"] = self.get_average_true_range(
            period=period, close_column=close_column, window=window
        )

        supertrend = self.get_supertrend(period=period, close_column=close_column)

        volatility_indicators["Supertrend"] = supertrend["Supertrend"]
        volatility_indicators["Supertrend Trend Direction"] = supertrend[
            "Trend Direction"
        ]

        keltner_channels = self.get_keltner_channels(
            period=period, close_column=close_column, window=window
        )

        volatility_indicators["Keltner Channel Upper"] = keltner_channels["Upper Line"]
        volatility_indicators["Keltner Channel Middle"] = keltner_channels[
            "Middle Line"
        ]
        volatility_indicators["Keltner Channel Lower"] = keltner_channels["Lower Line"]

        donchian_channels = self.get_donchian_channels(
            period=period, close_column=close_column, window=window
        )

        volatility_indicators["Donchian Channel Upper"] = donchian_channels[
            "Upper Channel"
        ]
        volatility_indicators["Donchian Channel Middle"] = donchian_channels[
            "Middle Channel"
        ]
        volatility_indicators["Donchian Channel Lower"] = donchian_channels[
            "Lower Channel"
        ]

        self._volatility_indicators = pd.concat(volatility_indicators, axis=1)

        self._volatility_indicators = apply_rounding(
            self._volatility_indicators,
            rounding if rounding is not None else self._rounding,
        ).loc[self._start_date : self._end_date]

        if growth:
            self._volatility_indicators_growth = calculate_growth(
                dataset=self._volatility_indicators,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="index",
            )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                self._volatility_indicators_growth = calculate_standardization(
                    dataset=self._volatility_indicators_growth,
                    rounding=standardize_rounding,
                    axis="rows",
                )
            else:
                self._volatility_indicators = calculate_standardization(
                    dataset=self._volatility_indicators,
                    rounding=standardize_rounding,
                    axis="rows",
                )

        if len(self._tickers) == 1:
            return (
                self._volatility_indicators_growth.xs(
                    self._tickers[0], level=1, axis="columns"
                )
                if growth
                else self._volatility_indicators.xs(
                    self._tickers[0], level=1, axis="columns"
                )
            )

        return (
            self._volatility_indicators_growth
            if growth
            else self._volatility_indicators
        )

    @handle_portfolio
    @handle_errors
    def get_true_range(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the True Range (TR) for a given price series.

        The True Range (TR) is a measure of market volatility that considers the differences
        between the high and low prices and the previous closing price. It provides insights
        into the price movement of an asset.

        The formula is a follows:

        - TR = max(high — low, abs(high — previous_close), abs(low — previous_close))

        Also known as: TR, true range.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the True Range.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: True Range values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          True Range for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the True Range
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_true_range()
        ```

        Which returns:

        | Date       |   AAPL |    MSFT |   Benchmark |
        |:-----------|-------:|--------:|------------:|
        | 2025-12-17 | 4.5077 |  4.9685 |      9.1414 |
        | 2025-12-18 | 6.6619 | 13.3951 |      9.2403 |
        | 2025-12-19 | 4.6872 |  5.3262 |      6.5618 |
        | 2025-12-22 | 3.3609 |  6.0019 |      4.733  |
        | 2025-12-23 | 2.932  |  3.0705 |      4.2964 |
        | 2025-12-24 | 3.2212 |  4.3027 |      3.0065 |
        | 2025-12-26 | 2.5032 |  2.1464 |      2.3715 |
        | 2025-12-29 | 2.0045 |  4.1437 |      4.2072 |
        | 2025-12-30 | 1.7951 |  4.1536 |      1.9647 |
        | 2025-12-31 | 1.9248 |  4.8095 |      5.6062 |
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

        true_range = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            true_range[ticker] = volatility_model.get_true_range(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
            ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=true_range,
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
    def get_average_true_range(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate the Average True Range (ATR) of a given price series.

        The Average True Range (ATR) is a technical indicator that measures the volatility
        of an asset's price movements over a specified number of periods. It provides insights
        into the potential price range of an asset, which can help traders and investors make
        more informed decisions.

        The formula is a follows:

        - TR = max(high — low, abs(high — previous_close), abs(low — previous_close))
        - ATR = Wilder's Smoothed Moving Average of TR over `window` periods

        The ATR is in price units, which makes it the common basis for stop distances (for example
        two ATRs below the entry) and for sizing positions to equal risk. For the same reason it
        cannot be compared across stocks or over long periods in which the price changed a lot;
        divide it by the price for a percentage volatility. It measures how far price moves, not in
        which direction.

        Also known as: ATR, volatility indicator. See `volatility_model.get_average_true_range`
        for the full formula; Wilder's smoothing constant (1/window) is slower than a
        standard EMA's (2/(window+1)) of the same window.

        The first period has no previous close, so its True Range is High — Low and the first ATR
        is the average of the first `window` True Ranges (the StockCharts convention; TA-Lib starts
        one period later).

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods over which to calculate the Average
                True Range. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding set on the Toolkit (4 unless set
                otherwise).
            growth (bool, optional): Whether to return the growth of the ATR instead of its
                level. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: ATR values per ticker or ATR growth rate (if growth is True).
                A pandas Series containing the calculated Average True Range values or growth rate for each period.

        Notes:
        - ATR values are typically used to assess the volatility and potential price movement of an asset.
        - A higher ATR value indicates higher volatility, while a lower ATR value suggests lower volatility.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_average_true_range()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-17 | 4.8747 | 9.116  |      7.3629 |
        | 2025-12-18 | 5.0024 | 9.4216 |      7.497  |
        | 2025-12-19 | 4.9799 | 9.1291 |      7.4302 |
        | 2025-12-22 | 4.8642 | 8.9057 |      7.2376 |
        | 2025-12-23 | 4.7262 | 8.4889 |      7.0275 |
        | 2025-12-24 | 4.6187 | 8.1899 |      6.7403 |
        | 2025-12-26 | 4.4676 | 7.7582 |      6.4282 |
        | 2025-12-29 | 4.2917 | 7.5    |      6.2696 |
        | 2025-12-30 | 4.1134 | 7.261  |      5.9621 |
        | 2025-12-31 | 3.957  | 7.0859 |      5.9367 |
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

        average_true_range = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            average_true_range[ticker] = volatility_model.get_average_true_range(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
            ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=average_true_range,
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
    def get_supertrend(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        multiplier: float = 3.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Supertrend indicator for a given price series.

        The Supertrend indicator plots a single trailing line that flips between sitting
        below price (in an uptrend) and above price (in a downtrend). The line is built from
        two bands offset from the median price ((High + Low) / 2) by a multiple of the
        Average True Range, which are then "ratcheted" period over period — each band can
        only move in the direction that tightens around price — so that the active band only
        flips to the other side once the closing price actually crosses it. This makes
        Supertrend both a trend filter (the flip direction signals a trend change) and a
        trailing stop-loss level.

        The formula is a follows:

        - Basic Upper Band = (High + Low) / 2 + multiplier * ATR(window)
        - Basic Lower Band = (High + Low) / 2 — multiplier * ATR(window)
        - Final Upper Band(t) = Basic Upper Band(t) if Basic Upper Band(t) < Final Upper
          Band(t-1) or Close(t-1) > Final Upper Band(t-1), else Final Upper Band(t-1)
        - Final Lower Band(t) = Basic Lower Band(t) if Basic Lower Band(t) > Final Lower
          Band(t-1) or Close(t-1) < Final Lower Band(t-1), else Final Lower Band(t-1)
        - While in an uptrend, Supertrend = Final Lower Band, until Close crosses below it,
          at which point the trend flips to a downtrend and Supertrend = Final Upper Band
          (and vice versa)

        Also known as: Supertrend, SuperTrend.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods for the underlying Average True Range
                calculation. Defaults to 10.
            multiplier (float, optional): Multiplier applied to the Average True Range to
                determine how far the bands sit from the median price. Defaults to 3.0.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Supertrend and
                Trend Direction values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame:
            Supertrend (the trailing indicator line) and Trend Direction (1 for an uptrend
            and -1 for a downtrend) values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Supertrend for each asset in the Toolkit instance.
        - There is no academic journal citation for Supertrend. Like the Parabolic SAR, it is
          a practitioner-developed trailing-stop/trend indicator rather than one derived from
          a published financial paper.
        - The trend is initialized as an uptrend on the first available period, since there
          is no prior period to determine the starting direction from.
        - If `growth` is set to True, the method calculates the growth of the Supertrend and
          Trend Direction values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_supertrend().xs("AAPL", level=1, axis="columns")
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
        if period == "intraday":
            if self._historical_data[period].empty:
                raise ValueError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        supertrend_dict = {}

        for ticker in historical_data[close_column].columns:
            supertrend_dict[ticker] = volatility_model.get_supertrend(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
                multiplier,
            ).loc[self._start_date : self._end_date]

        supertrend = (
            pd.concat(supertrend_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

        return finalize_dataset(
            dataset=supertrend,
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
    def get_keltner_channels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        atr_window: int = 10,
        atr_multiplier: int = 2,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Keltner Channels for a given price series.

        The Keltner Channels consist of three lines: a middle line, the Exponential Moving Average
        (EMA) of the closing prices, and an upper and a lower line a multiple of the Average True
        Range (ATR) above and below it.

        The formula is a follows:

        - Middle Line = EMA(Close, window)
        - Upper Line = Middle Line + atr_multiplier * ATR(atr_window)
        - Lower Line = Middle Line — atr_multiplier * ATR(atr_window), where the ATR uses Wilder's smoothing.

        Also known as: ATR-based bands, volatility channels.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods for the Exponential Moving Average (the
                middle line). Defaults to 20, as on StockCharts.
            atr_window (int, optional): Number of periods for ATR calculation.
                Defaults to 10, as on StockCharts.
            atr_multiplier (int, optional): Multiplier for ATR to determine channel width.
                Defaults to 2.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the channels.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Keltner Channels (upper, middle, lower).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates Keltner Channels
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the channels using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_keltner_channels().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Lower Line |   Middle Line |   Upper Line |
        |:-----------|-------------:|--------------:|-------------:|
        | 2025-12-17 |      265.433 |       274.92  |      284.408 |
        | 2025-12-18 |      264.719 |       274.59  |      284.461 |
        | 2025-12-19 |      264.61  |       274.432 |      284.253 |
        | 2025-12-22 |      264.52  |       274.032 |      283.544 |
        | 2025-12-23 |      264.655 |       273.802 |      282.949 |
        | 2025-12-24 |      264.856 |       273.732 |      282.609 |
        | 2025-12-26 |      265.14  |       273.63  |      282.119 |
        | 2025-12-29 |      265.53  |       273.572 |      281.613 |
        | 2025-12-30 |      265.858 |       273.454 |      281.05  |
        | 2025-12-31 |      266.01  |       273.232 |      280.454 |
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

        keltner_channels_dict = {}

        for ticker in historical_data[close_column].columns:
            keltner_channels_dict[ticker] = volatility_model.get_keltner_channels(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
                atr_window,
                atr_multiplier,
            ).loc[self._start_date : self._end_date]

        kelter_channels = (
            pd.concat(keltner_channels_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

        return finalize_dataset(
            dataset=kelter_channels,
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
    def get_donchian_channels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Donchian Channels for a given price series.

        Donchian Channels plot the highest high and lowest low over the `window` periods
        *preceding* the current one, with the middle line being the average of the two. They
        are used to identify breakouts and the overall volatility of the price range.

        The formula is a follows:

        - Upper Channel = Highest High over Window, ending one period ago
        - Lower Channel = Lowest Low over Window, ending one period ago
        - Middle Channel = (Upper Channel + Lower Channel) / 2

        The current period is deliberately excluded from the lookback, per Donchian's
        original breakout rule and StockCharts' Price Channels definition — including it
        would make a channel break impossible by construction.

        Also known as: Donchian Channels, price channel breakout.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the Donchian Channels.
                Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the channels.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Donchian Channels (upper, middle, lower).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates Donchian Channels
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the channels using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_donchian_channels().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Lower Channel |   Middle Channel |   Upper Channel |
        |:-----------|----------------:|-----------------:|----------------:|
        | 2025-12-17 |         264.6   |          276.218 |         287.837 |
        | 2025-12-18 |         264.779 |          276.308 |         287.837 |
        | 2025-12-19 |         264.949 |          276.393 |         287.837 |
        | 2025-12-22 |         264.949 |          276.393 |         287.837 |
        | 2025-12-23 |         266.225 |          277.031 |         287.837 |
        | 2025-12-24 |         266.225 |          277.031 |         287.837 |
        | 2025-12-26 |         266.225 |          277.031 |         287.837 |
        | 2025-12-29 |         266.225 |          277.031 |         287.837 |
        | 2025-12-30 |         266.225 |          277.031 |         287.837 |
        | 2025-12-31 |         266.225 |          277.031 |         287.837 |
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

        donchian_channels_dict = {}

        for ticker in historical_data[close_column].columns:
            donchian_channels_dict[ticker] = volatility_model.get_donchian_channels(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                window,
            ).loc[self._start_date : self._end_date]

        donchian_channels = (
            pd.concat(donchian_channels_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

        return finalize_dataset(
            dataset=donchian_channels,
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
    def get_volatility_cone(
        self,
        windows: list[int] | None = None,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Retrieves the Volatility Cone for the specified period and assets.

        The Volatility Cone summarizes the distribution of historical annualized realized
        volatility over a range of rolling windows, showing how the current realized
        volatility for each window compares to its own historical range. It is commonly
        used to judge whether current (or implied) volatility is cheap or expensive
        relative to history.

        Realized volatility is the standard deviation of the log returns over each rolling
        window, annualized with the square root of the observations per year of the period
        (252 for daily, 52 for weekly and 12 for monthly data). For each
        window, the cone reports the minimum, the 10th, 25th, 50th, 75th and 90th percentiles,
        the maximum and the most recent value of that realized volatility series.

        Options traders compare implied volatility with the cone at the matching horizon:
        implied volatility near the top of the historical range for that window suggests
        options are expensive relative to history, near the bottom cheap. The percentiles
        depend on the sample, so a period that includes a crash widens the cone for as long
        as it stays in the data; five years of daily data or more gives a more stable picture
        than one.

        Also known as: volatility cone, realized volatility term structure.

        Args:
            windows (list[int] | None, optional): The rolling windows (in periods) to
                calculate realized volatility for. Defaults to [10, 20, 30, 60, 90, 120].
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                If None, the rounding value specified during the initialization of the Toolkit instance will be used.
                Defaults to None.

        Returns:
            pd.DataFrame: The Volatility Cone for each asset, indexed by rolling window.

        Raises:
            ValueError: If the specified `period` is not one of the valid options.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Volatility Cone for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        volatility_cone = toolkit.technicals.get_volatility_cone()

        volatility_cone.xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        |   Window |   10th Percentile |   25th Percentile |   75th Percentile |   90th Percentile |   Current |    Max |   Median |    Min |
        |---------:|------------------:|------------------:|------------------:|------------------:|----------:|-------:|---------:|-------:|
        |       10 |            0.1347 |            0.1716 |            0.3027 |            0.3905 |    0.0927 | 1.0972 |   0.2269 | 0.0579 |
        |       20 |            0.1503 |            0.1903 |            0.3014 |            0.3693 |    0.0978 | 0.8034 |   0.2406 | 0.0941 |
        |       30 |            0.1654 |            0.196  |            0.2976 |            0.3682 |    0.1307 | 0.6883 |   0.2424 | 0.1253 |
        |       60 |            0.1801 |            0.2038 |            0.3005 |            0.4023 |    0.1784 | 0.5229 |   0.2371 | 0.1384 |
        |       90 |            0.1905 |            0.2039 |            0.3279 |            0.3805 |    0.2012 | 0.459  |   0.241  | 0.1688 |
        |      120 |            0.1951 |            0.2068 |            0.3528 |            0.3783 |    0.2142 | 0.4089 |   0.2483 | 0.1732 |
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

        volatility_cone = {}

        # The volatility is annualized with the observations per year of the period;
        # intraday that is the trading days times the bars per day.
        if period == "intraday":
            index = historical_data.index
            days = (
                index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
            ).normalize()
            periods_per_year = (
                volatility_model.TRADING_DAYS_PER_YEAR
                * len(index)
                / max(days.nunique(), 1)
            )
        else:
            periods_per_year = PERIODS_PER_YEAR[period]

        for ticker in historical_data[close_column].columns:
            volatility_cone[ticker] = volatility_model.get_volatility_cone(
                historical_data[close_column][ticker],
                windows=windows,
                periods_per_year=periods_per_year,
            )

        volatility_cone_df = (
            pd.concat(volatility_cone, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

        return apply_rounding(
            volatility_cone_df, rounding if rounding is not None else self._rounding
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the TRIN (Arms Index) for a given price series.

        TRIN compares the ratio of advancing to declining issues against the ratio of
        volume in advancing issues to volume in declining issues. It is a market-wide
        breadth reading computed across all tickers in the Toolkit instance (excluding
        the synthetic "Portfolio" and "Benchmark" columns), and the resulting single
        reading is broadcast to every ticker column so it lines up with the other
        breadth indicators.

        The formula is a follows:

        - TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)

        A TRIN below 1 means the volume in rising stocks outweighs their number, which is read
        as buying pressure, and above 1 as selling pressure; readings far above 2 tend to come
        with capitulation days. It is meant for a broad universe, such as all stocks of an
        exchange. Calculated across a handful of tickers it jumps between extremes and is
        undefined on days when none of them fell.

        Also known as: Arms Index, TRIN.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
            tickers=["AAPL", "MSFT"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_trin()
        ```

        Which returns:

        | Date       |     AAPL |     MSFT |   Benchmark |
        |:-----------|---------:|---------:|------------:|
        | 2025-12-17 | nan      | nan      |    nan      |
        | 2025-12-18 | nan      | nan      |    nan      |
        | 2025-12-19 | nan      | nan      |    nan      |
        | 2025-12-22 | nan      | nan      |    nan      |
        | 2025-12-23 | nan      | nan      |    nan      |
        | 2025-12-24 | nan      | nan      |    nan      |
        | 2025-12-26 | nan      | nan      |    nan      |
        | 2025-12-29 |   0.4593 |   0.4593 |      0.4593 |
        | 2025-12-30 |   1.5877 |   1.5877 |      1.5877 |
        | 2025-12-31 | nan      | nan      |    nan      |
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

        trin_series = breadth_model.get_trin(
            historical_data[close_column][constituents],
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the New Highs — New Lows for a given price series.

        New Highs — New Lows measures the number of tickers reaching a new high over
        the specified window minus the number of tickers reaching a new low over the
        same window. It is a market-wide breadth reading computed across all tickers in
        the Toolkit instance (excluding the synthetic "Portfolio" and "Benchmark"
        columns), and the resulting single reading is broadcast to every ticker column
        so it lines up with the other breadth indicators.

        The formula is a follows:

        - New Highs — New Lows = (Number of tickers at a window-period high) — (Number of tickers at a window-period low)

        The reading counts tickers at a high or low of the lookback window, so it needs a broad
        universe and a history of at least one window before the start of the period of interest:
        in the first periods every ticker is compared with only a few prices, so it is often counted
        at a high or a low, and on the first date at both. Persistently more new lows than new highs
        while an index still rises is a classic sign of narrowing breadth.

        Also known as: new highs minus new lows, record high percent.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the new high / new low lookback.
                Defaults to 252.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: New Highs — New Lows values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the New Highs — New Lows across all tickers in the Toolkit instance.
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

        toolkit.technicals.get_new_highs_new_lows()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-17 |      0 |      0 |           0 |
        | 2025-12-18 |      0 |      0 |           0 |
        | 2025-12-19 |      0 |      0 |           0 |
        | 2025-12-22 |      0 |      0 |           0 |
        | 2025-12-23 |      0 |      0 |           0 |
        | 2025-12-24 |      0 |      0 |           0 |
        | 2025-12-26 |      0 |      0 |           0 |
        | 2025-12-29 |      0 |      0 |           0 |
        | 2025-12-30 |      0 |      0 |           0 |
        | 2025-12-31 |      0 |      0 |           0 |
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

        new_highs_new_lows_series = breadth_model.get_new_highs_new_lows(
            historical_data[close_column][constituents],
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
