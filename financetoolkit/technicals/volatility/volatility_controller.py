"""Volatility Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.technicals.helpers import (
    calculate_per_ticker,
    get_observations_per_year,
)
from financetoolkit.technicals.volatility import volatility_model
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
class VolatilityTechnicals:
    """
    Volatility indicators: the Bollinger Bands, %B and BandWidth, the true range, average true range and normalized average true range, the average day range, Chaikin's volatility, the relative volatility index, the mass index, the Supertrend, the Chandelier Exit, the Chande Kroll Stop, the Profit Maximizer, the Keltner and Donchian channels, the acceleration bands, the adaptive price zone, the aberration bands, the price distance, Elder's market thermometer, the squeeze momentum and the volatility cone.
    """

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
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects various volatility indicators based on the provided data.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            window (int, optional): The window size for calculating indicators.
                Defaults to 14. The Bollinger Bands and their %B and BandWidth keep their own
                window of 20 periods (with 2 standard deviations), as defined by Bollinger, and
                the Supertrend, Chaikin Volatility, acceleration bands, Chande Kroll Stop,
                adaptive price zone, aberration bands, Profit Maximizer, market thermometer and
                squeeze momentum their own defaults.
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

        | Date       |   Bollinger Band Upper |   Bollinger Band Middle |   Bollinger Band Lower |   Bollinger %B |   Bollinger BandWidth |   True Range |   Average True Range |   Mass Index |   Supertrend |   Supertrend Trend Direction |   Chandelier Exit Long |   Chandelier Exit Short |   Keltner Channel Upper |   Keltner Channel Middle |   Keltner Channel Lower |   Donchian Channel Upper |   Donchian Channel Middle |   Donchian Channel Lower |   Normalized Average True Range |   Average Day Range |   Chaikin Volatility |   Relative Volatility Index |   Acceleration Band Upper |   Acceleration Band Middle |   Acceleration Band Lower |   Chande Kroll High Stop |   Chande Kroll Low Stop |   Adaptive Price Zone Upper |   Adaptive Price Zone Middle |   Adaptive Price Zone Lower |   Aberration Upper |   Aberration Middle |   Aberration Lower |    PMax |   PMax Trend Direction |   Price Distance |   Market Thermometer |   Market Thermometer Average |   Squeeze Momentum |   Squeeze |
        |:-----------|-----------------------:|------------------------:|-----------------------:|---------------:|----------------------:|-------------:|---------------------:|-------------:|-------------:|-----------------------------:|-----------------------:|------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|-------------------------:|--------------------------------:|--------------------:|---------------------:|----------------------------:|--------------------------:|---------------------------:|--------------------------:|-------------------------:|------------------------:|----------------------------:|-----------------------------:|----------------------------:|-------------------:|--------------------:|-------------------:|--------:|-----------------------:|-----------------:|---------------------:|-----------------------------:|-------------------:|----------:|
        | 2025-12-17 |                285.548 |                 276     |                266.452 |         0.2327 |                0.0692 |       4.5043 |               4.871  |      24.5798 |      268.465 |                            1 |                272.67  |                 279.347 |                 284.794 |                  275.314 |                 265.833 |                  287.618 |                   279.232 |                  270.846 |                          0.018  |              4.5036 |              -0.1546 |                     49.6227 |                   288.841 |                    276     |                   263.609 |                  282.999 |                 270.062 |                     284.262 |                      275.342 |                     266.422 |            279.563 |             274.669 |            269.775 | 263.457 |                      1 |           6.2482 |               0.6577 |                       1.7931 |             0.179  |         1 |
        | 2025-12-18 |                285.211 |                 276.181 |                267.151 |         0.2267 |                0.0654 |       6.6568 |               4.9986 |      24.6224 |      268.465 |                            1 |                272.442 |                 279.575 |                 284.635 |                  274.771 |                 264.907 |                  287.618 |                   279.158 |                  270.697 |                          0.0184 |              4.7648 |              -0.1054 |                     52.2511 |                   288.909 |                    276.181 |                   263.684 |                  282.999 |                 270.955 |                     283.909 |                      274.378 |                     264.847 |            278.441 |             273.429 |            268.418 | 263.457 |                      1 |          13.6624 |               4.6738 |                       2.0436 |            -1.5338 |         1 |
        | 2025-12-19 |                284.285 |                 276.55  |                268.816 |         0.2524 |                0.0559 |       4.6837 |               4.9761 |      24.6659 |      268.465 |                            1 |                272.493 |                 279.704 |                 284.312 |                  274.498 |                 264.683 |                  287.618 |                   276.821 |                  266.023 |                          0.0182 |              4.5812 |              -0.036  |                     54.7332 |                   288.375 |                    276.55  |                   264.349 |                  282.999 |                 270.93  |                     283.524 |                      273.768 |                     264.013 |            277.346 |             272.356 |            267.367 | 263.457 |                      1 |           7.8924 |               0.9667 |                       1.95   |            -3.209  |         1 |
        | 2025-12-22 |                284.343 |                 276.525 |                268.707 |         0.0846 |                0.0565 |       3.3583 |               4.8605 |      24.609  |      268.465 |                            1 |                272.722 |                 279.644 |                 283.406 |                  273.902 |                 264.397 |                  287.618 |                   276.821 |                  266.023 |                          0.018  |              4.4815 |              -0.0369 |                     51.5807 |                   287.967 |                    276.525 |                   265.009 |                  282.999 |                 270.776 |                     282.464 |                      273.024 |                     263.583 |            276.457 |             271.577 |            266.696 | 263.457 |                      1 |           5.6404 |               0      |                       1.7804 |            -5.0602 |         1 |
        | 2025-12-23 |                284.455 |                 276.347 |                268.24  |         0.1958 |                0.0587 |       2.9298 |               4.7226 |      24.5236 |      268.465 |                            1 |                273     |                 279.366 |                 282.71  |                  273.57  |                 264.43  |                  287.618 |                   276.821 |                  266.023 |                          0.0174 |              4.3121 |              -0.0391 |                     54.401  |                   287.422 |                    276.347 |                   265.252 |                  282.999 |                 270.593 |                     281.343 |                      272.509 |                     263.674 |            275.83  |             271.08  |            266.329 | 263.457 |                      1 |           4.4745 |               0.9467 |                       1.7079 |            -6.1264 |         1 |
        | 2025-12-24 |                284.438 |                 276.19  |                267.941 |         0.2981 |                0.0597 |       3.2188 |               4.6152 |      24.4118 |      268.465 |                            1 |                273.225 |                 280.416 |                 282.345 |                  273.476 |                 264.606 |                  283.741 |                   274.882 |                  266.023 |                          0.0169 |              4.105  |              -0.037  |                     56.7182 |                   286.983 |                    276.19  |                   265.287 |                  282.999 |                 270.458 |                     280.619 |                      272.348 |                     264.077 |            275.848 |             271.199 |            266.551 | 263.457 |                      1 |           4.9926 |               2.9198 |                       1.8133 |            -6.7609 |         1 |
        | 2025-12-26 |                284.387 |                 275.983 |                267.579 |         0.2899 |                0.0609 |       2.5013 |               4.4642 |      24.2185 |      268.465 |                            1 |                273.538 |                 280.103 |                 281.822 |                  273.339 |                 264.856 |                  280.164 |                   273.094 |                  266.023 |                          0.0164 |              4.0637 |              -0.1593 |                     54.2874 |                   286.737 |                    275.983 |                   265.137 |                  282.851 |                 270.265 |                     279.904 |                      272.297 |                     264.69  |            276.293 |             271.788 |            267.282 | 263.457 |                      1 |           4.594  |               0      |                       1.6556 |            -7.2886 |         1 |
        | 2025-12-29 |                284.195 |                 275.729 |                267.264 |         0.3276 |                0.0614 |       2.003  |               4.2884 |      23.8779 |      268.465 |                            1 |                273.905 |                 279.736 |                 281.304 |                  273.268 |                 265.233 |                  279.177 |                   272.6   |                  266.023 |                          0.0157 |              3.9562 |              -0.1596 |                     55.4689 |                   286.405 |                    275.729 |                   265.055 |                  282.851 |                 270.041 |                     279.204 |                      272.345 |                     265.486 |            276.279 |             271.941 |            267.602 | 263.457 |                      1 |           3.6472 |               0.5082 |                       1.5559 |            -7.4802 |         1 |
        | 2025-12-30 |                283.299 |                 275.23  |                267.162 |         0.308  |                0.0586 |       1.7937 |               4.1102 |      23.5003 |      268.465 |                            1 |                274.284 |                 279.357 |                 280.707 |                  273.117 |                 265.526 |                  279.177 |                   272.6   |                  266.023 |                          0.0151 |              3.863  |              -0.3267 |                     54.0398 |                   285.385 |                    275.23  |                   265.399 |                  279.001 |                 269.818 |                     278.457 |                      272.336 |                     266.215 |            276.381 |             272.212 |            268.043 | 263.457 |                      1 |           4.2652 |               0.0698 |                       1.4266 |            -7.2914 |         0 |
        | 2025-12-31 |                281.366 |                 274.516 |                267.666 |         0.2372 |                0.0499 |       1.9233 |               3.954  |      23.1557 |      268.465 |                            1 |                274.628 |                 279.013 |                 280.039 |                  272.823 |                 265.607 |                  279.177 |                   272.6   |                  266.023 |                          0.0146 |              3.7647 |              -0.3494 |                     52.7332 |                   284.415 |                    274.516 |                   265.137 |                  275.232 |                 269.631 |                     277.662 |                      272.143 |                     266.623 |            276.421 |             272.402 |            268.383 | 263.457 |                      1 |           2.6707 |               0.5282 |                       1.3485 |            -6.9011 |         1 |
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

        # Bollinger's own 20-period average with 2 standard deviations, whatever the window of the collection: the bands, and the %B and BandWidth derived from them, are defined on that window and a 14-period band is a different, narrower channel.  # noqa: E501
        bollinger_bands = self.get_bollinger_bands(
            period=period, close_column=close_column
        )

        volatility_indicators["Bollinger Band Upper"] = bollinger_bands["Upper Band"]
        volatility_indicators["Bollinger Band Middle"] = bollinger_bands["Middle Band"]
        volatility_indicators["Bollinger Band Lower"] = bollinger_bands["Lower Band"]

        percent_b_and_bandwidth = self.get_bollinger_percent_b_and_bandwidth(
            period=period, close_column=close_column
        )

        volatility_indicators["Bollinger %B"] = percent_b_and_bandwidth["%B"]
        volatility_indicators["Bollinger BandWidth"] = percent_b_and_bandwidth[
            "BandWidth"
        ]

        volatility_indicators["True Range"] = self.get_true_range(
            period=period, close_column=close_column
        )

        volatility_indicators["Average True Range"] = self.get_average_true_range(
            period=period, close_column=close_column, window=window
        )

        volatility_indicators["Mass Index"] = self.get_mass_index(
            period=period, close_column=close_column
        )

        supertrend = self.get_supertrend(period=period, close_column=close_column)

        volatility_indicators["Supertrend"] = supertrend["Supertrend"]
        volatility_indicators["Supertrend Trend Direction"] = supertrend[
            "Trend Direction"
        ]

        chandelier_exit = self.get_chandelier_exit(
            period=period, close_column=close_column
        )

        volatility_indicators["Chandelier Exit Long"] = chandelier_exit["Long Exit"]
        volatility_indicators["Chandelier Exit Short"] = chandelier_exit["Short Exit"]

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

        volatility_indicators["Normalized Average True Range"] = (
            self.get_normalized_average_true_range(
                period=period, close_column=close_column, window=window
            )
        )
        volatility_indicators["Average Day Range"] = self.get_average_day_range(
            period=period, close_column=close_column, window=window
        )
        volatility_indicators["Chaikin Volatility"] = self.get_chaikin_volatility(
            period=period, close_column=close_column
        )
        volatility_indicators["Relative Volatility Index"] = (
            self.get_relative_volatility_index(
                period=period, close_column=close_column, window=window
            )
        )

        acceleration_bands = self.get_acceleration_bands(
            period=period, close_column=close_column
        )
        for band in ["Upper", "Middle", "Lower"]:
            volatility_indicators[f"Acceleration Band {band}"] = acceleration_bands[
                f"{band} Band"
            ]

        chande_kroll_stop = self.get_chande_kroll_stop(
            period=period, close_column=close_column
        )
        volatility_indicators["Chande Kroll High Stop"] = chande_kroll_stop["High Stop"]
        volatility_indicators["Chande Kroll Low Stop"] = chande_kroll_stop["Low Stop"]

        adaptive_price_zone = self.get_adaptive_price_zone(
            period=period, close_column=close_column
        )
        for band in ["Upper", "Middle", "Lower"]:
            volatility_indicators[f"Adaptive Price Zone {band}"] = adaptive_price_zone[
                f"{band} Band"
            ]

        aberration = self.get_aberration(period=period, close_column=close_column)
        for band in ["Upper", "Middle", "Lower"]:
            volatility_indicators[f"Aberration {band}"] = aberration[f"{band} Band"]

        profit_maximizer = self.get_profit_maximizer(
            period=period, close_column=close_column
        )
        volatility_indicators["PMax"] = profit_maximizer["PMax"]
        volatility_indicators["PMax Trend Direction"] = profit_maximizer[
            "Trend Direction"
        ]

        volatility_indicators["Price Distance"] = self.get_price_distance(
            period=period, close_column=close_column
        )

        market_thermometer = self.get_market_thermometer(
            period=period, close_column=close_column
        )
        volatility_indicators["Market Thermometer"] = market_thermometer["Thermometer"]
        volatility_indicators["Market Thermometer Average"] = market_thermometer[
            "Thermometer Average"
        ]

        squeeze_momentum = self.get_squeeze_momentum(
            period=period, close_column=close_column
        )
        volatility_indicators["Squeeze Momentum"] = squeeze_momentum["Momentum"]
        volatility_indicators["Squeeze"] = squeeze_momentum["Squeeze"]

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
    def get_bollinger_bands(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        num_std_dev: float = 2,
        moving_average: str = "simple",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Bollinger Bands for a given price series.

        Bollinger Bands are a volatility indicator that consists of three lines: an upper band,
        a middle band (simple moving average), and a lower band. The upper and lower bands are
        calculated as the moving average plus and minus a specified number of standard deviations,
        respectively.

        The formula is as follows:

        - Middle Band = SMA(Close, Window), or another moving average with `moving_average`
        - Upper Band = Middle Band + (Num Std Dev * Std Dev)
        - Lower Band = Middle Band — (Num Std Dev * Std Dev)

        The standard deviation is the *population* standard deviation (dividing by n), as
        Bollinger himself specifies and as TA-Lib and StockCharts both define it, not
        pandas' default sample standard deviation.

        The band width tracks volatility: bands that narrow sharply (a squeeze) often precede a
        larger move, without saying in which direction. A close at or beyond a band is not a sell
        or buy signal by itself, since prices can walk along a band in a trend. The defaults are
        Bollinger's own setting, a 20-period average with 2 standard deviations, as on most
        charting platforms.

        For more information, see the following sources:

        - Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill.
        - Referenced against TA-Lib's BBANDS (https://ta-lib.org/functions/bbands.html).
        - pandas-ta-classic: bbands (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: Bollinger Bands, BB, volatility bands, price channels.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for moving average calculation.
                The number of periods (time intervals) over which to calculate the moving average. Defaults to 20.
            num_std_dev (float, optional): Number of standard deviations for the bands.
                Defaults to 2.
            moving_average (str, optional): The moving average of the middle band, one of the methods
                of `get_moving_average`: "simple", "exponential", "double_exponential",
                "triple_exponential", "zero_lag", "wilder", "triangular", "weighted", "hull", "sine",
                "fibonacci" or "pascal". Defaults to "simple", Bollinger's own.
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
            pd.Series or pd.DataFrame:
            Bollinger Bands (upper, middle, lower).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Bollinger Bands for each asset in the Toolkit instance.
        - The bands equal TA-Lib's BBANDS with a simple moving average (`matype=0`), and with the
          exponential, weighted, double exponential, triple exponential and triangular averages its
          `matype` 1 to 5. The standard deviation stays the one around the simple average of the
          window, as in TA-Lib, whatever the middle band.
        - If `growth` is set to True, the method calculates the growth of the Bollinger Bands
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

        toolkit.technicals.get_bollinger_bands().xs("USDJPY", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Close |   Lower Band |   Middle Band |   Upper Band |
        |:-----------|--------:|-------------:|--------------:|-------------:|
        | 2025-12-26 | 156.5   |      154.463 |       155.975 |      157.486 |
        | 2025-12-28 | 156.324 |      154.582 |       156.038 |      157.495 |
        | 2025-12-29 | 156.031 |      154.652 |       156.073 |      157.494 |
        | 2025-12-30 | 156.329 |      154.744 |       156.124 |      157.504 |
        | 2025-12-31 | 156.852 |      154.759 |       156.171 |      157.582 |
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

        bollinger_bands = calculate_per_ticker(
            historical_data,
            [close_column],
            volatility_model.get_bollinger_bands,
            window,
            num_std_dev,
            moving_average,
        ).loc[self._start_date : self._end_date]

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
    def get_bollinger_percent_b_and_bandwidth(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        num_std_dev: float = 2,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Bollinger %B and BandWidth for a given price series.

        Bollinger (2001, "Bollinger on Bollinger Bands") derived two indicators from his bands.
        %B places the close within the bands: 0 is on the lower band, 0.5 on the middle band and
        1 on the upper band, with values below 0 or above 1 for a close outside the bands.
        BandWidth is the distance between the bands relative to the middle band, a measure of
        volatility that can be compared across securities and over time.

        The formula is as follows:

        - Middle Band = SMA(Close, window)
        - Upper Band = Middle Band + (num_std_dev * Std Dev), Lower Band = Middle Band — (num_std_dev * Std Dev)
        - %B = (Close — Lower Band) / (Upper Band — Lower Band)
        - BandWidth = (Upper Band — Lower Band) / Middle Band

        The bands are those of get_bollinger_bands, with the *population* standard deviation
        (dividing by n) that Bollinger specifies and TA-Lib and StockCharts implement, and the
        defaults are Bollinger's own 20 periods and 2 standard deviations. Both results are
        decimals: a BandWidth of 0.10 means bands 10% of the middle band apart (StockCharts
        multiplies it by 100). A BandWidth at a six-month low marks what Bollinger calls the
        Squeeze, which often precedes a larger move without saying in which direction; %B above
        1 or below 0 shows a close outside the bands, which in a trend can last for weeks.

        For more information, see the following sources:

        - Referenced against TA-Lib's PERCENTB and BBW (https://ta-lib.org/functions/percentb.html,
          https://ta-lib.org/functions/bbw.html).

        Also known as: Percent B, %b, Bollinger Band Width, BBW, Bollinger Squeeze.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods of the moving average and standard deviation.
                Defaults to 20.
            num_std_dev (float, optional): Number of standard deviations for the bands.
                Defaults to 2.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of %B and BandWidth.
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
            pd.Series or pd.DataFrame: %B and BandWidth values, both decimals.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          %B and BandWidth for each asset in the Toolkit instance.
        - A window of identical prices gives bands without width, which leaves %B missing and
          BandWidth at 0.
        - With a simple moving average (`matype=0`), %B equals TA-Lib's PERCENTB and BandWidth
          equals TA-Lib's BBW divided by 100, since TA-Lib's BBW is in percent.
        - If `growth` is set to True, the method calculates the growth of %B and BandWidth
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_bollinger_percent_b_and_bandwidth().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |     %B |   BandWidth |
        |:-----------|-------:|------------:|
        | 2025-12-24 | 0.2981 |      0.0597 |
        | 2025-12-26 | 0.2899 |      0.0609 |
        | 2025-12-29 | 0.3276 |      0.0614 |
        | 2025-12-30 | 0.308  |      0.0586 |
        | 2025-12-31 | 0.2372 |      0.0499 |
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

        percent_b_and_bandwidth = calculate_per_ticker(
            historical_data,
            [close_column],
            volatility_model.get_bollinger_percent_b_and_bandwidth,
            window,
            num_std_dev,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=percent_b_and_bandwidth,
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
    def get_true_range(
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
        Calculate the True Range (TR) for a given price series.

        The True Range (TR) is a measure of market volatility that considers the differences
        between the high and low prices and the previous closing price. It provides insights
        into the price movement of an asset.

        The formula is as follows:

        - TR = max(high — low, abs(high — previous_close), abs(low — previous_close))

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - Referenced against TA-Lib's TRANGE (https://ta-lib.org/functions/trange.html).

        Also known as: TR, true range.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the True Range.
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
            pd.Series or pd.DataFrame: True Range values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          True Range for each asset in the Toolkit instance.
        - The True Range equals TA-Lib's TRANGE from the second period on. The first period, which
          has no previous close, is High — Low here and empty in TA-Lib.
        - If `growth` is set to True, the method calculates the growth of the True Range
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

        toolkit.technicals.get_true_range().tail(5)
        ```

        Which returns:

        | Date       |   NGUSD |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-26 |   0.52  |      2.3715 |
        | 2025-12-28 |   0.475 |    nan      |
        | 2025-12-29 |   0.242 |      4.2071 |
        | 2025-12-30 |   0.26  |      1.9647 |
        | 2025-12-31 |   0.304 |      5.6062 |
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

        true_range = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_true_range,
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
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Average True Range (ATR) of a given price series.

        The Average True Range (ATR) is a technical indicator that measures the volatility
        of an asset's price movements over a specified number of periods. It provides insights
        into the potential price range of an asset, which can help traders and investors make
        more informed decisions.

        The formula is as follows:

        - TR = max(high — low, abs(high — previous_close), abs(low — previous_close))
        - ATR = Wilder's Smoothed Moving Average of TR over `window` periods

        The ATR is in price units, which makes it the common basis for stop distances (for example
        two ATRs below the entry) and for sizing positions to equal risk. For the same reason it
        cannot be compared across stocks or over long periods in which the price changed a lot;
        divide it by the price for a percentage volatility. It measures how far price moves, not in
        which direction.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - Referenced against TA-Lib's ATR (https://ta-lib.org/functions/atr.html).

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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: ATR values per ticker or ATR growth rate (if growth is True).
                A pandas Series containing the calculated Average True Range values or growth rate for each period.

        Notes:
        - ATR values are typically used to assess the volatility and potential price movement of an asset.
        - A higher ATR value indicates higher volatility, while a lower ATR value suggests lower volatility.
        - Because the True Range starts on the first period, the ATR starts one period before
          TA-Lib's ATR and differs from it until the two converge (within 0.01% after about 80
          periods with the default window).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["CLUSD", "NGUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_average_true_range().tail(5)
        ```

        Which returns:

        | Date       |   CLUSD |   NGUSD |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-26 |  1.2031 |  0.304  |      6.4283 |
        | 2025-12-28 |  1.1614 |  0.3162 |    nan      |
        | 2025-12-29 |  1.1778 |  0.3109 |      6.2696 |
        | 2025-12-30 |  1.1558 |  0.3073 |      5.9621 |
        | 2025-12-31 |  1.1697 |  0.3071 |      5.9367 |
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

        average_true_range = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_average_true_range,
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
    def get_mass_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        ema_window: int = 9,
        sum_window: int = 25,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Mass Index for a given price series.

        The Mass Index, developed by Donald Dorsey, looks for reversals in the width of the trading
        range rather than in price. It divides an EMA of the High — Low range by an EMA of that EMA,
        a ratio above 1 while the range widens and below 1 while it narrows, and sums the ratio over
        `sum_window` periods.

        The formula is as follows:

        - Single EMA = EMA(High — Low, ema_window)
        - Double EMA = EMA(Single EMA, ema_window)
        - Mass Index = Sum(Single EMA / Double EMA, sum_window)

        With the default windows a range that neither widens nor narrows gives a Mass Index of 25.
        Dorsey's "reversal bulge" is the index rising above 27 and then falling back below 26.5,
        after which the current trend is expected to reverse. The index does not say in which
        direction, so it is read together with a trend indicator such as a moving average. The ratio
        has no unit, so the level can be compared across stocks.

        For more information, see the following sources:

        - Dorsey, D. (1992). "The Mass Index." Technical Analysis of Stocks & Commodities, 10(6), 265-267.
        - Referenced against TA-Lib's MASSI (https://ta-lib.org/functions/massi.html).

        Also known as: Mass Index, MI. See `volatility_model.get_mass_index` for the references.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The close the high and low are taken on the basis of.
                Defaults to "Adj Close", in which case the high and low prices are adjusted
                for dividends in the same way.
            ema_window (int, optional): The number of periods of the single and double EMA.
                Defaults to 9, as defined by Dorsey.
            sum_window (int, optional): The number of periods over which the EMA ratio is summed.
                Defaults to 25, as defined by Dorsey.
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
            pd.Series or pd.DataFrame: Mass Index values.

        Notes:
        - Both EMAs are seeded with the simple average of their first values, as in TA-Lib and on
          StockCharts, so the first Mass Index is on period 2 × (`ema_window` — 1) + `sum_window`,
          the 41st with the default windows. The Mass Index equals TA-Lib's MASSI.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AMZN", "GOOGL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_mass_index().tail(5)
        ```

        Which returns:

        | Date       |    AMZN |   GOOGL |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 23.5078 | 24.9252 |     24.0591 |
        | 2025-12-26 | 23.285  | 24.4336 |     23.7267 |
        | 2025-12-29 | 22.9215 | 23.8576 |     23.1418 |
        | 2025-12-30 | 22.6104 | 23.4152 |     22.5802 |
        | 2025-12-31 | 22.4079 | 23.0341 |     22.2524 |
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

        mass_index = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            volatility_model.get_mass_index,
            ema_window,
            sum_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=mass_index,
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
        show_columns: str | list[str] | None = None,
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

        The formula is as follows:

        - Basic Upper Band = (High + Low) / 2 + multiplier * ATR(window)
        - Basic Lower Band = (High + Low) / 2 — multiplier * ATR(window)
        - Final Upper Band(t) = Basic Upper Band(t) if Basic Upper Band(t) < Final Upper
          Band(t-1) or Close(t-1) > Final Upper Band(t-1), else Final Upper Band(t-1)
        - Final Lower Band(t) = Basic Lower Band(t) if Basic Lower Band(t) > Final Lower
          Band(t-1) or Close(t-1) < Final Lower Band(t-1), else Final Lower Band(t-1)
        - While in an uptrend, Supertrend = Final Lower Band, until Close crosses below it,
          at which point the trend flips to a downtrend and Supertrend = Final Upper Band
          (and vice versa)

        For more information, see the following sources:

        - Referenced against TA-Lib's SUPERTREND (https://ta-lib.org/functions/supertrend.html).

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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Supertrend and
                Trend Direction values. Defaults to False.
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
        - The ATR starts one period before TA-Lib's (see `get_average_true_range`), so the
          Supertrend starts one period before TA-Lib's SUPERTREND and differs from it until the two
          converge (within 0.01% after about 40 periods with the defaults). The Trend Direction
          equals TA-Lib's `trend` from TA-Lib's first value.
        - If `growth` is set to True, the method calculates the growth of the Supertrend and
          Trend Direction values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SHOP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_supertrend().xs("SHOP", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Supertrend |   Trend Direction |
        |:-----------|-------------:|------------------:|
        | 2025-12-24 |      150.646 |                 1 |
        | 2025-12-26 |      151.942 |                 1 |
        | 2025-12-29 |      152.113 |                 1 |
        | 2025-12-30 |      152.113 |                 1 |
        | 2025-12-31 |      152.113 |                 1 |
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

        supertrend = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_supertrend,
            window,
            multiplier,
        ).loc[self._start_date : self._end_date]

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
    def get_chandelier_exit(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 22,
        multiplier: float = 3.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chandelier Exit for a given price series.

        The Chandelier Exit, developed by Chuck Le Beau, is a trailing stop set a multiple of the
        Average True Range away from the most extreme price of the last `window` periods: the long
        exit "hangs" below the highest high like a chandelier from a ceiling, and the short exit sits
        above the lowest low.

        The formula is as follows:

        - Long Exit = Highest High(window) — multiplier × ATR(window)
        - Short Exit = Lowest Low(window) + multiplier × ATR(window)

        A long position is closed when the price closes below the long exit, and a short position
        when it closes above the short exit. Because the distance is in ATRs, the stop is wider in
        volatile markets and tighter in quiet ones. The stop moves with both the highest high and
        the ATR, so it can fall while the high holds when volatility rises. The exits are in price
        units, so they are compared with the stock's own price rather than across stocks.

        Also known as: Chandelier Exit, Chandelier Stop, CE.

        For more information, see the following sources:

        - Le Beau, C. & Lucas, D.W. (1992). "Technical Traders Guide to Computer Analysis of the Futures
          Market." Business One Irwin.
        - Elder, A. (2002). "Come Into My Trading Room." John Wiley & Sons, which popularised the exit.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the highest high, the lowest low and the
                Average True Range. Defaults to 22, about a month of trading days, as defined by Le Beau.
            multiplier (float, optional): The number of Average True Ranges the exits sit from the
                highest high and the lowest low. Defaults to 3.0, as defined by Le Beau.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Long Exit and Short Exit.
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
            pd.Series or pd.DataFrame: The Long Exit and Short Exit, in price units.

        Notes:
        - The Average True Range is that of `get_average_true_range`: Wilder's smoothing of the True
          Range, with the first True Range High — Low and the first ATR the average of the first
          `window` True Ranges (the StockCharts convention), so the exits start on the `window`-th
          period. The highest high and the lowest low include the current period.
        - Because the ATR is seeded as on StockCharts, the first exits differ from those built on
          TA-Lib's ATR, which starts the True Range on the second period, until the two converge
          (within 0.01% after about 80 periods with the defaults).
        - If `growth` is set to True, the method calculates the growth of the Long Exit and Short Exit
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chandelier_exit().xs("TSLA", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Long Exit |   Short Exit |
        |:-----------|------------:|-------------:|
        | 2025-12-24 |     444.423 |      455.497 |
        | 2025-12-26 |     444.814 |      459.966 |
        | 2025-12-29 |     445.062 |      470.658 |
        | 2025-12-30 |     446.239 |      474.711 |
        | 2025-12-31 |     447.641 |      473.309 |
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

        chandelier_exit = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_chandelier_exit,
            window,
            multiplier,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=chandelier_exit,
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
        atr_multiplier: float = 2,
        method: str = "exponential",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Keltner Channels for a given price series.

        The Keltner Channels consist of three lines: a middle line, the Exponential Moving Average
        (EMA) of the closing prices, and an upper and a lower line a multiple of the Average True
        Range (ATR) above and below it.

        The formula is as follows:

        - Middle Line = EMA(Close, window)
        - Upper Line = Middle Line + atr_multiplier * ATR(atr_window)
        - Lower Line = Middle Line — atr_multiplier * ATR(atr_window), where the ATR uses Wilder's smoothing.

        With method="original" the channel is Keltner's own: a simple average of the typical price
        with the average High — Low range on either side.

        - Middle Line = SMA((High + Low + Close) / 3, window)
        - Upper Line = Middle Line + atr_multiplier * SMA(High — Low, window)
        - Lower Line = Middle Line — atr_multiplier * SMA(High — Low, window)

        For more information, see the following sources:

        - Keltner, C.W. (1960). "How to Make Money in Commodities." The Keltner Statistical Service.
        - Referenced against TA-Lib's KC (https://ta-lib.org/functions/kc.html).
        - ta: volatility.KeltnerChannel (https://github.com/bukosabino/ta), whose default
          original_version=True is the original channel.

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
            atr_multiplier (float, optional): Multiplier for ATR (or, for the original channel, the
                average range) to determine channel width. Defaults to 2; Keltner's original channel
                uses 1 (method="original", atr_multiplier=1).
            method (str, optional): "exponential" for the EMA of the close with ATR bands (StockCharts)
                or "original" for Keltner's simple average of the typical price with average range
                bands. Defaults to "exponential".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the channels.
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
            pd.Series or pd.DataFrame: Keltner Channels (upper, middle, lower).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates Keltner Channels
          for each asset in the Toolkit instance.
        - The middle line is the EMA of the close, as on StockCharts. TA-Lib's KC takes the EMA of
          the typical price ((High + Low + Close) / 3) and starts its ATR one period later, so the
          channels differ throughout (by about 0.3% on daily data).
        - The original channel uses `window` for both averages (Keltner used 10 days with one
          average range, atr_multiplier=1) and no ATR, so `atr_window` does not apply. With
          atr_multiplier=1 it equals ta's KeltnerChannel with original_version=True.
        - If `growth` is set to True, the method calculates the growth of the channels using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^GDAXI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_keltner_channels().xs("^GDAXI", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Lower Line |   Middle Line |   Upper Line |
        |:-----------|-------------:|--------------:|-------------:|
        | 2025-12-24 |        nan   |         nan   |        nan   |
        | 2025-12-26 |        nan   |         nan   |        nan   |
        | 2025-12-29 |      23670.6 |       24089.3 |      24508   |
        | 2025-12-30 |      23710.7 |       24127.5 |      24544.2 |
        | 2025-12-31 |        nan   |         nan   |        nan   |
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

        keltner_channels = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_keltner_channels,
            window,
            atr_window,
            atr_multiplier,
            method,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=keltner_channels,
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
        include_current: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Donchian Channels for a given price series.

        Donchian Channels plot the highest high and lowest low over the `window` periods
        *preceding* the current one, with the middle line being the average of the two. They
        are used to identify breakouts and the overall volatility of the price range.

        The formula is as follows:

        - Upper Channel = Highest High over Window, ending one period ago
        - Lower Channel = Lowest Low over Window, ending one period ago
        - Middle Channel = (Upper Channel + Lower Channel) / 2

        The current period is deliberately excluded from the lookback, per Donchian's
        original breakout rule and StockCharts' Price Channels definition — including it
        would make a channel break impossible by construction.

        For more information, see the following sources:

        - Donchian, R.D. (1960). "Commodities: High Finance in Copper." Financial Analysts Journal, 16(6),
          133-142.
        - Kaufman, P.J. (2013). "Trading Systems and Methods." 5th ed. Wiley.
        - Referenced against TA-Lib's DONCHIAN (https://ta-lib.org/functions/donchian.html).

        Also known as: Donchian Channels, price channel breakout.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the Donchian Channels.
                Defaults to 20.
            include_current (bool, optional): Whether the window ends on the current period instead
                of the period before it, as in TA-Lib's DONCHIAN, pandas-ta-classic's donchian and
                ta's DonchianChannel. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the channels.
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
            pd.Series or pd.DataFrame: Donchian Channels (upper, middle, lower).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates Donchian Channels
          for each asset in the Toolkit instance.
        - TA-Lib's DONCHIAN includes the current period, so each channel here equals TA-Lib's of the
          previous period; with include_current=True it equals TA-Lib's DONCHIAN.
        - If `growth` is set to True, the method calculates the growth of the channels using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["HGUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_donchian_channels().xs("HGUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Lower Channel |   Middle Channel |   Upper Channel |
        |:-----------|----------------:|-----------------:|----------------:|
        | 2025-12-26 |          5.2415 |           5.499  |          5.7565 |
        | 2025-12-28 |          5.28   |           5.584  |          5.888  |
        | 2025-12-29 |          5.28   |           5.6018 |          5.9235 |
        | 2025-12-30 |          5.28   |           5.6018 |          5.9235 |
        | 2025-12-31 |          5.28   |           5.6018 |          5.9235 |
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

        donchian_channels = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            volatility_model.get_donchian_channels,
            window,
            include_current,
        ).loc[self._start_date : self._end_date]

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
        show_columns: str | list[str] | None = None,
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
        (252 for daily data, or 365 for a ticker that also trades on weekends such as a
        cryptocurrency, 52 for weekly and 12 for monthly data). Each ticker is measured over
        the days it traded, so a stock in a Toolkit with a cryptocurrency keeps its own
        returns rather than flat weekend days. For each
        window, the cone reports the minimum, the 10th, 25th, 50th, 75th and 90th percentiles,
        the maximum and the most recent value of that realized volatility series.

        The formula is as follows:

        - Log Return (t) = ln(Close (t) / Close (t-1))
        - Realized Volatility (t, Window) = Sample Standard Deviation (ddof=1) of the Log Returns over the last
          Window periods * √(Periods per Year)
        - Cone (Window) = Min, 10th, 25th, 50th (Median), 75th and 90th Percentile, Max and Current (latest) value
          of Realized Volatility (t, Window) over the full history, for each window (10, 20, 30, 60, 90 and 120
          by default)

        Where Close is the `close_column`, the Adj Close by default (the Close for intraday data).

        Options traders compare implied volatility with the cone at the matching horizon:
        implied volatility near the top of the historical range for that window suggests
        options are expensive relative to history, near the bottom cheap. The percentiles
        depend on the sample, so a period that includes a crash widens the cone for as long
        as it stays in the data; five years of daily data or more gives a more stable picture
        than one.

        For more information, see the following sources:

        - Burghardt, G. & Lane, M. (1990). "How to Tell if Options are Cheap." Journal of Portfolio Management,
          16(2), 72-78.
        - Natenberg, S. (1994). "Option Volatility and Pricing." McGraw-Hill.

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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            tickers=["BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        volatility_cone = toolkit.technicals.get_volatility_cone().xs("BTCUSD", level=1, axis="columns")

        volatility_cone.loc[:, ["Min", "10th Percentile", "Median", "90th Percentile", "Max", "Current"]]
        ```

        Which returns:

        |   Window |    Min |   10th Percentile |   Median |   90th Percentile |    Max |   Current |
        |---------:|-------:|------------------:|---------:|------------------:|-------:|----------:|
        |       10 | 0.0875 |            0.2495 |   0.4784 |            0.8814 | 1.5987 |    0.1574 |
        |       20 | 0.1146 |            0.2978 |   0.4996 |            0.8581 | 1.3232 |    0.2658 |
        |       30 | 0.1704 |            0.317  |   0.5149 |            0.8225 | 1.215  |    0.3448 |
        |       60 | 0.2618 |            0.3441 |   0.5461 |            0.7757 | 1.0868 |    0.4155 |
        |       90 | 0.2776 |            0.3564 |   0.5356 |            0.7652 | 0.9631 |    0.4132 |
        |      120 | 0.2903 |            0.3656 |   0.5216 |            0.7495 | 0.9164 |    0.3861 |
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

        # The cone describes the volatility between the start and end date.
        historical_data = self._historical_data[period].loc[
            self._start_date : self._end_date
        ]

        # The volatility is annualized with the observations per year of the period: per
        # ticker 252 trading days, or 365 calendar days for a ticker that also trades on
        # weekends (a cryptocurrency); intraday those days times the bars per day.
        bars_per_day = 1.0
        if period == "intraday":
            index = historical_data.index
            days = (
                index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
            ).normalize()
            bars_per_day = len(index) / max(days.nunique(), 1)

        def get_volatility_cone(prices: pd.Series) -> pd.DataFrame:
            return volatility_model.get_volatility_cone(
                prices,
                windows=windows,
                periods_per_year=(
                    get_observations_per_year(prices) * bars_per_day
                    if period in ("daily", "intraday")
                    else PERIODS_PER_YEAR[period]
                ),
            )

        volatility_cone_df = calculate_per_ticker(
            historical_data, [close_column], get_volatility_cone
        )

        return apply_rounding(
            volatility_cone_df, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_normalized_average_true_range(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Normalized Average True Range (NATR) of a given price series.

        The Normalized Average True Range is the Average True Range divided by the close, so the
        average range is a share of the price. Unlike the ATR, which is in price units, it compares
        across stocks with different prices and over periods in which the price changed a lot.

        The formula is as follows:

        - NATR = ATR(window) / Close, where the ATR uses Wilder's smoothing.

        The result is a decimal: 0.02 is an average range of 2% of the close.

        For more information, see the following sources:

        - Forman, J. (2006). "Cross-Market Evaluations With Normalized Average True Range."
          Technical Analysis of Stocks & Commodities, 24(5).
        - Referenced against TA-Lib's NATR (https://ta-lib.org/functions/natr.html).
        - pandas-ta-classic: natr (https://github.com/xgboosted/pandas-ta-classic).
        - Tulip Indicators: natr (https://tulipindicators.org/natr).

        Also known as: NATR, ATR percent, ATRP.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the Average True Range.
                Defaults to 14, as in TA-Lib.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the NATR.
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
            pd.DataFrame: NATR values per ticker as decimals, or their growth (if growth is True).

        Notes:
        - The ATR is that of `get_average_true_range` (the StockCharts seeding), so the NATR starts
          one period before TA-Lib's NATR and equals it, divided by 100, once the seeds have
          converged (within 1e-8 after about 150 periods with the default window).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_normalized_average_true_range().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.0169 | 0.0169 |      0.0098 |
        | 2025-12-26 | 0.0164 | 0.016  |      0.0094 |
        | 2025-12-29 | 0.0157 | 0.0155 |      0.0092 |
        | 2025-12-30 | 0.0151 | 0.015  |      0.0087 |
        | 2025-12-31 | 0.0146 | 0.0147 |      0.0088 |
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

        normalized_average_true_range = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_normalized_average_true_range,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=normalized_average_true_range,
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
    def get_average_day_range(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Average Day Range (ADR) of a given price series.

        The Average Day Range is the simple average of the High — Low range over `window` periods.
        Unlike the Average True Range it leaves out the gaps between periods, so it measures how
        far the price moves within a period. It is in price units, which makes it a basis for
        intraday targets and stops (for example a stop half an ADR away).

        The formula is as follows:

        - ADR = SMA(High — Low, window)

        For more information, see the following sources:

        - Keltner, C.W. (1960). "How to Make Money in Commodities." The Keltner Statistical
          Service, whose channel is this average range around the typical price.
        - Referenced against TA-Lib's ADR (https://ta-lib.org/functions/adr.html).

        Also known as: ADR, average daily range, average high-low range.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the average. Defaults to 14, as in
                TA-Lib.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ADR.
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
            pd.DataFrame: ADR values per ticker in price units, or their growth (if growth is True).

        Notes:
        - Equals TA-Lib's ADR.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AMZN", "GOOGL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_average_day_range().tail(5)
        ```

        Which returns:

        | Date       |   AMZN |   GOOGL |   Benchmark |
        |:-----------|-------:|--------:|------------:|
        | 2025-12-24 | 3.7636 |  7.0625 |      5.6971 |
        | 2025-12-26 | 3.7007 |  6.9786 |      5.5972 |
        | 2025-12-29 | 3.5057 |  6.5637 |      5.4608 |
        | 2025-12-30 | 3.4421 |  6.4496 |      5.4033 |
        | 2025-12-31 | 3.3643 |  6.2008 |      5.2624 |
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

        average_day_range = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            volatility_model.get_average_day_range,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=average_day_range,
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
    def get_chaikin_volatility(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        roc_window: int = 10,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Chaikin's Volatility of a given price series.

        Chaikin's Volatility is the rate of change of an Exponential Moving Average of the High —
        Low range: how much wider or narrower the average range is than `roc_window` periods ago.
        Chaikin read a sharp rise over a short time as a sign of an approaching bottom (panic
        selling) and a long decline as typical of a maturing top.

        The formula is as follows:

        - Range Average = EMA(High — Low, window)
        - Chaikin Volatility = Range Average / Range Average `roc_window` periods ago — 1

        The result is a decimal: 0.25 is an average range 25% wider than `roc_window` periods ago.

        For more information, see the following sources:

        - Achelis, S.B. (2000). "Technical Analysis from A to Z." 2nd ed. McGraw-Hill,
          "Chaikin's Volatility".
        - Referenced against TA-Lib's CVI (https://ta-lib.org/functions/cvi.html).
        - pandas-ta-classic: cvi (https://github.com/xgboosted/pandas-ta-classic).
        - Tulip Indicators: cvi (https://tulipindicators.org/cvi).

        Also known as: CV, CVI, Chaikin Volatility Indicator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the EMA of the range. Defaults to 10,
                as defined by Chaikin.
            roc_window (int, optional): The number of periods the rate of change looks back.
                Defaults to 10, as defined by Chaikin.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Chaikin Volatility.
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
            pd.DataFrame: Chaikin Volatility per ticker as decimals, or its growth (if growth is
                True).

        Notes:
        - The EMA is seeded with the simple average of its first `window` ranges, so this equals
          TA-Lib's CVI and pandas-ta-classic's cvi divided by 100.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NVDA", "AMD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chaikin_volatility().tail(5)
        ```

        Which returns:

        | Date       |    NVDA |     AMD |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | -0.0771 | -0.0443 |     -0.0722 |
        | 2025-12-26 | -0.0633 | -0.2027 |     -0.1996 |
        | 2025-12-29 | -0.2297 | -0.2813 |     -0.3266 |
        | 2025-12-30 | -0.2501 | -0.3586 |     -0.3925 |
        | 2025-12-31 | -0.1966 | -0.3463 |     -0.3378 |
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

        chaikin_volatility = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            volatility_model.get_chaikin_volatility,
            window,
            roc_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=chaikin_volatility,
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
    def get_relative_volatility_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        std_window: int = 10,
        method: str = "original",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Dorsey's Relative Volatility Index (RVI) of a given price series.

        The Relative Volatility Index is built like the Relative Strength Index, but measures the
        direction of volatility instead of the size of price changes: on a period that closes
        higher, the standard deviation of the last `std_window` closes counts as upward volatility,
        and on one that closes lower as downward volatility. Dorsey designed it to confirm other
        signals: buy signals when the RVI is above 50 and sell signals when it is below 50. The
        refined RVI averages the RVI of the highs and of the lows.

        The formula is as follows:

        - Up = StdDev(Close, std_window) if the close rose, else 0
        - Down = StdDev(Close, std_window) if the close fell, else 0
        - RVI = 100 × Wilder(Up, window) / (Wilder(Up, window) + Wilder(Down, window))
        - Refined RVI = (RVI of the highs + RVI of the lows) / 2

        Wilder is Wilder's smoothing, seeded with a simple average. The RVI runs from 0 to 100.

        For more information, see the following sources:

        - Dorsey, D. (1993). "The Relative Volatility Index." Technical Analysis of Stocks &
          Commodities, 11(6).
        - Dorsey, D. (1995). "Refining the Relative Volatility Index." Technical Analysis of
          Stocks & Commodities, 13(9).
        - Referenced against TA-Lib's RVI and RVIR (https://ta-lib.org/functions/rvi.html,
          https://ta-lib.org/functions/rvir.html).
        - pandas-ta-classic: rvi (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: RVI, RVIR, Relative Volatility Index. Not to be confused with the Relative
        Vigor Index (`get_relative_vigor_index`).

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of Wilder's smoothing. Defaults to 14,
                as defined by Dorsey.
            std_window (int, optional): The number of periods of the standard deviation. Defaults
                to 10, as defined by Dorsey.
            method (str, optional): "original" for the RVI of the close (Dorsey, 1993) or "refined"
                for the average of the RVI of the highs and the lows (Dorsey, 1995). Defaults to
                "original".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the RVI.
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
            pd.DataFrame: RVI values per ticker between 0 and 100, or their growth (if growth is
                True).

        Notes:
        - Equals TA-Lib's RVI (method "original") and RVIR (method "refined").
          pandas-ta-classic's rvi smooths with an EMA instead of Wilder's smoothing by default,
          which gives different values.
        - A window in which the price did not move has no RVI (missing) instead of 0.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["JPM", "GS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_relative_volatility_index().tail(5)
        ```

        Which returns:

        | Date       |     JPM |      GS |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 61.2674 | 73.4633 |     72.3434 |
        | 2025-12-26 | 56.7194 | 69.8239 |     66.3989 |
        | 2025-12-29 | 52.4974 | 66.3047 |     60.8153 |
        | 2025-12-30 | 48.5788 | 62.8379 |     55.7399 |
        | 2025-12-31 | 45.1855 | 59.4845 |     51.3189 |
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

        if method not in ("original", "refined"):
            raise ValueError("The method must be 'original' or 'refined'.")

        # The refined RVI averages the RVI of the highs and of the lows (Dorsey, 1995).
        columns = [close_column] if method == "original" else ["High", "Low"]
        relative_volatility_index = sum(
            calculate_per_ticker(
                historical_data,
                [column],
                volatility_model.get_relative_volatility_index,
                window,
                std_window,
            )
            for column in columns
        ) / len(columns)
        relative_volatility_index = relative_volatility_index.loc[
            self._start_date : self._end_date
        ]

        return finalize_dataset(
            dataset=relative_volatility_index,
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
    def get_acceleration_bands(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        factor: float = 4.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Headley's Acceleration Bands of a given price series.

        Acceleration Bands place an envelope around a simple average of the close whose width
        follows the High — Low range of each period relative to its price. Headley used a close
        outside the bands as the start of an accelerating move (a breakout) and a close back
        inside them as the end of it.

        The formula is as follows:

        - Range Ratio = factor × (High — Low) / (High + Low)
        - Upper Band = SMA(High × (1 + Range Ratio), window)
        - Middle Band = SMA(Close, window)
        - Lower Band = SMA(Low × (1 — Range Ratio), window)

        For more information, see the following sources:

        - Headley, P. (2002). "Big Trends in Trading: Strategies to Master Major Market Moves."
          John Wiley & Sons.
        - Referenced against TA-Lib's ACCBANDS (https://ta-lib.org/functions/accbands.html).
        - pandas-ta-classic: accbands (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: ACCBANDS, Price Headley's Acceleration Bands.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the simple averages. Defaults to 20,
                as defined by Headley.
            factor (float, optional): The multiplier of the relative range. Defaults to 4.0, as
                defined by Headley.
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
            pd.Series or pd.DataFrame: Upper Band, Middle Band and Lower Band in price units.

        Notes:
        - Equals TA-Lib's ACCBANDS and pandas-ta-classic's accbands.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_acceleration_bands().xs("TSLA", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Lower Band |   Middle Band |   Upper Band |
        |:-----------|-------------:|--------------:|-------------:|
        | 2025-12-24 |      422.209 |       458.558 |      493.941 |
        | 2025-12-26 |      424.545 |       460.988 |      497.583 |
        | 2025-12-29 |      425.825 |       462.462 |      499.78  |
        | 2025-12-30 |      427.161 |       463.676 |      501.346 |
        | 2025-12-31 |      429.244 |       464.7   |      501.571 |
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

        acceleration_bands = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_acceleration_bands,
            window,
            factor,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=acceleration_bands,
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
    def get_chande_kroll_stop(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        multiplier: float = 1.0,
        stop_window: int = 9,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chande Kroll Stop of a given price series.

        The Chande Kroll Stop is a pair of trailing stops a multiple of the Average True Range from
        the extreme price of the last `window` periods, smoothed a second time by taking the
        extreme of those first stops over `stop_window` periods. The high stop trails below the
        highs and the low stop above the lows: a close below the high stop exits a long position,
        a close above the low stop exits a short one, and a close above both lines marks an
        uptrend.

        The formula is as follows:

        - First High Stop = Highest High(window) — multiplier × ATR(window)
        - First Low Stop = Lowest Low(window) + multiplier × ATR(window)
        - High Stop = Highest(First High Stop, stop_window)
        - Low Stop = Lowest(First Low Stop, stop_window)

        For more information, see the following sources:

        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." John Wiley & Sons.
        - Referenced against TA-Lib's CKSP (https://ta-lib.org/functions/cksp.html).
        - pandas-ta-classic: cksp (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: CKSP, Chande Kroll stop loss. Chande and Kroll call the high stop the "stop
        short" and the low stop the "stop long".

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the highest high, the lowest low and
                the Average True Range. Defaults to 10, as in TA-Lib.
            multiplier (float, optional): The number of Average True Ranges the first stops sit
                from the extremes. Defaults to 1.0, as in TA-Lib.
            stop_window (int, optional): The number of periods over which the first stops are
                smoothed. Defaults to 9, as in TA-Lib.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the stops.
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
            pd.Series or pd.DataFrame: High Stop and Low Stop in price units.

        Notes:
        - The ATR is that of `get_average_true_range` (the StockCharts seeding), so the stops start
          one period before TA-Lib's CKSP and equal it, and pandas-ta-classic's cksp, once the
          seeds have converged (within 1e-8 after about 120 periods with the defaults).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["META"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chande_kroll_stop().xs("META", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   High Stop |   Low Stop |
        |:-----------|------------:|-----------:|
        | 2025-12-24 |     691.835 |    653.213 |
        | 2025-12-26 |     692.75  |    652.298 |
        | 2025-12-29 |     692.75  |    652.209 |
        | 2025-12-30 |     692.75  |    652.209 |
        | 2025-12-31 |     692.75  |    652.209 |
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

        chande_kroll_stop = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_chande_kroll_stop,
            window,
            multiplier,
            stop_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=chande_kroll_stop,
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
    def get_adaptive_price_zone(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        multiplier: float = 2.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Leibfarth's Adaptive Price Zone (APZ) of a given price series.

        The Adaptive Price Zone is a band around a fast, double-smoothed Exponential Moving Average
        of the close, with a width that is the same double-smoothed average of the High — Low
        range. Both averages use the square root of `window` as their period, so the zone follows
        the price closely. Leibfarth designed it for markets without a trend (an ADX below 30): a
        close above the zone is a chance to sell, and a close below it a chance to buy, in
        anticipation of a reversal.

        The formula is as follows:

        - Smoothing Period = √window
        - Middle Band = EMA(EMA(Close, Smoothing Period), Smoothing Period)
        - Range = EMA(EMA(High — Low, Smoothing Period), Smoothing Period)
        - Upper Band = Middle Band + multiplier × Range
        - Lower Band = Middle Band — multiplier × Range

        For more information, see the following sources:

        - Leibfarth, L. (2006). "Trading With An Adaptive Price Zone." Technical Analysis of
          Stocks & Commodities, 24(9).
        - finta: TA.APZ (https://github.com/peerchemist/finta).

        Also known as: APZ, Adaptive Price Zone.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods whose square root is the smoothing
                period. Defaults to 20, as in Leibfarth's article.
            multiplier (float, optional): The number of smoothed ranges between the middle band and
                the outer bands. Defaults to 2.0.
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
            pd.Series or pd.DataFrame: Upper Band, Middle Band and Lower Band in price units.

        Notes:
        - Each EMA has the smoothing constant 2 / (√window + 1) and is seeded with the simple
          average of its first ⌈√window⌉ values; after about 35 periods the zone equals
          Leibfarth's TradeStation formula, which starts from the first price.
        - finta's APZ smooths over the full `window` instead of its square root and centres the
          zone on a Double Exponential Moving Average, so it differs by design.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_adaptive_price_zone().xs("QQQ", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Lower Band |   Middle Band |   Upper Band |
        |:-----------|-------------:|--------------:|-------------:|
        | 2025-12-24 |      602.35  |       614.275 |      626.2   |
        | 2025-12-26 |      605.75  |       616.032 |      626.315 |
        | 2025-12-29 |      607.778 |       617.105 |      626.431 |
        | 2025-12-30 |      609.082 |       617.566 |      626.05  |
        | 2025-12-31 |      608.453 |       617.038 |      625.623 |
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

        adaptive_price_zone = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_adaptive_price_zone,
            window,
            multiplier,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=adaptive_price_zone,
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
    def get_aberration(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 5,
        atr_window: int = 15,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Aberration bands of a given price series.

        The Aberration bands place one Average True Range above and below a short simple average
        of the typical price. They come from Keith Fitschen's Aberration breakout system, which
        buys a close above the upper band and sells a close below the lower band.

        The formula is as follows:

        - Typical Price = (High + Low + Close) / 3
        - Middle Band = SMA(Typical Price, window)
        - Upper Band = Middle Band + ATR(atr_window)
        - Lower Band = Middle Band — ATR(atr_window)

        For more information, see the following sources:

        - Fitschen, K. (2013). "Building Reliable Trading Systems." John Wiley & Sons, on the
          Aberration system (1993), to which these bands are attributed; no definitive source of
          the formula exists.
        - pandas-ta-classic: aberration (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: ABER, Aberration.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the simple average of the typical
                price. Defaults to 5, as in pandas-ta-classic.
            atr_window (int, optional): The number of periods of the Average True Range. Defaults
                to 15, as in pandas-ta-classic.
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
            pd.Series or pd.DataFrame: Upper Band, Middle Band and Lower Band in price units.

        Notes:
        - The ATR is that of `get_average_true_range`; pandas-ta-classic seeds its ATR one period
          later and its bands equal these once the seeds have converged.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NFLX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_aberration().xs("NFLX", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Lower Band |   Middle Band |   Upper Band |
        |:-----------|-------------:|--------------:|-------------:|
        | 2025-12-24 |      90.8442 |       93.7507 |      96.6571 |
        | 2025-12-26 |      90.8786 |       93.686  |      96.4934 |
        | 2025-12-29 |      90.9345 |       93.644  |      96.3535 |
        | 2025-12-30 |      91.0784 |       93.6613 |      96.2442 |
        | 2025-12-31 |      91.3519 |       93.8367 |      96.3214 |
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

        aberration = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_aberration,
            window,
            atr_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=aberration,
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
    def get_profit_maximizer(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 10,
        atr_window: int = 10,
        multiplier: float = 3.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Profit Maximizer (PMax) of a given price series.

        The Profit Maximizer applies the Supertrend's trailing logic to a moving average instead of
        the price: two stops a multiple of the Average True Range below and above an Exponential
        Moving Average of the median price, each of which only tightens while the average stays
        on its side. The trend turns down when the average falls below the long stop and up when
        it rises above the short stop, and PMax is the active stop. Because the average is
        smoother than the price, PMax flips less often than the Supertrend.

        The formula is as follows:

        - Average = EMA((High + Low) / 2, window)
        - Long Stop = Average — multiplier × ATR(atr_window), never lowered while the Average stays
          above the previous Long Stop
        - Short Stop = Average + multiplier × ATR(atr_window), never raised while the Average stays
          below the previous Short Stop
        - PMax = Long Stop in an uptrend and Short Stop in a downtrend

        For more information, see the following sources:

        - Ozbilgic, K. (2020). "Profit Maximizer - PMax." TradingView script.
        - pandas-ta-classic: pmax (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: PMax, Profit Maximizer.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the EMA of the median price.
                Defaults to 10, as defined by Ozbilgic.
            atr_window (int, optional): The number of periods of the Average True Range. Defaults
                to 10, as defined by Ozbilgic.
            multiplier (float, optional): The number of Average True Ranges between the average and
                the stops. Defaults to 3.0, as defined by Ozbilgic.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the PMax.
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
            pd.Series or pd.DataFrame: PMax (the active stop, in price units) and Trend Direction (1
                for an uptrend, -1 for a downtrend).

        Notes:
        - This follows Ozbilgic's script (EMA of the median price, Wilder's ATR, starting in an
          uptrend). pandas-ta-classic's pmax averages the close and turns the trend on the close
          instead of the average, so it differs by design.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_profit_maximizer().xs("BTCUSD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |    PMax |   Trend Direction |
        |:-----------|--------:|------------------:|
        | 2025-12-27 | 96012   |                -1 |
        | 2025-12-28 | 95350.2 |                -1 |
        | 2025-12-29 | 95350.2 |                -1 |
        | 2025-12-30 | 95350.2 |                -1 |
        | 2025-12-31 | 95350.2 |                -1 |
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

        profit_maximizer = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_profit_maximizer,
            window,
            atr_window,
            multiplier,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=profit_maximizer,
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
    def get_price_distance(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Price Distance of a given price series.

        The Price Distance estimates how far the price travelled within a period: twice the High —
        Low range (down to the low and up to the high, or the reverse), less the open-to-close
        body that the range already covers once, plus the gap from the previous close to the open.

        The formula is as follows:

        - Price Distance = 2 × (High — Low) — |Close — Open| + |Open — Previous Close|

        For more information, see the following sources:

        - ProRealCode, "Price Distance" indicator.
        - pandas-ta-classic: pdist (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: PDIST, price distance volatility.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Price Distance.
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
            pd.DataFrame: Price Distance per ticker in price units, or its growth (if growth is
                True).

        Notes:
        - Equals pandas-ta-classic's pdist.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["KO", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_price_distance().tail(5)
        ```

        Which returns:

        | Date       |     KO |    PEP |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.6278 | 2.0768 |      3.6118 |
        | 2025-12-26 | 0.9417 | 1.611  |      4.6735 |
        | 2025-12-29 | 1.0889 | 2.3679 |      8.6524 |
        | 2025-12-30 | 0.9515 | 2.6979 |      3.8896 |
        | 2025-12-31 | 0.3924 | 1.7274 |      6.1619 |
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

        price_distance = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            volatility_model.get_price_distance,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=price_distance,
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
    def get_market_thermometer(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 22,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Elder's Market Thermometer of a given price series.

        The Market Thermometer measures how far each period's range pushes outside the previous
        one: the larger of the rise of the high above the previous high and the fall of the low
        below the previous low, or 0 for an inside period. Elder compares it with its Exponential
        Moving Average: a temperature far above the average (Elder uses three times) marks a burst
        of volatility in which to take profits, and several periods below the average a quiet
        market that often precedes an explosive move.

        The formula is as follows:

        - Thermometer = Max(High — Previous High, Previous Low — Low, 0)
        - Thermometer Average = EMA(Thermometer, window)

        For more information, see the following sources:

        - Elder, A. (2002). "Come Into My Trading Room." John Wiley & Sons.
        - pandas-ta-classic: thermo (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: THERMO, Elder's thermometer, market temperature.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the EMA. Defaults to 22, as defined by
                Elder.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Thermometer and its average.
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
            pd.Series or pd.DataFrame: Thermometer and Thermometer Average in price units.

        Notes:
        - An inside period has a temperature of 0, as Elder defines it. pandas-ta-classic's thermo
          takes the absolute changes of both the high and the low, so a falling high or a rising
          low also counts there; the two agree on periods with a higher high and a lower low.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_market_thermometer().xs("GLD", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Thermometer |   Thermometer Average |
        |:-----------|--------------:|----------------------:|
        | 2025-12-24 |          0    |                2.6893 |
        | 2025-12-26 |          5.82 |                2.9615 |
        | 2025-12-29 |         19.42 |                4.3927 |
        | 2025-12-30 |          0.04 |                4.0142 |
        | 2025-12-31 |          2.97 |                3.9234 |
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

        market_thermometer = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            volatility_model.get_market_thermometer,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=market_thermometer,
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
    def get_squeeze_momentum(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 20,
        num_std_dev: float = 2.0,
        keltner_multipliers: list[float] | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Squeeze Momentum (TTM Squeeze) of a given price series.

        The squeeze compares two volatility bands around the same simple average of the close:
        Bollinger Bands, which follow the standard deviation, and a Keltner Channel, which follows
        the average True Range. When the Bollinger Bands contract inside the Keltner Channel,
        volatility is unusually low and the market is "in a squeeze". Carter trades the release of
        the squeeze (the first period the bands leave the channel) in the direction of the
        momentum, LazyBear's linear regression of the close relative to the middle of its recent
        range and its average.

        The formula is as follows:

        - Middle = SMA(Close, window)
        - Bollinger Bands = Middle ± num_std_dev × StdDev(Close, window)
        - Keltner Channel(m) = Middle ± m × SMA(True Range, window)
        - Squeeze = the number of `keltner_multipliers` m with the Bollinger Bands inside Keltner
          Channel(m)
        - Momentum = Linear Regression(Close — ((Highest High + Lowest Low) / 2 + Middle) / 2, window)

        With the default multipliers 2, 1.5 and 1 (the wide, normal and narrow channels of the TTM
        Squeeze Pro), the Squeeze is 0 without a squeeze and 1, 2 or 3 for a low, normal or high
        compression. Carter's original squeeze (a Keltner Channel of 1.5) is on from 2;
        `keltner_multipliers=[1.5]` returns it as 1 (on) or 0 (off).

        For more information, see the following sources:

        - Carter, J.F. (2005). "Mastering the Trade." McGraw-Hill, chapter 11.
        - LazyBear (2014). "Squeeze Momentum Indicator." TradingView script.
        - Referenced against TA-Lib's SQZMOM (https://ta-lib.org/functions/sqzmom.html).
        - pandas-ta-classic: squeeze and squeeze_pro (https://github.com/xgboosted/pandas-ta-classic).
        - finta: TA.SQZMI (https://github.com/peerchemist/finta).

        Also known as: TTM Squeeze, Squeeze Pro, SQZ, SQZMOM, squeeze momentum indicator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the bands, the channels, the range
                and the regression. Defaults to 20, as defined by Carter.
            num_std_dev (float, optional): The number of standard deviations of the Bollinger
                Bands. Defaults to 2.0, as defined by Carter.
            keltner_multipliers (list[float] | None, optional): The widths, in average True Ranges,
                of the Keltner Channels the Bollinger Bands are compared with. Defaults to None,
                which uses [2.0, 1.5, 1.0], as in TA-Lib's SQZMOM.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Momentum and Squeeze.
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
            pd.Series or pd.DataFrame: Momentum (in price units) and Squeeze (the number of Keltner
                Channels the Bollinger Bands lie inside).

        Notes:
        - The Momentum equals TA-Lib's SQZMOM and pandas-ta-classic's `squeeze(lazybear=True)`.
          The Squeeze equals TA-Lib's squeeze output, with 0 instead of -1 for no squeeze. It starts
          on the `window`-th period, while TA-Lib reports it from the first Momentum on.
        - The Keltner Channel here is LazyBear's (a simple average of the close and of the True
          Range), not that of `get_keltner_channels`, and the Bollinger standard deviation is the
          population one, as in `get_bollinger_bands`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_squeeze_momentum().xs("QQQ", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Momentum |   Squeeze |
        |:-----------|-----------:|----------:|
        | 2025-12-24 |     3.8112 |         1 |
        | 2025-12-26 |     3.072  |         1 |
        | 2025-12-29 |     2.2261 |         1 |
        | 2025-12-30 |     0.9902 |         1 |
        | 2025-12-31 |    -0.3707 |         1 |
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

        if keltner_multipliers is None:
            keltner_multipliers = [2.0, 1.5, 1.0]

        squeeze_momentum = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_squeeze_momentum,
            window,
            num_std_dev,
            keltner_multipliers,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=squeeze_momentum,
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
