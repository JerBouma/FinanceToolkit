"""Volatility Module"""

__docformat__ = "google"

from typing import Literal

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.technicals.helpers import (
    calculate_per_ticker,
    get_observations_per_year,
    report_missing_values,
)
from financetoolkit.technicals.volatility import volatility_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
)
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


@report_missing_values
@filter_shown_columns
@track_sources
class VolatilityTechnicals:
    """
    Volatility indicators: the Bollinger Bands, %B and BandWidth, the true range, average true range (also normalized, and the average day range), Chaikin's volatility, the relative volatility index, the mass index, the Supertrend and the Profit Maximizer, the Chandelier Exit, the Chande Kroll Stop, the Keltner channels and aberration bands, the Donchian channels, the acceleration bands, the adaptive price zone, the price distance, Elder's market thermometer, the squeeze momentum, the signal-to-noise ratio, the price bands (`get_bollinger_bands(method="price_bands")`), the volatility cone and the standard deviation channel.
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
        Collect the volatility indicators of every ticker in one table: how widely the prices move and the bands and
        channels around them.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            window (int, optional): The window size for calculating indicators.
                Defaults to 14. The Bollinger Bands and their %B and BandWidth keep their own
                window of 20 periods (with 2 standard deviations), as defined by Bollinger, and
                the Supertrend, Chaikin Volatility, acceleration bands, Chande Kroll Stop,
                adaptive price zone, aberration bands, Profit Maximizer, market thermometer,
                squeeze momentum and Price Bands their own defaults.
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
        - The Standard Deviation Channel is left out: it is fitted over the whole date range at
          once, so its values depend on later prices.
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
        self._check_arguments(period, close_column, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday" and self._historical_data[period].empty:
            raise InvalidArgumentError(
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
            self.get_average_true_range(
                period=period,
                close_column=close_column,
                window=window,
                method="normalized",
            )
        )
        volatility_indicators["Average Day Range"] = self.get_average_true_range(
            period=period, close_column=close_column, window=window, method="day_range"
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

        aberration = self.get_keltner_channels(
            period=period, close_column=close_column, method="aberration"
        )
        for band in ["Upper", "Middle", "Lower"]:
            volatility_indicators[f"Aberration {band}"] = aberration[f"{band} Band"]

        profit_maximizer = self.get_supertrend(
            period=period, close_column=close_column, method="profit_maximizer"
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

        volatility_indicators["Signal-to-Noise Ratio"] = self.get_signal_to_noise_ratio(
            period=period, close_column=close_column, window=window
        )

        price_bands = self.get_bollinger_bands(
            period=period, close_column=close_column, method="price_bands"
        )
        for band in ["Upper", "Middle", "Lower"]:
            volatility_indicators[f"Price Band {band}"] = price_bands[f"{band} Band"]

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
        short_window: int = 2,
        method: Literal["standard", "price_bands"] = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Bollinger Bands or the Price Bands: bands around a moving average, a number of standard
        deviations wide, which show how high or low the price is relative to its recent volatility.

        Bollinger Bands are a volatility indicator that consists of three lines: an upper band,
        a middle band (simple moving average), and a lower band. The upper and lower bands are
        calculated as the moving average plus and minus a specified number of standard deviations,
        respectively.

        Price Bands are a variant of the Bollinger Bands whose width comes from the distance
        between the moving average and a fast moving average of the price, rather than from the
        distance between the price and the moving average. The fast average filters out the noise
        of single periods, so a spike widens the bands less than it widens the Bollinger Bands,
        and the bands react to sustained moves away from the average instead.

        The formula for the Bollinger Bands is as follows:

        - Middle Band = SMA(Close, Window), or another moving average with `moving_average`
        - Upper Band = Middle Band + (Num Std Dev * Std Dev)
        - Lower Band = Middle Band — (Num Std Dev * Std Dev)

        The formula for the Price Bands is as follows:

        - Middle Band = SMA(Close, window), or another moving average with `moving_average`
        - Deviation = Middle Band — SMA(Close, short_window)
        - Width = Population Standard Deviation(Deviation, window)
        - Upper Band = Middle Band + num_std_dev * Width
        - Lower Band = Middle Band — num_std_dev * Width

        The standard deviation is the *population* standard deviation (dividing by n), as
        Bollinger himself specifies and as TA-Lib and StockCharts both define it, not
        pandas' default sample standard deviation.

        The band width tracks volatility: bands that narrow sharply (a squeeze) often precede a
        larger move, without saying in which direction. A close at or beyond a band is not a sell
        or buy signal by itself, since prices can walk along a band in a trend. The defaults are
        Bollinger's own setting, a 20-period average with 2 standard deviations, as on most
        charting platforms. The Price Bands are read the same way, but since their width follows
        the smoothed price, they are narrower than the Bollinger Bands of the same window and a
        close outside them is more common.

        For more information, see the following sources:

        - Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill.
        - The Price Bands are defined by Brian G. Peterson in TTR's PBands, after the bands of
          Bollinger (2001), and referenced against TTR's PBands (https://cran.r-project.org/package=TTR).

        Referenced against:
        - TA-Lib, BBANDS: https://ta-lib.org/functions/bbands.html
        - TTR (R), PBands (Price Bands): https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/priceBands.R#L68

        Shares similar logic:
        - pandas-ta-classic, bbands: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/bbands.py#L48

        Also known as: Bollinger Bands, BB, volatility bands, price channels, PBands, price bands,
        smoothed Bollinger Bands.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for moving average calculation.
                The number of periods (time intervals) over which to calculate the moving average
                and the standard deviation. Defaults to 20, as in TTR's PBands.
            num_std_dev (float, optional): Number of standard deviations for the bands.
                Defaults to 2.
            moving_average (str, optional): The moving average of the middle band (and, with
                method="price_bands", of the fast average), one of the window-only methods of
                `get_moving_average`. Options are: "simple" (Simple Moving Average), "exponential" (Exponential
                Moving Average), "double_exponential" (Double Exponential Moving Average),
                "triple_exponential" (Triple Exponential Moving Average), "zero_lag" (Zero Lag Exponential
                Moving Average), "wilder" (Wilder's Moving Average), "triangular" (Triangular Moving Average),
                "weighted" (Weighted Moving Average), "hull" (Hull Moving Average), "sine" (Sine Weighted Moving
                Average), "fibonacci" (Fibonacci Weighted Moving Average), "pascal" (Pascal Weighted Moving
                Average). Defaults to "simple", Bollinger's own.
            short_window (int, optional): The number of periods of the fast moving average, with
                method="price_bands" only. Defaults to 2.
            method (str, optional): The variant of the bands. Options are: "standard" (the Bollinger Bands),
                "price_bands" (the Price Bands, whose width comes from the fast moving average). Defaults to
                "standard".
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
            The Close and the Bollinger Bands or Price Bands (upper, middle, lower), in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Bollinger Bands or Price Bands for each asset in the Toolkit instance.
        - The bands agree with TA-Lib's BBANDS with a simple moving average, and with the exponential,
          weighted, double exponential, triple exponential and triangular averages its moving average
          types 1 to 5. The standard deviation stays the one around the simple average of the
          window, as in TA-Lib, whatever the middle band.
        - The standard deviation of the Price Bands is the population one (dividing by `window`), as
          in TTR's PBands. The Price Bands start on period 2 * `window` — 2.
        - TTR's PBands can also centre the bands (`centered=TRUE`), which adds the moving average
          of the deviation divided by the width to the middle band. That adds a number of standard
          deviations to a price, so the shift does not scale with the price, and the Toolkit
          leaves it out.
        - When growth is requested, the method calculates the growth of the bands
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
        self._check_arguments(
            period, close_column, window=window, short_window=short_window
        )
        check_option("method", method, ("standard", "price_bands"))
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

        if method == "price_bands":
            # The Close next to the bands, as the Bollinger Bands return it.
            bollinger_bands = calculate_per_ticker(
                historical_data,
                [close_column],
                lambda prices, *args: volatility_model.get_price_bands(
                    prices, *args
                ).assign(Close=prices),
                window,
                num_std_dev,
                short_window,
                moving_average,
            )
        else:
            bollinger_bands = calculate_per_ticker(
                historical_data,
                [close_column],
                volatility_model.get_bollinger_bands,
                window,
                num_std_dev,
                moving_average,
            )

        return finalize_dataset(
            dataset=bollinger_bands.loc[self._start_date : self._end_date],
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
        Calculate the Bollinger %B and BandWidth: where the close lies within the Bollinger Bands and how wide the
        bands are, which show overbought and oversold levels and squeezes of volatility.

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

        Also known as: Percent B, %b, Bollinger Band Width, BBW, Bollinger Squeeze.

        Referenced against:
        - TA-Lib, PERCENTB: https://ta-lib.org/functions/percentb.html
        - TA-Lib, BBW: https://ta-lib.org/functions/bbw.html

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - With a simple moving average, %B agrees with TA-Lib's PERCENTB and BandWidth
          agrees with TA-Lib's BBW divided by 100, since TA-Lib's BBW is in percent.
        - When growth is requested, the method calculates the growth of %B and BandWidth
          over the given lag.

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
        Calculate the True Range (TR): the range of a period including any gap from the previous close, the basic
        measure of how far the price moved.

        The True Range (TR) is a measure of market volatility that considers the differences
        between the high and low prices and the previous closing price. It provides insights
        into the price movement of an asset.

        The formula is as follows:

        - TR = max(high — low, abs(high — previous_close), abs(low — previous_close))

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.

        Referenced against:
        - TA-Lib, TRANGE: https://ta-lib.org/functions/trange.html

        Also known as: TR, true range.

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
        - The True Range agrees with TA-Lib's TRANGE from the second period on. The first period, which
          has no previous close, is High — Low here and empty in TA-Lib.
        - When growth is requested, the method calculates the growth of the True Range
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
        method: Literal["standard", "normalized", "day_range"] = "standard",
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Average True Range (ATR), normalised by the close (NATR) or as the Average Day Range (ADR):
        the average range of the periods, a measure of how much the price moves.

        The Average True Range (ATR) is a technical indicator that measures the volatility
        of an asset's price movements over a specified number of periods. It provides insights
        into the potential price range of an asset, which can help traders and investors make
        more informed decisions. The Normalized Average True Range is the
        Average True Range divided by the close, so the average range is a share of the price:
        unlike the ATR, which is in price units, it compares across stocks with different prices
        and over periods in which the price changed a lot. The Average Day Range
        is the simple average of the High — Low range: unlike the Average True
        Range it leaves out the gaps between periods, so it measures how far the price moves within
        a period, a basis for intraday targets and stops (for example a stop half an ADR away).

        The formula is as follows, per method:

        - Average True Range: TR = max(high — low, abs(high — previous_close), abs(low — previous_close))
          and ATR = Wilder's Smoothed Moving Average of TR over `window` periods
        - Normalized Average True Range: NATR = ATR(window) / Close, where the ATR uses Wilder's smoothing; 0.02 is
          an average range of 2% of the close
        - Average Day Range: ADR = SMA(High — Low, window)

        The ATR is in price units, which makes it the common basis for stop distances (for example
        two ATRs below the entry) and for sizing positions to equal risk. For the same reason it
        cannot be compared across stocks or over long periods in which the price changed a lot;
        divide it by the price for a percentage volatility. It measures how far price moves, not in
        which direction.

        For more information, see the following sources:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - Forman, J. (2006). "Cross-Market Evaluations With Normalized Average True Range."
          Technical Analysis of Stocks & Commodities, 24(5).
        - Keltner, C.W. (1960). "How to Make Money in Commodities." The Keltner Statistical
          Service, whose channel is the average day range around the typical price.

        Referenced against:
        - TA-Lib, ATR: https://ta-lib.org/functions/atr.html
        - TA-Lib, NATR: https://ta-lib.org/functions/natr.html
        - TA-Lib, ADR: https://ta-lib.org/functions/adr.html

        Shares similar logic:
        - pandas-ta-classic, natr: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/natr.py#L20
        - Tulip Indicators, natr: https://tulipindicators.org/natr

        Also known as: ATR, volatility indicator, NATR, normalized average true range, ATR percent,
        ATRP, ADR, average day range, average daily range, average high-low range.

        See `volatility_model.get_average_true_range` for the full formula; Wilder's smoothing
        constant (1/window) is slower than a standard EMA's (2/(window+1)) of the same window.

        The first period has no previous close, so its True Range is High — Low and the first ATR
        is the average of the first `window` True Ranges (the StockCharts convention; TA-Lib starts
        one period later).

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods over which to calculate the Average
                True Range, or of the average of the range with method="day_range". Defaults to 14,
                as in TA-Lib.
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
            method (str, optional): Options are: "standard" (the ATR), "normalized" (the NATR, a decimal of the
                close), "day_range" (the Average Day Range). Defaults to "standard".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: ATR values per ticker in price units, or their growth when growth is requested.
                For the NATR the values are decimals and for the ADR price units.

        Notes:
        - ATR values are typically used to assess the volatility and potential price movement of an asset.
        - A higher ATR value indicates higher volatility, while a lower ATR value suggests lower volatility.
        - Because the True Range starts on the first period, the ATR starts one period before
          TA-Lib's ATR and differs from it until the two converge (within 0.01% after about 80
          periods with the default window).
        - Normalized Average True Range: the ATR is the one above (the StockCharts seeding), so the NATR starts one
          period before TA-Lib's NATR and equals it, divided by 100, once the seeds have converged
          (within 1e-8 after about 150 periods with the default window).
        - Average Day Range: equals TA-Lib's ADR.

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
        self._check_arguments(period, close_column, window=window)
        methods = ("standard", "normalized", "day_range")
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

        average_true_range = calculate_per_ticker(
            historical_data,
            ["High", "Low"] if method == "day_range" else ["High", "Low", close_column],
            {
                "standard": volatility_model.get_average_true_range,
                "normalized": volatility_model.get_normalized_average_true_range,
                "day_range": volatility_model.get_average_day_range,
            }[method],
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
        Calculate the Mass Index: the sum of the ratios of an exponential moving average of the high-low range to
        its own average, which signals a reversal when the range widens and then narrows.

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

        Referenced against:
        - TA-Lib, MASSI: https://ta-lib.org/functions/massi.html

        Also known as: Mass Index, MI. See `volatility_model.get_mass_index` for the references.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
          the 41st with the default windows. The Mass Index agrees with TA-Lib's MASSI.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

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
        self._check_arguments(
            period, close_column, ema_window=ema_window, sum_window=sum_window
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
        method: Literal["standard", "profit_maximizer"] = "standard",
        atr_window: int = 10,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Supertrend or the Profit Maximizer: a trailing stop that shows the direction of the trend and
        flips when it is crossed.

        The stop is a multiple of the Average True Range away from the price, or for the Profit Maximizer from its
        moving average.

        The Supertrend indicator plots a single trailing line that flips between sitting
        below price (in an uptrend) and above price (in a downtrend). The line is built from
        two bands offset from the median price ((High + Low) / 2) by a multiple of the
        Average True Range, which are then "ratcheted" period over period — each band can
        only move in the direction that tightens around price — so that the active band only
        flips to the other side once the closing price actually crosses it. This makes
        Supertrend both a trend filter (the flip direction signals a trend change) and a
        trailing stop-loss level.

        The Profit Maximizer applies the Supertrend's trailing logic to
        a moving average instead of the price: two stops a multiple of the Average True Range below
        and above an Exponential Moving Average of the median price, each of which only tightens
        while the average stays on its side. The trend turns down when the average falls below the
        long stop and up when it rises above the short stop, and PMax is the active stop. Because
        the average is smoother than the price, PMax flips less often than the Supertrend.

        The formula is as follows, per method:

        "standard":

        - Basic Upper Band = (High + Low) / 2 + multiplier * ATR(window)
        - Basic Lower Band = (High + Low) / 2 — multiplier * ATR(window)
        - Final Upper Band(t) = Basic Upper Band(t) if Basic Upper Band(t) < Final Upper
          Band(t-1) or Close(t-1) > Final Upper Band(t-1), else Final Upper Band(t-1)
        - Final Lower Band(t) = Basic Lower Band(t) if Basic Lower Band(t) > Final Lower
          Band(t-1) or Close(t-1) < Final Lower Band(t-1), else Final Lower Band(t-1)
        - While in an uptrend, Supertrend = Final Lower Band, until Close crosses below it,
          at which point the trend flips to a downtrend and Supertrend = Final Upper Band
          (and vice versa)

        "profit_maximizer":

        - Average = EMA((High + Low) / 2, window)
        - Long Stop = Average — multiplier × ATR(atr_window), never lowered while the Average stays
          above the previous Long Stop
        - Short Stop = Average + multiplier × ATR(atr_window), never raised while the Average stays
          below the previous Short Stop
        - PMax = Long Stop in an uptrend and Short Stop in a downtrend

        For more information, see the following sources:

        - Ozbilgic, K. (2020). "Profit Maximizer - PMax." TradingView script.

        Referenced against:
        - TA-Lib, SUPERTREND: https://ta-lib.org/functions/supertrend.html

        Shares similar logic:
        - pandas-ta-classic, pmax (Profit Maximizer): https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/pmax.py#L51

        Also known as: Supertrend, SuperTrend, PMax, Profit Maximizer.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): Number of periods for the underlying Average True Range
                calculation, and with method="profit_maximizer" of the EMA of the median price.
                Defaults to 10, as defined by Ozbilgic for the Profit Maximizer.
            multiplier (float, optional): Multiplier applied to the Average True Range to
                determine how far the bands (or the stops of the Profit Maximizer) sit from the median
                price (or its average). Defaults to 3.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Supertrend and
                Trend Direction values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): Options are: "standard" (the Supertrend), "profit_maximizer" (the Profit
                Maximizer, PMax). Defaults to "standard".
            atr_window (int, optional): The number of periods of the Average True Range of the
                Profit Maximizer. Only used when method="profit_maximizer". Defaults to 10, as
                defined by Ozbilgic.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame:
            Supertrend (the trailing indicator line) and Trend Direction (1 for an uptrend
            and -1 for a downtrend) values, or for the Profit Maximizer PMax (the active
            stop, in price units) and Trend Direction.

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
          agrees with TA-Lib's `trend` from TA-Lib's first value.
        - Profit Maximizer: this follows Ozbilgic's script (EMA of the median price, Wilder's ATR,
          starting in an uptrend). pandas-ta-classic's pmax averages the close and turns the trend on
          the close instead of the average, so it differs by design.
        - When growth is requested, the method calculates the growth of the Supertrend and
          Trend Direction values over the given lag.

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
        self._check_arguments(
            period, close_column, window=window, atr_window=atr_window
        )
        methods = ("standard", "profit_maximizer")
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

        supertrend = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            *(
                (volatility_model.get_profit_maximizer, window, atr_window, multiplier)
                if method == "profit_maximizer"
                else (volatility_model.get_supertrend, window, multiplier)
            ),
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
        Calculate the Chandelier Exit: a trailing stop a multiple of the Average True Range below the highest high
        (or above the lowest low), which keeps a trade open while the trend lasts.

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

        Shares similar logic:
        - TA-Lib, CE: https://ta-lib.org/functions/ce.html

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - When growth is requested, the method calculates the growth of the Long Exit and Short Exit
          over the given lag.

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
        window: int | None = None,
        atr_window: int | None = None,
        multiplier: float = 2,
        method: Literal["exponential", "original", "aberration"] = "exponential",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Keltner Channels or the Aberration bands: bands a multiple of the Average True Range above and
        below an average of the price, which show breakouts and overbought and oversold levels.

        The Keltner Channels consist of three lines: a middle line, the Exponential Moving Average
        (EMA) of the closing prices, and an upper and a lower line a multiple of the Average True
        Range (ATR) above and below it.

        The formula is as follows:

        - Middle Line = EMA(Close, window)
        - Upper Line = Middle Line + multiplier * ATR(atr_window)
        - Lower Line = Middle Line — multiplier * ATR(atr_window), where the ATR uses Wilder's smoothing.

        Keltner's original channel is a simple average of the typical price
        with the average High — Low range on either side.

        - Middle Line = SMA((High + Low + Close) / 3, window)
        - Upper Line = Middle Line + multiplier * SMA(High — Low, window)
        - Lower Line = Middle Line — multiplier * SMA(High — Low, window)

        The Aberration bands place one Average True
        Range above and below a short simple average of the typical price. They come from Keith
        Fitschen's Aberration breakout system, which buys a close above the upper band and sells a
        close below the lower band.

        - Middle Band = SMA((High + Low + Close) / 3, window)
        - Upper Band = Middle Band + ATR(atr_window)
        - Lower Band = Middle Band — ATR(atr_window)

        For more information, see the following sources:

        - Keltner, C.W. (1960). "How to Make Money in Commodities." The Keltner Statistical Service.
        - Fitschen, K. (2013). "Building Reliable Trading Systems." John Wiley & Sons, on the
          Aberration system (1993), to which the Aberration bands are attributed; no definitive
          source of the formula exists.

        Referenced against:
        - TA-Lib, KC: https://ta-lib.org/functions/kc.html
        - ta, KeltnerChannel: https://github.com/bukosabino/ta/blob/0.11.0/ta/volatility.py#L183
        - pandas-ta-classic, aberration (Aberration): https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/aberration.py#L14

        Also known as: Keltner Channels, KC, ATR-based bands, volatility channels, ABER, Aberration, Aberration bands.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int | None, optional): Number of periods for the Exponential Moving Average (the
                middle line), or of the simple average of the typical price ("original" and
                "aberration"). Defaults to None, which uses 20, as on StockCharts, and 5 for
                "aberration", as in pandas-ta-classic.
            atr_window (int | None, optional): Number of periods for ATR calculation. Not used by
                "original". Defaults to None, which uses 10, as on StockCharts, and 15 for
                "aberration", as in pandas-ta-classic.
            multiplier (float, optional): Multiplier for ATR (or, for the original channel, the
                average range) to determine channel width. Defaults to 2; Keltner's original channel
                uses 1. Not used by "aberration", whose bands sit one ATR away.
            method (str, optional): The channels. Options are: "exponential" (the EMA of the close with ATR
                bands, as StockCharts), "original" (Keltner's simple average of the typical price with average
                range bands), "aberration" (the Aberration bands). Defaults to "exponential".
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
            pd.Series or pd.DataFrame: Keltner Channels ("Upper Line", "Middle Line" and "Lower
            Line"), or for the Aberration bands the "Upper Band", "Middle Band" and "Lower Band", in
            price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates Keltner Channels
          for each asset in the Toolkit instance.
        - The middle line is the EMA of the close, as on StockCharts. TA-Lib's KC takes the EMA of
          the typical price ((High + Low + Close) / 3) and starts its ATR one period later, so the
          channels differ throughout (by about 0.3% on daily data).
        - The original channel uses `window` for both averages (Keltner used 10 days with one
          average range, a multiplier of 1) and no ATR, so `atr_window` does not apply. With
          a multiplier of 1 it agrees with ta's KeltnerChannel in its original version.
        - Aberration bands: the ATR is that of `get_average_true_range`; pandas-ta-classic seeds its ATR
          one period later and its bands equal these once the seeds have converged.
        - When growth is requested, the method calculates the growth of the channels over the given lag.
        - ta's KeltnerChannel defaults to Keltner's original channel.

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
        self._check_arguments(
            period, close_column, window=window, atr_window=atr_window
        )
        methods = ("exponential", "original", "aberration")
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
            window = 5 if method == "aberration" else 20
        if atr_window is None:
            atr_window = 15 if method == "aberration" else 10

        historical_data = self._get_historical_data(period, close_column)

        keltner_channels = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            *(
                (volatility_model.get_aberration, window, atr_window)
                if method == "aberration"
                else (
                    volatility_model.get_keltner_channels,
                    window,
                    atr_window,
                    multiplier,
                    method,
                )
            ),
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
        Calculate the Donchian Channels: the highest high and the lowest low of the previous periods, whose
        breakouts signal the start of a trend.

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

        Referenced against:
        - TA-Lib, DONCHIAN: https://ta-lib.org/functions/donchian.html

        Also known as: Donchian Channels, price channel breakout.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - TA-Lib's DONCHIAN includes the current period, so each channel here agrees with TA-Lib's of the
          previous period; including the current period, it agrees with TA-Lib's DONCHIAN.
        - When growth is requested, the method calculates the growth of the channels over the given lag.

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
        Calculate the Volatility Cone for the specified period: the range of the historical annualised realised
        volatility over several windows, which shows whether the current volatility is high or low for each window.

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
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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

        volatility_cone_df = calculate_per_ticker(
            historical_data,
            [close_column],
            lambda prices: volatility_model.get_volatility_cone(
                prices,
                windows=windows,
                periods_per_year=(
                    get_observations_per_year(prices) * bars_per_day
                    if period in ("daily", "intraday")
                    else PERIODS_PER_YEAR[period]
                ),
            ),
        )

        return apply_rounding(
            volatility_cone_df, rounding if rounding is not None else self._rounding
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
        Calculate Chaikin's Volatility: the percentage change of the average high-low range over a window, which
        shows whether the range is widening or narrowing.

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

        Referenced against:
        - TA-Lib, CVI: https://ta-lib.org/functions/cvi.html

        Shares similar logic:
        - pandas-ta-classic, cvi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/cvi.py#L12
        - Tulip Indicators, cvi: https://tulipindicators.org/cvi

        Also known as: CV, CVI, Chaikin Volatility Indicator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        self._check_arguments(
            period, close_column, window=window, roc_window=roc_window
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
        method: Literal["original", "refined"] = "original",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Dorsey's Relative Volatility Index (RVI): the standard deviation of the up periods against that of
        the down periods, from 0 to 100, which shows the direction of volatility.

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

        Referenced against:
        - TA-Lib, RVI: https://ta-lib.org/functions/rvi.html
        - TA-Lib, RVIR: https://ta-lib.org/functions/rvir.html

        Shares similar logic:
        - pandas-ta-classic, rvi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/rvi.py#L61

        Also known as: RVI, RVIR, Relative Volatility Index. Not to be confused with the Relative
        Vigor Index (`get_relative_vigor_index`).

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of Wilder's smoothing. Defaults to 14,
                as defined by Dorsey.
            std_window (int, optional): The number of periods of the standard deviation. Defaults
                to 10, as defined by Dorsey.
            method (str, optional): Options are: "original" (the RVI of the close; Dorsey, 1993), "refined" (the
                average of the RVI of the highs and the lows; Dorsey, 1995). Defaults to "original".
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
        self._check_arguments(
            period, close_column, window=window, std_window=std_window
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

        check_option("method", method, ("original", "refined"))

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
        multiplier: float = 4.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Headley's Acceleration Bands: bands around a moving average that widen with the range of the
        candles, whose breakouts show an accelerating trend.

        Acceleration Bands place an envelope around a simple average of the close whose width
        follows the High — Low range of each period relative to its price. Headley used a close
        outside the bands as the start of an accelerating move (a breakout) and a close back
        inside them as the end of it.

        The formula is as follows:

        - Range Ratio = multiplier × (High — Low) / (High + Low)
        - Upper Band = SMA(High × (1 + Range Ratio), window)
        - Middle Band = SMA(Close, window)
        - Lower Band = SMA(Low × (1 — Range Ratio), window)

        For more information, see the following sources:

        - Headley, P. (2002). "Big Trends in Trading: Strategies to Master Major Market Moves."
          John Wiley & Sons.

        Referenced against:
        - TA-Lib, ACCBANDS: https://ta-lib.org/functions/accbands.html
        - pandas-ta-classic, accbands: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/accbands.py#L17

        Also known as: ACCBANDS, Price Headley's Acceleration Bands.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods of the simple averages. Defaults to 20,
                as defined by Headley.
            multiplier (float, optional): The multiplier of the relative range. Defaults to 4.0, as
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

        acceleration_bands = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_acceleration_bands,
            window,
            multiplier,
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
        Calculate the Chande Kroll Stop: a long and a short stop a multiple of the Average True Range from the
        recent extremes, which shows where a trend would be broken.

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

        Referenced against:
        - TA-Lib, CKSP: https://ta-lib.org/functions/cksp.html

        Shares similar logic:
        - pandas-ta-classic, cksp: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/cksp.py#L12

        Also known as: CKSP, Chande Kroll stop loss. Chande and Kroll call the high stop the "stop
        short" and the low stop the "stop long".

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        self._check_arguments(
            period, close_column, window=window, stop_window=stop_window
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
        Calculate Leibfarth's Adaptive Price Zone (APZ): bands around a double-smoothed exponential moving average
        that adapt to the range, which show turning points in a sideways market.

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

        Shares similar logic:
        - finta, APZ: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1907

        Also known as: APZ, Adaptive Price Zone.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        Calculate the Price Distance: the distance the price travelled within each period, including the gap from
        the previous close, a measure of how much the price moves.

        The Price Distance estimates how far the price travelled within a period: twice the High —
        Low range (down to the low and up to the high, or the reverse), less the open-to-close
        body that the range already covers once, plus the gap from the previous close to the open.

        The formula is as follows:

        - Price Distance = 2 × (High — Low) — |Close — Open| + |Open — Previous Close|

        For more information, see the following sources:

        - ProRealCode, "Price Distance" indicator.

        Referenced against:
        - pandas-ta-classic, pdist: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/pdist.py#L17

        Also known as: PDIST, price distance volatility.

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
        Calculate Elder's Market Thermometer: how far each period's range pushes outside that of the previous
        period, which shows how hot or quiet the market is.

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

        Shares similar logic:
        - pandas-ta-classic, thermo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/thermo.py#L18

        Also known as: THERMO, Elder's thermometer, market temperature.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        Calculate the Squeeze Momentum (TTM Squeeze): whether the Bollinger Bands lie inside the Keltner Channels (a
        squeeze of volatility) and the momentum that shows the direction of the breakout.

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
        a single Keltner multiplier of 1.5 returns it as 1 (on) or 0 (off).

        For more information, see the following sources:

        - Carter, J.F. (2005). "Mastering the Trade." McGraw-Hill, chapter 11.
        - LazyBear (2014). "Squeeze Momentum Indicator." TradingView script.

        Referenced against:
        - TA-Lib, SQZMOM: https://ta-lib.org/functions/sqzmom.html
        - pandas-ta-classic, squeeze: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/squeeze.py#L94

        Shares similar logic:
        - pandas-ta-classic, squeeze_pro: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/squeeze_pro.py#L17
        - finta, SQZMI: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1941

        Also known as: TTM Squeeze, Squeeze Pro, SQZ, SQZMOM, squeeze momentum indicator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
        - The Momentum agrees with TA-Lib's SQZMOM and pandas-ta-classic's `squeeze` in LazyBear's form.
          The Squeeze agrees with TA-Lib's squeeze output, with 0 instead of -1 for no squeeze. It starts
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

    @handle_portfolio
    @handle_errors
    def get_standard_deviation_channel(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int | None = None,
        num_std_dev: float | list[float] | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Standard Deviation Channel: a least-squares line through the prices with bands a number of
        standard deviations above and below it.

        The line runs through all prices of the Toolkit's date range, or the last `window` of them.

        The channel shows the trend of the whole period and how far the prices stray from it: a
        price near an outer band lies far from the trend of the period. It is the thinkorswim
        study StDevAll, which draws the regression line of all bars (InertiaAll) with bands at
        1, 2 and 3 standard deviations.

        The formula is as follows, with time x = 0, ..., n - 1 over the n prices of the period:

        - Linear Regression = b0 + b1 * x, the least-squares line through the prices
        - Upper Band k = Linear Regression + k * StdDev(Close)
        - Lower Band k = Linear Regression - k * StdDev(Close)

        with the sample standard deviation (ddof=1) of the prices of the period.

        For more information, see the following sources:

        - thinkorswim (TD Ameritrade). thinkScript studies StDevAll and InertiaAll.

        Referenced against:
        - pandas-ta-classic, tos_stdevall: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/tos_stdevall.py#L15

        Also known as: standard deviation channel, StDevAll, linear regression channel, regression bands.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int | None, optional): The number of latest periods the line is fitted over,
                at least 2. Defaults to None, which uses every period of the Toolkit's date range.
            num_std_dev (float | list[float] | None, optional): The number or numbers of standard
                deviations of the bands. Defaults to None, which uses 1, 2 and 3.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Standard Deviation Channel.
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
            Linear Regression with a Lower Band k and Upper Band k per number of standard deviations k.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Standard Deviation Channel for each asset in the Toolkit instance.
        - The channel agrees with pandas-ta-classic's tos_stdevall.
        - The line is fitted over the whole period at once, so every value depends on the later prices of the
          period and the channel is redrawn when the date range changes. It describes a period and is not a
          signal that was known at the time, which is why `collect_volatility_indicators` leaves it out.
        - When growth is requested, the method calculates the growth of the Standard Deviation Channel
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

        toolkit.technicals.get_standard_deviation_channel(num_std_dev=[1, 2]).xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Linear Regression |   Lower Band 1 |   Lower Band 2 |   Upper Band 1 |   Upper Band 2 |
        |:-----------|--------------------:|---------------:|---------------:|---------------:|---------------:|
        | 2025-12-24 |             240.281 |        200.328 |        160.375 |        280.234 |        320.188 |
        | 2025-12-26 |             240.38  |        200.426 |        160.473 |        280.333 |        320.286 |
        | 2025-12-29 |             240.478 |        200.525 |        160.572 |        280.431 |        320.385 |
        | 2025-12-30 |             240.577 |        200.623 |        160.67  |        280.53  |        320.483 |
        | 2025-12-31 |             240.675 |        200.722 |        160.769 |        280.628 |        320.582 |
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

        historical_data = self._historical_data[period].loc[
            self._start_date : self._end_date
        ]

        standard_deviation_channel = calculate_per_ticker(
            historical_data,
            [close_column],
            volatility_model.get_standard_deviation_channel,
            window,
            num_std_dev,
        )

        return finalize_dataset(
            dataset=standard_deviation_channel,
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
    def get_signal_to_noise_ratio(
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
        Calculate the Signal-to-Noise Ratio (SNR): the net move of the close over a window divided by the Average
        True Range, which shows how cleanly the price trends.

        The Signal-to-Noise Ratio divides the net move of the close over the window (the signal)
        by the Average True Range over the same window (the noise). A ratio of 5 means the price
        moved five times its average range in `window` periods, a market that trends cleanly,
        while a ratio near 0 means the price moved a lot but went nowhere. Skeggs and Hill use it
        to measure how much trend a market offers a trend follower.

        The formula is as follows:

        - Signal = |Close(t) — Close(t — window)|
        - Noise = ATR(window), Wilder's Smoothed Moving Average of the True Range
        - SNR = Signal / Noise

        Because the ATR includes the gaps between periods, the noise covers the overnight moves a
        close-to-close measure would miss. The ratio says how far the price trended, not in
        which direction.

        For more information, see the following sources:

        - Skeggs, J. and Hill, A. (2015). "Back in Black Part 2: The Opportunity Set for Trend
          Following." Man AHL.

        Referenced against:
        - TTR (R), SNR: https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/SNR.R#L47

        Also known as: SNR, signal to noise ratio, trend strength ratio.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted for
                dividends in the same way.
            window (int, optional): The number of periods of the move and of the Average True
                Range. Defaults to 14, the default window of the Average True Range.
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
            pd.Series or pd.DataFrame: Signal-to-Noise Ratio values, in multiples of the Average
            True Range.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Signal-to-Noise Ratio for each asset in the Toolkit instance.
        - The ATR is that of `get_average_true_range`, whose first True Range is High — Low (the
          StockCharts convention). TTR's ATR starts one period later, so the ratio equals TTR's
          SNR once the seeds of the averages have converged (within 0.01% after about 100
          periods with the default window).
        - A window without any range (an ATR of 0) has no ratio.
        - When growth is requested, the method calculates the growth of the Signal-to-Noise
          Ratio over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_signal_to_noise_ratio().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 1.4877 | 0.8712 |      1.1787 |
        | 2025-12-26 | 1.201  | 0.5828 |      1.025  |
        | 2025-12-29 | 0.9597 | 0.5194 |      0.9867 |
        | 2025-12-30 | 0.994  | 0.6213 |      0.9957 |
        | 2025-12-31 | 1.744  | 0.7096 |      0.6057 |
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

        signal_to_noise_ratio = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            volatility_model.get_signal_to_noise_ratio,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=signal_to_noise_ratio,
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
