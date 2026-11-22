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
    Volatility indicators: the Bollinger Bands, %B and BandWidth, the true range and average true range, the mass index, the Supertrend, the Chandelier Exit, the Keltner and Donchian channels and the volatility cone.
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
                the Supertrend its own defaults.
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

        | Date       |   Bollinger Band Upper |   Bollinger Band Middle |   Bollinger Band Lower |   Bollinger %B |   Bollinger BandWidth |   True Range |   Average True Range |   Supertrend |   Supertrend Trend Direction |   Keltner Channel Upper |   Keltner Channel Middle |   Keltner Channel Lower |   Donchian Channel Upper |   Donchian Channel Middle |   Donchian Channel Lower |
        |:-----------|-----------------------:|------------------------:|-----------------------:|---------------:|----------------------:|-------------:|---------------------:|-------------:|-----------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|-------------------------:|
        | 2025-12-17 |                285.548 |                 276     |                266.452 |         0.2327 |                0.0692 |       4.5043 |               4.871  |      268.465 |                            1 |                 284.794 |                  275.314 |                 265.833 |                  287.618 |                   279.232 |                  270.846 |
        | 2025-12-18 |                285.211 |                 276.181 |                267.151 |         0.2267 |                0.0654 |       6.6568 |               4.9986 |      268.465 |                            1 |                 284.635 |                  274.771 |                 264.907 |                  287.618 |                   279.158 |                  270.697 |
        | 2025-12-19 |                284.285 |                 276.55  |                268.816 |         0.2524 |                0.0559 |       4.6837 |               4.9761 |      268.465 |                            1 |                 284.312 |                  274.498 |                 264.683 |                  287.618 |                   276.821 |                  266.023 |
        | 2025-12-22 |                284.343 |                 276.525 |                268.707 |         0.0846 |                0.0565 |       3.3583 |               4.8605 |      268.465 |                            1 |                 283.406 |                  273.902 |                 264.397 |                  287.618 |                   276.821 |                  266.023 |
        | 2025-12-23 |                284.455 |                 276.347 |                268.24  |         0.1958 |                0.0587 |       2.9298 |               4.7226 |      268.465 |                            1 |                 282.71  |                  273.57  |                 264.43  |                  287.618 |                   276.821 |                  266.023 |
        | 2025-12-24 |                284.438 |                 276.19  |                267.941 |         0.2981 |                0.0597 |       3.2188 |               4.6152 |      268.465 |                            1 |                 282.345 |                  273.476 |                 264.606 |                  283.741 |                   274.882 |                  266.023 |
        | 2025-12-26 |                284.387 |                 275.983 |                267.579 |         0.2899 |                0.0609 |       2.5013 |               4.4642 |      268.465 |                            1 |                 281.822 |                  273.339 |                 264.856 |                  280.164 |                   273.094 |                  266.023 |
        | 2025-12-29 |                284.195 |                 275.729 |                267.264 |         0.3276 |                0.0614 |       2.003  |               4.2884 |      268.465 |                            1 |                 281.304 |                  273.268 |                 265.233 |                  279.177 |                   272.6   |                  266.023 |
        | 2025-12-30 |                283.299 |                 275.23  |                267.162 |         0.308  |                0.0586 |       1.7937 |               4.1102 |      268.465 |                            1 |                 280.707 |                  273.117 |                 265.526 |                  279.177 |                   272.6   |                  266.023 |
        | 2025-12-31 |                281.366 |                 274.516 |                267.666 |         0.2372 |                0.0499 |       1.9233 |               3.954  |      268.465 |                            1 |                 280.039 |                  272.823 |                 265.607 |                  279.177 |                   272.6   |                  266.023 |
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
        Bollinger himself specifies and as TA-Lib and StockCharts both implement, not
        pandas' default sample standard deviation.

        The band width tracks volatility: bands that narrow sharply (a squeeze) often precede a
        larger move, without saying in which direction. A close at or beyond a band is not a sell
        or buy signal by itself, since prices can walk along a band in a trend. The defaults are
        Bollinger's own setting, a 20-period average with 2 standard deviations, as on most
        charting platforms.

        For more information, see the following sources:

        - Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill.
        - TA-Lib: BBANDS, the reference implementation (https://ta-lib.org/functions/bbands.html).
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

        - TA-Lib: PERCENTB and BBW, the reference implementations (https://ta-lib.org/functions/percentb.html,
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

        - TA-Lib: TRANGE, the reference implementation (https://ta-lib.org/functions/trange.html).

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

        - TA-Lib: ATR, the reference implementation (https://ta-lib.org/functions/atr.html).

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

        - TA-Lib: MASSI, the reference implementation (https://ta-lib.org/functions/massi.html).

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

        - TA-Lib: SUPERTREND, the reference implementation (https://ta-lib.org/functions/supertrend.html).

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

        See `volatility_model.get_chandelier_exit` for the references.

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

        For more information, see the following sources:

        - TA-Lib: KC, the reference implementation (https://ta-lib.org/functions/kc.html).

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
            atr_multiplier (float, optional): Multiplier for ATR to determine channel width.
                Defaults to 2.
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

        - TA-Lib: DONCHIAN, the reference implementation (https://ta-lib.org/functions/donchian.html).

        Also known as: Donchian Channels, price channel breakout.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the Donchian Channels.
                Defaults to 20.
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
          previous period.
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
