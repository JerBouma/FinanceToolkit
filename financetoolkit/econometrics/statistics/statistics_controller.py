"""Statistics Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.econometrics.helpers import BENCHMARK_TICKER, calculate_per_series
from financetoolkit.econometrics.statistics import statistics_model
from financetoolkit.helpers import handle_portfolio
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class StatisticsEconometrics:
    """
    Rolling statistics of a series: the rolling linear regression and slope, the correlation with another series, the median, percentile, percent rank, z-score, mean absolute deviation, standard error and entropy.
    """

    @handle_portfolio
    @handle_errors
    def get_linear_regression(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 14,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the rolling Linear Regression of a price series: the end point of the
        least-squares line over the window, its slope, intercept and angle, and the Time Series
        Forecast one period ahead.

        Over every window, a straight line is fitted through the prices by ordinary least
        squares. Its end point is a moving average that lags less than a simple average, as it
        follows the trend of the window rather than its middle (Chande and Kroll call it the
        linear regression indicator), the slope is the trend in price per period, and the Time
        Series Forecast extends the line by one period. A price above a rising regression line
        is read as a confirmed uptrend, and the slope turning negative as the trend of the
        window turning down.

        The formula is as follows, with time x = 0, ..., window - 1 from the oldest price of the
        window to the latest:

        - Slope (b1) = Sum((x - Mean(x)) * (Close - Mean(Close))) / Sum((x - Mean(x))^2)
        - Intercept (b0) = Mean(Close) - b1 * Mean(x), the line at the oldest period of the window
        - Linear Regression = b0 + b1 * (window - 1), the line at the latest period
        - Time Series Forecast = b0 + b1 * window, the line one period ahead
        - Angle = arctan(b1) in degrees

        For more information, see the following sources:

        - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.
        - Referenced against TA-Lib's LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE and TSF
          (https://ta-lib.org/functions/linearreg.html).
        - pandas-ta-classic: linreg (https://github.com/xgboosted/pandas-ta-classic).
        - Tulip Indicators: linreg, linregslope, linregintercept and tsf
          (https://tulipindicators.org/linreg).

        Also known as: LINEARREG, linear regression indicator, end-point moving average, least squares moving average,
        LSMA, TSF, time series forecast, linear regression slope, intercept, angle.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of periods the line is fitted over, at least 2.
                Defaults to 14.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Linear Regression, Slope, Intercept, Angle and Time Series Forecast.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Linear Regression for each
          asset, each over the dates it has a value. The first value needs a full window of data from the start date.
        - The columns equal TA-Lib's LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE and TSF.
        - The slope is in price per period and the angle depends on the scale of the prices, so neither compares
          across securities with different share prices; the Correlation Trend Indicator
          (`toolkit.technicals.get_correlation_trend_indicator`) does.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_linear_regression().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |    Angle |   Intercept |   Linear Regression |   Slope |   Time Series Forecast |
        |:-----------|---------:|------------:|--------------------:|--------:|-----------------------:|
        | 2025-12-24 | -30.142  |     277.998 |             270.449 | -0.5807 |                269.868 |
        | 2025-12-26 | -27.157  |     277.175 |             270.506 | -0.513  |                269.993 |
        | 2025-12-29 | -23.7844 |     276.411 |             270.682 | -0.4407 |                270.241 |
        | 2025-12-30 | -21.4583 |     275.809 |             270.7   | -0.3931 |                270.306 |
        | 2025-12-31 | -17.2038 |     274.774 |             270.749 | -0.3096 |                270.44  |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        linear_regression = calculate_per_series(
            data, statistics_model.get_linear_regression, window
        )

        return apply_rounding(
            linear_regression, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_slope(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 1,
        as_angle: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Slope of a price series: the average change in price per period over a
        window, the rise over the run between the price `window` periods ago and the latest
        price.

        The formula is as follows:

        - Slope = (Close - Close(t - window)) / window
        - Angle = arctan(Slope) in degrees

        The slope over one period is the change from the previous close. Only the two ends of
        the window count: the least-squares slope, which uses every price of the window, is the
        Slope of `get_linear_regression`.

        For more information, see the following sources:

        - Referenced against pandas-ta-classic's slope
          (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: slope, rise over run, average change per period, angle.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of periods between the two prices. Defaults to 1.
            as_angle (bool, optional): Whether to return the angle of the slope in degrees instead
                of the slope. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Slope values, in price per period, or in degrees with `as_angle`.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Slope for each asset, each
          over the dates it has a value. The first value needs a full window of data from the start date.
        - The slope equals pandas-ta-classic's slope and the angle its slope with as_angle=True and
          to_degrees=True (pandas-ta-classic gives radians unless asked for degrees).
        - The slope is in price per period, so it does not compare across securities with different share
          prices.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_slope(window=5).tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |
        |:-----------|--------:|--------:|
        | 2025-12-24 |  0.3926 |  2.3604 |
        | 2025-12-26 |  0.2412 |  0.7399 |
        | 2025-12-29 |  0.0179 |  0.2341 |
        | 2025-12-30 |  0.4205 |  0.5078 |
        | 2025-12-31 | -0.0997 | -0.6407 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        slope = calculate_per_series(data, statistics_model.get_slope, window, as_angle)

        return apply_rounding(
            slope, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_rolling_correlation(
        self,
        period: str | None = None,
        column: str = "Return",
        window: int = 30,
        against: str | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the rolling Pearson Correlation of every series with another over a window:
        with the benchmark, another ticker of the Toolkit (or series of the data) or another
        price column of the same ticker, such as the high against the low as in TA-Lib.

        The correlation is +1 when the two series move up and down together in a straight-line
        relation, -1 when they move in opposite directions and around 0 when they are unrelated.
        By default it correlates the returns, so it shows whether the moves of two assets go
        together over the window. On price levels (column="Adj Close"), two series that both
        trend have a high correlation even when their daily moves are unrelated, so it then shows
        whether two prices follow the same course over the window rather than whether their
        returns are related.

        The formula is as follows:

        - Correlation = Covariance(X, Y, window) / (StdDev(X, window) * StdDev(Y, window))

        with X the `column` series of each asset and Y that of `against`.

        For more information, see the following sources:

        - Pearson, K. (1895). "Notes on Regression and Inheritance in the Case of Two Parents."
          Proceedings of the Royal Society of London, 58, 240-242.
        - Referenced against TA-Lib's CORREL (https://ta-lib.org/functions/correl.html).
        - pandas-ta-classic: correl (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: CORREL, Pearson correlation coefficient, rolling correlation, correlation with the benchmark.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column that is correlated: "Return" for the
                returns (the changes of the series), "Adj Close" for the prices (the series as given
                with `Econometrics(data=...)`) or another price column such as "High". Defaults to
                "Return".
            window (int, optional): The number of periods. Defaults to 30.
            against (str | None, optional): What to correlate with: a ticker of the Toolkit (or a
                series of the data), such as "Benchmark" or "MSFT", or another column of the same
                ticker, such as "Low" with column="High". Defaults to None, which uses the
                Benchmark or, without one, the second series.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Correlation values between -1 and 1, a column per asset (the series it is correlated
            with is left out, as its correlation with itself is 1).

        Notes:
        - The method takes the `column` series of the specified `period` and correlates each asset with
          `against` over the dates both have a value, so a cryptocurrency's weekends do not count against
          a stock.
        - The correlation equals TA-Lib's CORREL and pandas-ta-classic's correl, which take the series as
          given; TA-Lib's default compares the high with the low (column="High", against="Low").
        - A window in which either series is flat has no correlation and is missing, where TA-Lib and pandas-ta-
          classic give 0.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rolling_correlation().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |
        |:-----------|-------:|-------:|
        | 2025-12-24 | 0.4834 | 0.3484 |
        | 2025-12-26 | 0.4882 | 0.3495 |
        | 2025-12-29 | 0.5123 | 0.2947 |
        | 2025-12-30 | 0.5132 | 0.3062 |
        | 2025-12-31 | 0.4705 | 0.315  |
        """
        period = period if period else "daily"

        data = self._get_series_column(period, column)
        series = list(data.columns)

        if against is None:
            against = (
                BENCHMARK_TICKER
                if BENCHMARK_TICKER in series
                else series[1] if len(series) > 1 else None
            )

        if against in series:
            # The same series of another asset, e.g. the returns of the Benchmark.
            others = {name: data[against] for name in series if name != against}
        elif against is not None and against in self._historical_data[
            period
        ].columns.get_level_values(0):
            # Another column of the same asset, e.g. the low against the high.
            other_data = self._get_series_column(period, against)
            others = {name: other_data[name] for name in series}
        else:
            raise ValueError(
                f"{against} is neither a series of the data nor one of its columns; choose "
                f"from {', '.join(series)}."
            )

        rolling_correlation = {}
        for name, other in others.items():
            # Over the dates both have a value, so neither counts the other's holidays.
            pair = pd.concat([data[name], other], axis=1).dropna()
            rolling_correlation[name] = statistics_model.get_rolling_correlation(
                pair.iloc[:, 0], pair.iloc[:, 1], window
            ).reindex(data.index)

        rolling_correlation = self._filter_benchmark(
            pd.DataFrame(rolling_correlation, index=data.index),
            include_benchmark=include_benchmark,
        )

        return apply_rounding(
            rolling_correlation, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_rolling_median(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 30,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the rolling Median of a price series: the middle price of the window.

        The median is a moving average that a single outlier, such as one day with a spike,
        does not pull along, as it depends only on the order of the prices.

        The formula is as follows:

        - Median = the middle value of the sorted prices of the window, or the mean of the two
          middle values for an even window

        For more information, see the following sources:

        - Tukey, J.W. (1977). "Exploratory Data Analysis." Addison-Wesley, for running medians.
        - Referenced against TA-Lib's MEDIAN (https://ta-lib.org/functions/median.html).
        - pandas-ta-classic: median (https://github.com/xgboosted/pandas-ta-classic).
        - finta: SMM (https://github.com/peerchemist/finta).

        Also known as: MEDIAN, moving median, simple moving median, SMM.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of periods. Defaults to 30.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Median values.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Median for each asset, each
          over the dates it has a value. The first value needs a full window of data from the start date.
        - The median equals TA-Lib's MEDIAN, pandas-ta-classic's median and finta's SMM (whose default window is
          9).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rolling_median().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |
        |:-----------|--------:|--------:|
        | 2025-12-24 | 273.408 | 481.221 |
        | 2025-12-26 | 273.408 | 481.221 |
        | 2025-12-29 | 273.408 | 481.221 |
        | 2025-12-30 | 273.408 | 481.221 |
        | 2025-12-31 | 273.408 | 480.468 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        rolling_median = calculate_per_series(
            data, statistics_model.get_rolling_median, window
        )

        return apply_rounding(
            rolling_median, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_rolling_percentile(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 100,
        percentile: float = 0.5,
        method: str = "nearest_rank",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the rolling Percentile of a price series: the price below which a given share
        of the prices of the window lies, such as the median (0.5) or the upper quartile (0.75).

        The formula is as follows, per method:

        - "nearest_rank": the k-th smallest price of the window, k = ceil(percentile * window)
          (Hyndman and Fan's definition 1)
        - "linear": interpolated between the two prices around position percentile * (window - 1)
          of the sorted window, counting from 0 (Hyndman and Fan's definition 7)

        For more information, see the following sources:

        - Hyndman, R.J. & Fan, Y. (1996). "Sample Quantiles in Statistical Packages." The
          American Statistician, 50(4), 361-365.
        - Referenced against TA-Lib's PERCENTILE
          (https://ta-lib.org/functions/percentile.html).
        - pandas-ta-classic: quantile (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: PERCENTILE, quantile, rolling quantile, moving percentile.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of periods. Defaults to 100.
            percentile (float, optional): The percentile as a decimal between 0 and 1. Defaults
                to 0.5.
            method (str, optional): "nearest_rank" or "linear". Defaults to "nearest_rank".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Percentile values.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Percentile for each asset,
          each over the dates it has a value. The first value needs a full window of data from the start date.
        - The nearest rank equals TA-Lib's PERCENTILE, and the linear interpolation pandas-ta-classic's
          quantile.
        - The percentile is a decimal (0.5 is the median) where TA-Lib takes a percentage.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rolling_percentile(percentile=0.9).tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |
        |:-----------|--------:|--------:|
        | 2025-12-24 | 276.586 | 518.336 |
        | 2025-12-26 | 276.586 | 517.138 |
        | 2025-12-29 | 276.586 | 516.368 |
        | 2025-12-30 | 276.586 | 516.368 |
        | 2025-12-31 | 276.586 | 516.368 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        rolling_percentile = calculate_per_series(
            data, statistics_model.get_rolling_percentile, window, percentile, method
        )

        return apply_rounding(
            rolling_percentile, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_percent_rank(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 100,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Percent Rank of a price series: the share of the previous `window` prices
        that are lower than the latest price.

        A Percent Rank of 1 means the price is above every price of the window, a new high for
        the window, and 0 that it is at or below all of them. It ranks the price within its
        recent history, and Connors uses it in ConnorsRSI to rank the latest return.

        The formula is as follows:

        - Percent Rank = Count(Close(t - i) < Close(t), i = 1, ..., window) / window

        For more information, see the following sources:

        - Connors, L., Alvarez, C. & Radtke, M. (2012). "An Introduction to ConnorsRSI."
          Connors Research Trading Strategy Series.
        - Referenced against TA-Lib's PERCENTRANK
          (https://ta-lib.org/functions/percentrank.html).

        Also known as: PERCENTRANK, percentile rank, percent rank.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of previous periods. Defaults to 100.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Percent Rank values as decimals.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Percent Rank for each asset,
          each over the dates it has a value. The first value needs a full window of data from the start date.
        - The Percent Rank equals TA-Lib's PERCENTRANK divided by 100: a decimal between 0 and 1 where TA-Lib
          gives a percentage.
        - The latest price is compared with the `window` prices before it, so the first value needs `window` + 1
          prices.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_percent_rank().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |
        |:-----------|-------:|-------:|
        | 2025-12-24 |   0.83 |   0.2  |
        | 2025-12-26 |   0.8  |   0.2  |
        | 2025-12-29 |   0.82 |   0.2  |
        | 2025-12-30 |   0.78 |   0.21 |
        | 2025-12-31 |   0.73 |   0.13 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        percent_rank = calculate_per_series(
            data, statistics_model.get_percent_rank, window
        )

        return apply_rounding(
            percent_rank, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_rolling_z_score(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 30,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the rolling Z-Score of a price series: the number of standard deviations the
        price lies above or below its moving average.

        A Z-Score of 2 puts the price two standard deviations above its average of the window,
        on the upper Bollinger Band of the same window measured with the sample standard
        deviation. Mean-reversion strategies read large absolute values as stretched prices.

        The formula is as follows:

        - Z-Score = (Close - SMA(Close, window)) / StdDev(Close, window)

        with the sample standard deviation (ddof=1).

        For more information, see the following sources:

        - Referenced against pandas-ta-classic's zscore
          (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: z-score, standard score, ZSCORE, normalized price.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of periods. Defaults to 30.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Z-Score values.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Z-Score for each asset, each
          over the dates it has a value. The first value needs a full window of data from the start date.
        - The Z-Score equals pandas-ta-classic's zscore; its cdl_z applies the same score to the open, high, low
          and close.
        - A window of flat prices has no standard deviation and is missing, where pandas-ta-classic gives 0.
        - The `standardize` parameter instead standardizes the result over the whole date range shown.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rolling_z_score().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |
        |:-----------|--------:|-------:|
        | 2025-12-24 | -0.2834 | 0.1807 |
        | 2025-12-26 | -0.3665 | 0.2499 |
        | 2025-12-29 | -0.2992 | 0.2546 |
        | 2025-12-30 | -0.4439 | 0.4538 |
        | 2025-12-31 | -0.7516 | 0.0311 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        rolling_z_score = calculate_per_series(
            data, statistics_model.get_rolling_z_score, window
        )

        return apply_rounding(
            rolling_z_score, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_rolling_mean_absolute_deviation(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 14,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the rolling Mean Absolute Deviation of a price series: the average distance of
        the prices of the window from their mean.

        It measures the dispersion of the prices like the standard deviation, but without
        squaring the deviations, so a single large deviation weighs less. Lambert's Commodity
        Channel Index divides by it.

        The formula is as follows:

        - Mean Absolute Deviation = Mean(|Close(t - i) - SMA(Close, window)|, i = 0, ..., window - 1)

        with the average of the window itself for every price of the window.

        For more information, see the following sources:

        - Lambert, D.R. (1980). "Commodity Channel Index: Tool for Trading Cyclic Trends."
          Commodities, October 1980.
        - Referenced against TA-Lib's AVGDEV (https://ta-lib.org/functions/avgdev.html).
        - pandas-ta-classic: mad and md (https://github.com/xgboosted/pandas-ta-classic).
        - Tulip Indicators: md (https://tulipindicators.org/md).

        Also known as: AVGDEV, average deviation, mean deviation, MAD, MD.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of periods. Defaults to 14.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Mean Absolute Deviation values, in price units.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Mean Absolute Deviation for
          each asset, each over the dates it has a value. The first value needs a full window of data from the start
          date.
        - The deviation equals TA-Lib's AVGDEV, pandas-ta-classic's mad and md (whose default window is 30) and
          Tulip Indicators' md.
        - `toolkit.risk.get_mean_absolute_deviation` measures the daily returns within each period instead of
          the prices over a rolling window.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rolling_mean_absolute_deviation().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |
        |:-----------|-------:|-------:|
        | 2025-12-24 | 2.5438 | 4.4227 |
        | 2025-12-26 | 2.3046 | 4.653  |
        | 2025-12-29 | 2.0477 | 4.4546 |
        | 2025-12-30 | 1.8328 | 4.2249 |
        | 2025-12-31 | 1.4702 | 3.8222 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        rolling_mean_absolute_deviation = calculate_per_series(
            data, statistics_model.get_rolling_mean_absolute_deviation, window
        )

        return apply_rounding(
            rolling_mean_absolute_deviation,
            rounding if rounding is not None else self._rounding,
        )

    @handle_portfolio
    @handle_errors
    def get_rolling_standard_error(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 14,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the rolling Standard Error of the mean of a price series: the standard
        deviation of the window divided by the square root of the number of periods.

        It is the uncertainty of the moving average as an estimate of the mean price of the
        window: the moving average plus and minus two standard errors spans roughly where that
        mean lies.

        The formula is as follows:

        - Standard Error = StdDev(Close, window) / sqrt(window)

        with the sample standard deviation (ddof=1).

        For more information, see the following sources:

        - Referenced against pandas-ta-classic's stderr
          (https://github.com/xgboosted/pandas-ta-classic).
        - Tulip Indicators: stderr (https://tulipindicators.org/stderr).

        Also known as: STDERR, standard error of the mean, SEM.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of periods. Defaults to 14.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Standard Error values, in price units.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Standard Error for each
          asset, each over the dates it has a value. The first value needs a full window of data from the start date.
        - The standard error equals pandas-ta-classic's stderr. Tulip Indicators' stderr uses the population
          standard deviation (ddof=0), which is sqrt((window - 1) / window) times this value.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rolling_standard_error().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |
        |:-----------|-------:|-------:|
        | 2025-12-24 | 0.7619 | 1.4639 |
        | 2025-12-26 | 0.7182 | 1.4995 |
        | 2025-12-29 | 0.6802 | 1.416  |
        | 2025-12-30 | 0.6541 | 1.2939 |
        | 2025-12-31 | 0.5702 | 1.2523 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        rolling_standard_error = calculate_per_series(
            data, statistics_model.get_rolling_standard_error, window
        )

        return apply_rounding(
            rolling_standard_error, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_rolling_entropy(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int = 10,
        base: float = 2.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the rolling Shannon Entropy of a price series: the entropy of the prices of the
        window taken as shares of their sum.

        The entropy is at its maximum, log(window), when all prices of the window are equal, and
        falls as the prices spread out, so a lower entropy points to a larger move within the
        window. In base 2 it is measured in bits.

        The formula is as follows:

        - P(t - i) = Close(t - i) / Sum(Close, window), i = 0, ..., window - 1
        - Entropy = -Sum(P * log_base(P))

        For more information, see the following sources:

        - Shannon, C.E. (1948). "A Mathematical Theory of Communication." Bell System Technical
          Journal, 27(3), 379-423.
        - Referenced against pandas-ta-classic's entropy
          (https://github.com/xgboosted/pandas-ta-classic).

        Also known as: ENTP, Shannon entropy, information entropy.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the statistic is calculated on: "Adj Close"
                for the prices (the series as given with `Econometrics(data=...)`), "Close" for the
                traded prices or "Return" for the returns (the changes of the series). Defaults to "Adj Close".
            window (int, optional): The number of periods. Defaults to 10.
            base (float, optional): The base of the logarithm. Defaults to 2.0, in bits.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame:
            Entropy values.

        Notes:
        - The method takes the `column` series of the specified `period` and calculates the Entropy for each asset, each
          over the dates it has a value. The first value needs a full window of data from the start date.
        - The entropy equals pandas-ta-classic's entropy.
        - A window with a negative price or a sum of 0 has no entropy.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rolling_entropy().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |
        |:-----------|-------:|-------:|
        | 2025-12-24 | 3.3219 | 3.3219 |
        | 2025-12-26 | 3.3219 | 3.3219 |
        | 2025-12-29 | 3.3219 | 3.3219 |
        | 2025-12-30 | 3.3219 | 3.3219 |
        | 2025-12-31 | 3.3219 | 3.3219 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        rolling_entropy = calculate_per_series(
            data, statistics_model.get_rolling_entropy, window, base
        )

        return apply_rounding(
            rolling_entropy, rounding if rounding is not None else self._rounding
        )
