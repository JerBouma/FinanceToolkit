"""Statistics Module"""

__docformat__ = "google"

from typing import Literal

import pandas as pd

from financetoolkit.econometrics.helpers import BENCHMARK_TICKER, calculate_per_series
from financetoolkit.econometrics.statistics import entropy_model, statistics_model
from financetoolkit.helpers import calculate_per_window, handle_portfolio
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class StatisticsEconometrics:
    """
    Rolling statistics of a series: the rolling linear regression and slope, the correlation with another series, the median, percentile, percent rank, z-score, mean absolute deviation, standard error and entropy, and the entropies and complexity of a series: sample, approximate, permutation, spectral, SVD, Shannon, Lempel-Ziv and Kontoyiannis entropy, the Lempel-Ziv complexity and the Hjorth parameters.
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
        - The columns agree with TA-Lib's LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE and TSF.
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
        - The slope agrees with pandas-ta-classic's slope and the angle its slope with as_angle=True and
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
        - The correlation agrees with TA-Lib's CORREL and pandas-ta-classic's correl, which take the series as
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
        - The median agrees with TA-Lib's MEDIAN, pandas-ta-classic's median and finta's SMM (whose default window is
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
        - The nearest rank agrees with TA-Lib's PERCENTILE, and the linear interpolation pandas-ta-classic's
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
        - The Percent Rank agrees with TA-Lib's PERCENTRANK divided by 100: a decimal between 0 and 1 where TA-Lib
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
        - The Z-Score agrees with pandas-ta-classic's zscore; its cdl_z applies the same score to the open, high, low
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
        - The deviation agrees with TA-Lib's AVGDEV, pandas-ta-classic's mad and md (whose default window is 30) and
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
        - The standard error agrees with pandas-ta-classic's stderr. Tulip Indicators' stderr uses the population
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
        - The entropy agrees with pandas-ta-classic's entropy.
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

    @handle_portfolio
    @handle_errors
    def get_entropy(
        self,
        method: Literal[
            "sample",
            "approximate",
            "permutation",
            "spectral",
            "svd",
            "shannon",
            "lempel_ziv",
            "kontoyiannis",
        ] = "sample",
        period: str | None = None,
        column: str = "Return",
        window: int | None = None,
        order: int | None = None,
        delay: int = 1,
        tolerance: float = 0.2,
        normalize: bool = True,
        estimator: str = "periodogram",
        encoding: str = "binary",
        bins: int = 10,
        word_length: int = 1,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Entropy of a series, by default the returns: how irregular and
        unpredictable it is, from the sample entropy of Richman and Moorman to the entropies of
        encoded returns of López de Prado.

        An entropy is low for a series that repeats itself, trends or is driven by a few
        cycles, and high for a series that looks like noise. On daily returns, which are close
        to noise, most entropies sit near their maximum, so they are mainly of use to compare
        assets or periods with each other (e.g. with `window` for a rolling value) rather than
        as an absolute level.

        The methods are as follows (with the default order m = 2 for the sample and
        approximate entropy and n = 3 for the permutation and SVD entropy):

        - Sample (Richman and Moorman, 2000): SampEn = -ln(A / B), with B the number of pairs
          of templates of m values within a Chebyshev distance of r = tolerance * Std(x) of
          each other (a template not counted with itself) and A the same for m + 1 values.
        - Approximate (Pincus, 1991): ApEn = Phi(m) - Phi(m + 1), Phi(m) = Mean(ln C_i(m)),
          C_i(m) the share of templates within r of template i, itself included.
        - Permutation (Bandt and Pompe, 2002): H = -Sum(p(pi) * log2(p(pi))) over the ordinal
          patterns pi of n values `delay` apart, divided by log2(n!) when normalized.
        - Spectral (Inouye et al., 1991): H = -Sum(p(f) * log2(p(f))), p(f) the power spectrum
          (the periodogram or Welch's estimate) as shares of the total, divided by log2 of the
          number of frequencies when normalized.
        - SVD (Roberts, Penny and Rezek, 1999): H = -Sum(s_k * log2(s_k)), s_k the singular
          values of the delay embedding of n values as shares of their sum, divided by log2(n)
          when normalized.
        - Shannon: the plug-in estimate -Sum(p(w) * log2(p(w))) / L over the words w of L
          symbols (`word_length`) of the encoded returns, in bits per symbol.
        - Lempel-Ziv: the number of distinct phrases of the Lempel-Ziv (1978) parsing of the
          encoded returns divided by their number.
        - Kontoyiannis (Kontoyiannis et al., 1998): Mean(log2(i + 1) / L_i) over the first half
          of the encoded returns, L_i one more than the longest match of the symbols from
          position i with the i symbols before it, in bits per symbol.

        The last three encode the returns into symbols first (`encoding`): "binary" (1 for a
        rise, 0 for a fall, returns of 0 left out), "quantile" (the decile of the return with
        bins=10) or "sigma" (the bucket of one standard deviation the return falls in).

        For more information, see the following sources:

        - Richman, J.S. & Moorman, J.R. (2000). "Physiological time-series analysis using
          approximate entropy and sample entropy." American Journal of Physiology - Heart and
          Circulatory Physiology, 278(6), H2039-H2049.
        - Pincus, S.M. (1991). "Approximate entropy as a measure of system complexity."
          Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
        - Bandt, C. & Pompe, B. (2002). "Permutation entropy: a natural complexity measure for
          time series." Physical Review Letters, 88(17), 174102.
        - Inouye, T. et al. (1991). "Quantification of EEG irregularity by use of the entropy of
          the power spectrum." Electroencephalography and Clinical Neurophysiology, 79(3),
          204-210.
        - Roberts, S.J., Penny, W. & Rezek, I. (1999). "Temporal and spatial complexity
          measures for electroencephalogram based brain-computer interfacing." Medical &
          Biological Engineering & Computing, 37(1), 93-98.
        - Kontoyiannis, I., Algoet, P.H., Suhov, Y.M. & Wyner, A.J. (1998). "Nonparametric
          entropy estimation for stationary processes and random fields, with applications to
          English text." IEEE Transactions on Information Theory, 44(3), 1319-1327.
        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 18.
        - Referenced against nolds' sampen (https://github.com/CSchoel/nolds), antropy's
          sample_entropy, app_entropy, perm_entropy, spectral_entropy and svd_entropy
          (https://github.com/raphaelvallat/antropy).

        Also known as: SampEn, ApEn, ordinal pattern entropy, spectral entropy, SVD entropy,
        plug-in entropy, LZ entropy, Kontoyiannis entropy, entropy features.

        Args:
            method (str, optional): The entropy, "sample", "approximate", "permutation",
                "spectral", "svd", "shannon", "lempel_ziv" or "kontoyiannis". Defaults to "sample".
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the entropy is calculated on:
                "Return" for the returns (the changes of the series), "Adj Close" for the prices
                (the series as given with `Econometrics(data=...)`) or "Close" for the traded
                prices. Defaults to "Return".
            window (int | None, optional): The number of periods of a rolling window. Defaults to
                None, which calculates one value over the full history of every series.
            order (int | None, optional): The template length m of the sample and approximate
                entropy or the number of values n of the permutation and SVD entropy. Defaults to
                None, which uses 2 for the sample and approximate and 3 for the permutation and
                SVD entropy.
            delay (int, optional): The number of periods between the values of a pattern of the
                permutation and SVD entropy. Defaults to 1.
            tolerance (float, optional): The tolerance r of the sample and approximate entropy
                as a multiple of the standard deviation (ddof=1). Defaults to 0.2.
            normalize (bool, optional): Whether to divide the permutation, spectral and SVD
                entropy by their maximum, so they lie between 0 and 1. Defaults to True.
            estimator (str, optional): The power spectrum of the spectral entropy, "periodogram"
                or "welch" (segments of 256 periods with a Hann window). Defaults to "periodogram".
            encoding (str, optional): The encoding of the Shannon, Lempel-Ziv and Kontoyiannis
                entropy, "binary", "quantile" or "sigma". Defaults to "binary".
            bins (int, optional): The number of quantiles of the "quantile" encoding. Defaults to 10.
            word_length (int, optional): The number of symbols per word of the Shannon entropy.
                Defaults to 1.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame:
            The entropy per asset, or with `window` per date (a column per asset). The sample,
            approximate and Lempel-Ziv entropies have no unit; the others are in bits (per symbol),
            or between 0 and 1 when normalized.

        Notes:
        - Each asset is measured over the dates it has a value. With a window, the first value
          needs a full window of data from the start date.
        - The sample and approximate entropy compare every template with every other one, so
          their time grows with the square of the number of observations: about 0.1 seconds for
          ten years of daily returns, and about 2 seconds for a rolling window of a year over ten
          years (6 seconds for a window of two years).
        - The sample entropy is NaN when no two templates match (a very short series), where
          nolds and antropy give infinity. The tolerance uses the sample standard deviation
          (ddof=1) and includes a distance of exactly r, as in Richman and Moorman.
        - The Shannon entropy counts every overlapping word, the last one included.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT", "NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_entropy(method="permutation", window=252).tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   NVDA |
        |:-----------|-------:|-------:|-------:|
        | 2025-12-24 | 0.9984 | 0.9911 | 0.9997 |
        | 2025-12-26 | 0.998  | 0.9919 | 0.9995 |
        | 2025-12-29 | 0.9982 | 0.9919 | 0.9993 |
        | 2025-12-30 | 0.998  | 0.9921 | 0.9992 |
        | 2025-12-31 | 0.9981 | 0.9919 | 0.9992 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        if method in ["sample", "approximate"]:
            function = (
                entropy_model.get_sample_entropy
                if method == "sample"
                else entropy_model.get_approximate_entropy
            )
            options = {"order": order or 2, "tolerance": tolerance}
        elif method in ["permutation", "svd"]:
            function = (
                entropy_model.get_permutation_entropy
                if method == "permutation"
                else entropy_model.get_svd_entropy
            )
            options = {"order": order or 3, "delay": delay, "normalize": normalize}
        elif method == "spectral":
            function = entropy_model.get_spectral_entropy
            options = {"estimator": estimator, "normalize": normalize}
        elif method in ["shannon", "lempel_ziv", "kontoyiannis"]:
            estimate = {
                "shannon": lambda symbols: entropy_model.get_shannon_entropy(
                    symbols, word_length=word_length
                ),
                "lempel_ziv": entropy_model.get_lempel_ziv_entropy,
                "kontoyiannis": entropy_model.get_kontoyiannis_entropy,
            }[method]

            def function(values):
                return estimate(entropy_model.encode_series(values, encoding, bins))

            options = {}
        else:
            raise ValueError(
                "Method should be 'sample', 'approximate', 'permutation', 'spectral', 'svd', "
                "'shannon', 'lempel_ziv' or 'kontoyiannis'."
            )

        entropy = calculate_per_window(data, function, window, **options)

        return apply_rounding(
            entropy, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_lempel_ziv_complexity(
        self,
        period: str | None = None,
        column: str = "Return",
        window: int | None = None,
        encoding: str = "binary",
        bins: int = 10,
        normalize: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Lempel-Ziv Complexity of a series, by default of the direction of the
        returns: the number of new patterns needed to write the series down.

        The returns are encoded into symbols (by default 1 for a rise and 0 for a fall) and
        parsed from the start into phrases, each the longest stretch that can be copied from
        earlier in the series plus one new symbol. A series that repeats itself needs few
        phrases, a random one many. Normalized, the complexity approaches 1 for independent,
        equally likely symbols and falls towards 0 for a regular series, so on daily returns it
        is mostly close to 1 and of use to compare assets or periods with each other.

        The formula is as follows:

        - c(N) = the number of phrases of the Lempel-Ziv (1976) parsing of the N symbols
        - Normalized c(N) = c(N) * log_k(N) / N

        with k the number of possible symbols: 2 for the "binary" encoding, `bins` for the
        "quantile" encoding and the number of distinct symbols for the "sigma" encoding.

        For more information, see the following sources:

        - Lempel, A. & Ziv, J. (1976). "On the complexity of finite sequences." IEEE Transactions
          on Information Theory, 22(1), 75-81.
        - Kaspar, F. & Schuster, H.G. (1987). "Easily calculable measure for the complexity of
          spatiotemporal patterns." Physical Review A, 36(2), 842-848.
        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 18.
        - Referenced against antropy's lziv_complexity (https://github.com/raphaelvallat/antropy).

        Also known as: LZ complexity, LZ76, Lempel-Ziv-Welch complexity, LZC.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column that is encoded: "Return" for the
                returns (the changes of the series), "Adj Close" for the prices (the series as given
                with `Econometrics(data=...)`) or "Close" for the traded prices. Defaults to "Return".
            window (int | None, optional): The number of periods of a rolling window. Defaults to
                None, which calculates one value over the full history of every series.
            encoding (str, optional): "binary" (1 for a positive and 0 for a negative value, values of
                0 left out), "quantile" (the quantile of the value) or "sigma" (the bucket of one
                standard deviation the value falls in). Defaults to "binary".
            bins (int, optional): The number of quantiles of the "quantile" encoding. Defaults to 10.
            normalize (bool, optional): Whether to normalize the number of phrases by N / log_k(N).
                Defaults to True.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series | pd.DataFrame:
            The (normalized) complexity per asset, or with `window` per date (a column per asset).

        Notes:
        - Each asset is measured over the dates it has a value. With a window, the first value
          needs a full window of data from the start date.
        - It equals antropy's lziv_complexity of the same symbols; antropy leaves the encoding to
          the user (often above or below the median, which for returns is close to the sign).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT", "NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_lempel_ziv_complexity(window=252).tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   NVDA |
        |:-----------|-------:|-------:|-------:|
        | 2025-12-24 | 1.0763 | 1.1396 | 1.1396 |
        | 2025-12-26 | 1.0763 | 1.108  | 1.1396 |
        | 2025-12-29 | 1.0763 | 1.108  | 1.1396 |
        | 2025-12-30 | 1.0763 | 1.108  | 1.1396 |
        | 2025-12-31 | 1.0763 | 1.108  | 1.1396 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )
        alphabet = {"binary": 2, "quantile": bins}.get(encoding)

        def complexity(values):
            return entropy_model.get_lempel_ziv_complexity(
                entropy_model.encode_series(values, encoding, bins),
                normalize=normalize,
                alphabet=alphabet,
            )

        lempel_ziv_complexity = calculate_per_window(data, complexity, window)

        return apply_rounding(
            lempel_ziv_complexity,
            rounding if rounding is not None else self._rounding,
        )

    @handle_portfolio
    @handle_errors
    def get_hjorth_parameters(
        self,
        period: str | None = None,
        column: str = "Return",
        window: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Hjorth parameters of a series, by default the returns: its Activity,
        Mobility and Complexity, and the number of times it crosses zero.

        The Activity is the variance. The Mobility, the standard deviation of the changes
        relative to that of the series, is a mean frequency: it is about Sqrt(2) = 1.41 for
        independent returns, lower when returns persist and higher when they alternate. The
        Complexity compares the Mobility of the changes with that of the series and grows as
        the series looks less like a single smooth wave. The Zero Crossings count how often the
        returns change sign.

        The formula is as follows:

        - Activity = Var(x)
        - Mobility = Sqrt(Var(dx) / Var(x))
        - Complexity = Sqrt(Var(d2x) / Var(dx)) / Mobility
        - Zero Crossings = the number of sign changes of x, values of exactly 0 skipped

        with dx the first and d2x the second differences and Var the sample variance (ddof=1).

        For more information, see the following sources:

        - Hjorth, B. (1970). "EEG analysis based on time domain properties."
          Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
        - Referenced against antropy's hjorth_params and num_zerocross
          (https://github.com/raphaelvallat/antropy).

        Also known as: Hjorth activity, Hjorth mobility, Hjorth complexity, zero-crossing rate.

        Args:
            period (str | None, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to None, which uses "daily".
            column (str, optional): The historical data column the parameters are calculated on:
                "Return" for the returns (the changes of the series), "Adj Close" for the prices
                (the series as given with `Econometrics(data=...)`) or "Close" for the traded
                prices. Defaults to "Return".
            window (int | None, optional): The number of periods of a rolling window. Defaults to
                None, which calculates the parameters over the full history of every series.
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
            The Activity, Mobility, Complexity and Zero Crossings per asset (a row per asset), or
            with `window` per date ((parameter, asset) columns).

        Notes:
        - Each asset is measured over the dates it has a value. With a window, the first value
          needs a full window of data from the start date.
        - antropy uses the population variance (ddof=0), which changes the Mobility and
          Complexity by less than 0.1% for 1,000 observations, and counts a value of 0 as
          positive in its zero crossings.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT", "NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_hjorth_parameters()
        ```

        Which returns:

        |      |   Activity |   Mobility |   Complexity |   Zero Crossings |
        |:-----|-----------:|-----------:|-------------:|-----------------:|
        | AAPL |     0.0003 |     1.4058 |       1.228  |              611 |
        | MSFT |     0.0003 |     1.4267 |       1.2057 |              625 |
        | NVDA |     0.0011 |     1.4484 |       1.2098 |              619 |
        """
        period = period if period else "daily"

        data = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        hjorth_parameters = calculate_per_window(
            data, entropy_model.get_hjorth_parameters, window
        )

        return apply_rounding(
            hjorth_parameters, rounding if rounding is not None else self._rounding
        )
