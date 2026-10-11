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

        | Date       |   Bollinger Band Upper |   Bollinger Band Middle |   Bollinger Band Lower |   True Range |
        |:-----------|-----------------------:|------------------------:|-----------------------:|-------------:|
        | 2025-12-17 |                285.486 |                 277.993 |                270.499 |       4.52   |
        | 2025-12-18 |                285.733 |                 277.518 |                269.304 |       6.68   |
        | 2025-12-19 |                284.913 |                 276.846 |                268.78  |       4.7    |
        | 2025-12-22 |                282.966 |                 275.762 |                268.558 |       3.37   |
        | 2025-12-23 |                281.038 |                 274.922 |                268.807 |       2.94   |
        | 2025-12-24 |                279.93  |                 274.432 |                268.933 |       3.8093 |
        | 2025-12-26 |                279.231 |                 274.048 |                268.866 |       2.51   |
        | 2025-12-29 |                278.663 |                 273.754 |                268.846 |       2.01   |
        | 2025-12-30 |                278.183 |                 273.462 |                268.742 |       1.8    |
        | 2025-12-31 |                277.084 |                 272.969 |                268.854 |       1.93   |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        | Date       |   McClellan Oscillator |   Advancers - Decliners |   On-Balance Volume |
        |:-----------|-----------------------:|------------------------:|--------------------:|
        | 2025-12-17 |                -0.0981 |                      -1 |         2.26688e+09 |
        | 2025-12-18 |                -0.0378 |                       1 |         2.31851e+09 |
        | 2025-12-19 |                 0.0139 |                       1 |         2.46314e+09 |
        | 2025-12-22 |                -0.0419 |                      -1 |         2.42657e+09 |
        | 2025-12-23 |                 0.0106 |                       1 |         2.45621e+09 |
        | 2025-12-24 |                 0.0554 |                       1 |         2.47412e+09 |
        | 2025-12-26 |                -0.0066 |                      -1 |         2.4526e+09  |
        | 2025-12-29 |                 0.0405 |                       1 |         2.47632e+09 |
        | 2025-12-30 |                -0.0195 |                      -1 |         2.45418e+09 |
        | 2025-12-31 |                -0.0707 |                      -1 |         2.42688e+09 |
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

        The McClellan Oscillator is a breadth indicator that measures the difference
        between the exponential moving average of advancing stocks and the exponential
        moving average of declining stocks.

        The formula is a follows:

        - McClellan Oscillator = EMA(Advancers) — EMA(Decliners)

        Also known as: McClellan oscillator, market breadth.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        Calculate the Advancers/Decliners ratio for a given price series.

        The Advancers/Decliners ratio is a breadth indicator that measures the number
        of advancing stocks (stocks with positive price changes) versus the number of
        declining stocks (stocks with negative price changes).

        The formula is a follows:

        - Advancers/Decliners = Advancers / Decliners

        Also known as: advance decline ratio, market breadth.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

        Returns:
            pd.Series or pd.DataFrame: Advancers/Decliners ratio values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Advancers/Decliners ratio for each asset in the Toolkit instance.
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

        - OBV = Previous OBV + Current Volume if Close > Previous Close

        Also known as: OBV, volume momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the OBV.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        - ADL = Previous ADL + Current ADL

        Also known as: ADL, Chaikin ADL, volume-price trend.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Accumulation/Distribution Line.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        | Date       |         AAPL |         MSFT |    Benchmark |
        |:-----------|-------------:|-------------:|-------------:|
        | 2025-12-17 | -1.45902e+11 | -9.37176e+10 | -7.798e+11   |
        | 2025-12-18 | -1.45884e+11 | -9.37314e+10 | -7.80119e+11 |
        | 2025-12-19 | -1.45842e+11 | -9.37925e+10 | -7.80274e+11 |
        | 2025-12-22 | -1.45885e+11 | -9.38141e+10 | -7.80375e+11 |
        | 2025-12-23 | -1.45873e+11 | -9.38379e+10 | -7.80477e+11 |
        | 2025-12-24 | -1.45881e+11 | -9.38435e+10 | -7.80588e+11 |
        | 2025-12-26 | -1.45906e+11 | -9.38632e+10 | -7.8078e+11  |
        | 2025-12-29 | -1.45914e+11 | -9.38748e+10 | -7.80985e+11 |
        | 2025-12-30 | -1.45935e+11 | -9.38961e+10 | -7.81265e+11 |
        | 2025-12-31 | -1.4598e+11  | -9.39293e+10 | -7.81472e+11 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): Number of periods for the short-term moving average.
                Defaults to 3.
            long_window (int, optional): Number of periods for the long-term moving average.
                Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Chaikin Oscillator.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        | Date       |         AAPL |         MSFT |    Benchmark |
        |:-----------|-------------:|-------------:|-------------:|
        | 2025-12-17 | -7.03961e+07 | -6.68531e+07 | -7.44616e+08 |
        | 2025-12-18 | -6.35575e+07 | -6.82533e+07 | -7.92549e+08 |
        | 2025-12-19 | -4.17226e+07 | -8.2076e+07  | -7.89548e+08 |
        | 2025-12-22 | -4.25369e+07 | -8.71447e+07 | -7.48581e+08 |
        | 2025-12-23 | -3.52131e+07 | -8.88686e+07 | -6.96233e+08 |
        | 2025-12-24 | -3.16563e+07 | -8.32606e+07 | -6.47027e+08 |
        | 2025-12-26 | -3.5274e+07  | -7.96613e+07 | -6.29072e+08 |
        | 2025-12-29 | -3.60858e+07 | -7.46637e+07 | -6.29609e+08 |
        | 2025-12-30 | -3.97223e+07 | -7.2589e+07  | -6.61784e+08 |
        | 2025-12-31 | -5.19347e+07 | -7.5703e+07  | -6.80757e+08 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to sum the Money Flow Volume and
                volume over. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Chaikin Money Flow.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        historical_data = self._historical_data[period]

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        historical_data = self._historical_data[period]

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            start_value (float, optional): The index value to start the series at.
                Defaults to 1000.0.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Negative Volume Index.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            start_value (float, optional): The index value to start the series at.
                Defaults to 1000.0.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Positive Volume Index.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            window (int, optional): The window size for calculating indicators.
                Defaults to 14.
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        | Date       |   Ichimoku Leading Span A |   Ichimoku Leading Span B |   Stochastic %K |   Stochastic %D |
        |:-----------|--------------------------:|--------------------------:|----------------:|----------------:|
        | 2025-12-17 |                   266.223 |                   251.635 |         -3.1678 |          4.1637 |
        | 2025-12-18 |                   266.223 |                   251.635 |         20.7711 |          9.9765 |
        | 2025-12-19 |                   266.223 |                   251.635 |         27.5828 |         15.062  |
        | 2025-12-22 |                   266.067 |                   251.635 |         15.1564 |         21.1701 |
        | 2025-12-23 |                   266.197 |                   251.635 |         26.2694 |         23.0029 |
        | 2025-12-24 |                   266.113 |                   251.635 |         43.1057 |         28.1772 |
        | 2025-12-26 |                   266.01  |                   251.635 |         43.2409 |         37.5387 |
        | 2025-12-29 |                   266.118 |                   251.635 |         45.9614 |         44.1027 |
        | 2025-12-30 |                   266.118 |                   251.635 |         40.8235 |         43.3419 |
        | 2025-12-31 |                   266.652 |                   251.985 |         31.6061 |         39.4636 |
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

        momentum_indicators["Relative Vigor Index"] = self.get_relative_vigor_index(
            period=period, close_column=close_column, window=window
        )

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for calculating the MFI.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | 39.7392 | 50.9834 |     44.2616 |
        | 2025-12-18 | 34.4001 | 53.0157 |     47.098  |
        | 2025-12-19 | 43.3298 | 65.0306 |     54.0785 |
        | 2025-12-22 | 36.4311 | 64.7749 |     54.3411 |
        | 2025-12-23 | 37.2983 | 72.343  |     54.6445 |
        | 2025-12-24 | 41.658  | 71.0457 |     53.7969 |
        | 2025-12-26 | 46.944  | 67.1793 |     52.2879 |
        | 2025-12-29 | 48.1202 | 62.6155 |     51.9377 |
        | 2025-12-30 | 43.4592 | 62.5152 |     52.4301 |
        | 2025-12-31 | 38.2002 | 66.6327 |     45.3837 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for calculating the Williams %R.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        | Date       |      AAPL |     MSFT |   Benchmark |
        |:-----------|----------:|---------:|------------:|
        | 2025-12-17 | -103.168  | -90.1043 |   -138.594  |
        | 2025-12-18 |  -79.2289 | -55.5752 |   -110.805  |
        | 2025-12-19 |  -72.4176 | -47.0526 |    -77.2    |
        | 2025-12-22 |  -84.8436 | -48.7255 |    -53.892  |
        | 2025-12-23 |  -73.7306 | -39.7722 |    -36.6859 |
        | 2025-12-24 |  -56.8943 | -34.3445 |    -29.5492 |
        | 2025-12-26 |  -56.7591 | -35.7829 |    -32.7468 |
        | 2025-12-29 |  -54.0386 | -38.0923 |    -44.6774 |
        | 2025-12-30 |  -59.1765 | -28.0484 |    -48.7512 |
        | 2025-12-31 |  -68.3939 | -48.4511 |    -73.436  |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for calculating the Aroon Indicator.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        historical_data = self._historical_data[period]

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods for calculating the CCI.
                Defaults to 14.
            constant (float, optional): Constant multiplier used in the CCI calculation.
                Defaults to 0.015.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | -139.878  | -74.8462 |   -239.064  |
        | 2025-12-18 | -162.46   |  26.1856 |   -118.613  |
        | 2025-12-19 | -103.254  |  54.4699 |    -50.7464 |
        | 2025-12-22 | -104.72   |  66.0257 |     45.5952 |
        | 2025-12-23 |  -97.9524 |  71.0192 |    102.705  |
        | 2025-12-24 |  -34.668  |  72.917  |    142.159  |
        | 2025-12-26 |  -25.6791 |  62.5848 |    138.631  |
        | 2025-12-29 |  -30.5711 |  56.6661 |     77.376  |
        | 2025-12-30 |  -35.5213 |  80.3202 |     62.6655 |
        | 2025-12-31 |  -56.8883 |  33.0457 |      5.5432 |
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
        window: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Vigor Index (RVI) for a given price series.

        The Relative Vigor Index is an oscillator that measures the conviction of a current price
        trend using the relationship between closing and opening prices.

        The formula is a follows:

        - RVI = Sum(Upward Close-Open Movement, window) / (Sum(Upward Close-Open Movement, window)
          + Sum(Downward Close-Open Movement, window))

        Also known as: vigor index. Note this is a bounded [0, 1] measure of the proportion
        of upward close-open movement, related in spirit to (but not numerically the same
        as) John Ehlers' published Relative Vigor Index. See `momentum_model.get_relative_vigor_index`
        for the full formula and caveats.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods for calculating the RVI.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

        Returns:
            pd.Series or pd.DataFrame: Relative Vigor Index (RVI) values.

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

        toolkit.technicals.get_relative_vigor_index()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-17 | 0.2613 | 0.2151 |           0 |
        | 2025-12-18 | 0.2279 | 0.239  |           0 |
        | 2025-12-19 | 0.1418 | 0.2404 |           0 |
        | 2025-12-22 | 0.0655 | 0.2174 |           0 |
        | 2025-12-23 | 0.096  | 0.2191 |           0 |
        | 2025-12-24 | 0.1358 | 0.2245 |           0 |
        | 2025-12-26 | 0.1412 | 0.226  |           0 |
        | 2025-12-29 | 0.1582 | 0.1717 |           0 |
        | 2025-12-30 | 0.1666 | 0.1669 |           0 |
        | 2025-12-31 | 0.1448 | 0.1712 |           0 |
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

        relative_vigor_index = momentum_model.get_relative_vigor_index(
            historical_data["Open"],
            historical_data[close_column],
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
        window: int = 14,
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods for calculating the Force Index.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 |      -3.27663e+07 | -3.99098e+07 | -1.71384e+08 |
        | 2025-12-18 |      -2.59949e+07 | -4.83203e+06 | -7.58704e+07 |
        | 2025-12-19 |       5.93465e+06 |  1.402e+07   |  1.80341e+07 |
        | 2025-12-22 |      -7.98689e+06 |  9.90314e+06 |  5.46472e+07 |
        | 2025-12-23 |      -1.44294e+06 |  1.23374e+07 |  7.42107e+07 |
        | 2025-12-24 |       2.20261e+06 |  1.16002e+07 |  7.69454e+07 |
        | 2025-12-26 |  735563           |  9.69036e+06 |  6.63004e+07 |
        | 2025-12-29 |       1.77297e+06 |  7.51784e+06 |  3.70993e+07 |
        | 2025-12-30 | -465435           |  7.2177e+06  |  2.69116e+07 |
        | 2025-12-31 |      -4.83113e+06 | -1.72373e+06 | -2.66057e+07 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | 27.9933 | 17.9054 |     -7.5735 |
        | 2025-12-18 | 35.8123 | 22.5115 |     -2.2767 |
        | 2025-12-19 | 36.5493 | 25.4702 |      0.5161 |
        | 2025-12-22 | 29.9138 | 20.0146 |      3.1977 |
        | 2025-12-23 | 33.3379 | 23.4866 |      9.4209 |
        | 2025-12-24 | 37.9768 | 25.935  |     13.1508 |
        | 2025-12-26 | 34.3218 | 23.0549 |     14.4572 |
        | 2025-12-29 | 39.1009 | 22.3076 |     19.2616 |
        | 2025-12-30 | 33.287  | 14.6673 |     16.3247 |
        | 2025-12-31 | 23.6054 |  6.9239 |      2.7084 |
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
        short_window: int = 7,
        long_window: int = 28,
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

        - PPO = ((Long-term EMA — Short-term EMA) / Short—term EMA) * 100

        Also known as: PPO, price oscillator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            short_window (int, optional): The number of periods for the short-term moving average.
                Defaults to 7.
            long_window (int, optional): The number of periods for the long-term moving average.
                Defaults to 28.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 |  0.5197 | -1.9375 |      0.261  |
        | 2025-12-18 |  0.2635 | -1.6172 |      0.1621 |
        | 2025-12-19 |  0.1776 | -1.2933 |      0.2538 |
        | 2025-12-22 | -0.0638 | -1.0821 |      0.4267 |
        | 2025-12-23 | -0.1396 | -0.844  |      0.6233 |
        | 2025-12-24 | -0.0941 | -0.6196 |      0.8137 |
        | 2025-12-26 | -0.0879 | -0.4638 |      0.9303 |
        | 2025-12-29 | -0.0582 | -0.3697 |      0.9299 |
        | 2025-12-30 | -0.0815 | -0.2834 |      0.8911 |
        | 2025-12-31 | -0.1771 | -0.3618 |      0.7136 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods to consider for the DPO calculation.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 |  0.0306 |  0.2009 |      3.1326 |
        | 2025-12-18 | -0.3825 |  8.5813 |      1.5837 |
        | 2025-12-19 | -0.4188 |  9.6332 |      0.8352 |
        | 2025-12-22 |  2.261  | -3.3814 |      4.9406 |
        | 2025-12-23 |  2.3529 |  0.8503 |      6.0923 |
        | 2025-12-24 |  3.093  | -4.5682 |     -1.806  |
        | 2025-12-26 | -0.6824 | -8.5777 |     -3.2957 |
        | 2025-12-29 |  0.1104 | -6.7393 |     -5.5777 |
        | 2025-12-30 | -2.36   | -6.6855 |    -13.3919 |
        | 2025-12-31 | -1.518  |  0.7659 |     -8.1193 |
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

        Also known as: ADX, trend strength indicator. See `momentum_model.get_average_directional_index`
        for the full formula, which uses Wilder's smoothing (not a plain SMA) throughout.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods to consider for the ADX calculation.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | 24.548  | 26.1163 |     14.9403 |
        | 2025-12-18 | 24.3281 | 24.8053 |     14.355  |
        | 2025-12-19 | 23.8204 | 23.5878 |     13.7378 |
        | 2025-12-22 | 23.3491 | 22.3379 |     13.2009 |
        | 2025-12-23 | 23.1372 | 21.1773 |     13.2036 |
        | 2025-12-24 | 21.9724 | 19.8989 |     13.6364 |
        | 2025-12-26 | 20.8908 | 18.7118 |     14.1712 |
        | 2025-12-29 | 20.0343 | 17.8823 |     13.912  |
        | 2025-12-30 | 19.2603 | 16.8765 |     13.6713 |
        | 2025-12-31 | 18.7103 | 16.2998 |     12.9966 |
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

        - CMO = ((Sum of Upward Change) — (Sum of Downward Change)) / ((Sum of Upward Change)
            + (Sum of Downward Change))

        Also known as: CMO, Chande momentum.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column in the historical data that represents
                the closing prices. Defaults to "Adj Close".
            window (int, optional): The number of periods to consider for the CMO calculation.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | -20.6213 | -13.3126 |    -21.3406 |
        | 2025-12-18 | -24.9065 | -11.1823 |    -17.2311 |
        | 2025-12-19 | -39.3405 |  -1.1974 |      5.3996 |
        | 2025-12-22 | -64.546  |  -7.6713 |     11.5321 |
        | 2025-12-23 | -51.4168 |  16.3207 |     13.0006 |
        | 2025-12-24 | -32.9191 |  13.311  |     16.4316 |
        | 2025-12-26 | -27.7033 |   8.7618 |     13.9793 |
        | 2025-12-29 | -21.8632 |  -8.7735 |     13.013  |
        | 2025-12-30 | -21.7389 | -10.3042 |     12.4225 |
        | 2025-12-31 | -37.4451 |  14.6837 |     -7.4375 |
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

        Also known as: Ichimoku Kinko Hyo, cloud indicator. The default windows (9, 26, 52)
        are Goichi Hosoda's original values, both leading spans are conventionally displaced
        forward by the base (Kijun-sen) period.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

        Returns:
            pd.Series or pd.DataFrame:
            Conversion Line, Base Line, Lead Span A, and Lead Span B values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ichimoku Cloud values for each asset in the Toolkit instance.
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

        toolkit.technicals.get_ichimoku_cloud().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   Base Line |   Conversion Line |   Leading Span A |   Leading Span B |
        |:-----------|------------:|------------------:|-----------------:|-----------------:|
        | 2025-12-17 |     276.97  |           276.39  |          266.223 |          251.635 |
        | 2025-12-18 |     276.97  |           273.55  |          266.223 |          251.635 |
        | 2025-12-19 |     276.97  |           273.55  |          266.223 |          251.635 |
        | 2025-12-22 |     276.97  |           273.55  |          266.067 |          251.635 |
        | 2025-12-23 |     276.97  |           273.55  |          266.197 |          251.635 |
        | 2025-12-24 |     276.97  |           273.55  |          266.113 |          251.635 |
        | 2025-12-26 |     277.06  |           273.55  |          266.01  |          251.635 |
        | 2025-12-29 |     277.145 |           271.555 |          266.118 |          251.635 |
        | 2025-12-30 |     277.145 |           271.555 |          266.118 |          251.635 |
        | 2025-12-31 |     277.785 |           271.19  |          266.652 |          251.985 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
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
        | 2025-12-17 |          4.1637 |         -3.1678 |
        | 2025-12-18 |          9.9765 |         20.7711 |
        | 2025-12-19 |         15.0619 |         27.5824 |
        | 2025-12-22 |         21.17   |         15.1564 |
        | 2025-12-23 |         23.0027 |         26.2694 |
        | 2025-12-24 |         28.1772 |         43.1057 |
        | 2025-12-26 |         37.5387 |         43.2409 |
        | 2025-12-29 |         44.1027 |         45.9614 |
        | 2025-12-30 |         43.3419 |         40.8235 |
        | 2025-12-31 |         39.4636 |         31.6061 |
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

        historical_data = self._historical_data[period]

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
        of the MACD line, signal line, and MACD histogram.

        The formula is a follows:

        - MACD Line = Short-term EMA — Long-term EMA
        - Signal Line = SMA(MACD Line)

        Also known as: MACD, momentum indicator.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-31 |     -0.0897 |        0.6285 |
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

        Also known as: RSI, momentum oscillator, overbought, oversold.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for RSI calculation. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the RSI.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-19 | 49.0173 | 48.7212 |     54.2478 |
        | 2025-12-22 | 44.0574 | 47.9724 |     57.6776 |
        | 2025-12-23 | 47.0291 | 49.5829 |     60.058  |
        | 2025-12-24 | 50.012  | 50.5816 |     61.8449 |
        | 2025-12-26 | 49.1688 | 50.2972 |     61.7588 |
        | 2025-12-29 | 49.9666 | 49.7053 |     58.6687 |
        | 2025-12-30 | 48.4209 | 50.0994 |     57.6089 |
        | 2025-12-31 | 45.6901 | 46.1448 |     51.5335 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the BOP.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | -0.8646 | -0.7583 |     -1.6944 |
        | 2025-12-18 | -0.3232 |  0.2339 |     -1.4299 |
        | 2025-12-19 |  0.1654 | -0.8402 |     -0.2759 |
        | 2025-12-22 | -0.7791 | -0.7048 |     -0.9261 |
        | 2025-12-23 |  0.2655 | -0.3881 |     -0.2983 |
        | 2025-12-24 |  0.225  | -0.1701 |     -0.9639 |
        | 2025-12-26 | -0.5985 | -0.9605 |     -2.3766 |
        | 2025-12-29 |  0.1626 | -0.1992 |     -1.6042 |
        | 2025-12-30 | -0.2618 | -0.3644 |     -2.9114 |
        | 2025-12-31 | -1.0041 | -1.5018 |     -1.8593 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 |  1.2158 | -14.8666 |      4.3348 |
        | 2025-12-18 | -0.0362 | -12.9889 |      2.9057 |
        | 2025-12-19 | -1.1611 | -10.7445 |      1.9561 |
        | 2025-12-22 | -2.1283 |  -7.9246 |      2.0471 |
        | 2025-12-23 | -2.6946 |  -4.7973 |      3.3903 |
        | 2025-12-24 | -2.844  |  -2.2009 |      5.7422 |
        | 2025-12-26 | -2.1811 |  -1.1385 |      7.759  |
        | 2025-12-29 | -2.0726 |  -0.6239 |      8.8979 |
        | 2025-12-30 | -1.9516 |   0.2001 |      9.5533 |
        | 2025-12-31 | -1.6105 |   0.6829 |      9.1537 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods to sum the directional movement and
                true range over. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VI+ and VI- values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | 0.9166 | 1.0481 |
        | 2025-12-18 | 0.861  | 1.0752 |
        | 2025-12-19 | 0.8987 | 1.1316 |
        | 2025-12-22 | 0.8073 | 1.2145 |
        | 2025-12-23 | 0.773  | 1.2658 |
        | 2025-12-24 | 0.8802 | 1.1495 |
        | 2025-12-26 | 0.8998 | 1.0898 |
        | 2025-12-29 | 0.9218 | 1.084  |
        | 2025-12-30 | 0.9046 | 1.0977 |
        | 2025-12-31 | 0.9015 | 1.1027 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for the EMA used as the trend baseline.
                Defaults to 13, as originally proposed by Elder.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Bull and Bear Power.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 |      -3.9033 |       0.6167 |
        | 2025-12-18 |      -8.0087 |      -1.3287 |
        | 2025-12-19 |      -4.7685 |      -0.0685 |
        | 2025-12-22 |      -3.525  |      -0.155  |
        | 2025-12-23 |      -4.1301 |      -1.1901 |
        | 2025-12-24 |      -1.4011 |       1.8289 |
        | 2025-12-26 |      -0.6063 |       1.9037 |
        | 2025-12-29 |      -1.0521 |       0.9579 |
        | 2025-12-30 |      -0.9702 |       0.8298 |
        | 2025-12-31 |      -1.1962 |       0.7338 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to look back for the rate of change
                calculation. Defaults to 12.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Rate of Change.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the Choppiness Index
                calculation. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Choppiness Index.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        historical_data = self._historical_data[period]

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            window (int, optional): The window size for calculating indicators.
                Defaults to 14.
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        | Date       |   Simple Moving Average (SMA) |   Exponential Moving Average (EMA) |
        |:-----------|------------------------------:|-----------------------------------:|
        | 2025-12-17 |                       277.993 |                            275.523 |
        | 2025-12-18 |                       277.518 |                            274.98  |
        | 2025-12-19 |                       276.846 |                            274.706 |
        | 2025-12-22 |                       275.762 |                            274.11  |
        | 2025-12-23 |                       274.922 |                            273.778 |
        | 2025-12-24 |                       274.432 |                            273.683 |
        | 2025-12-26 |                       274.048 |                            273.547 |
        | 2025-12-29 |                       273.754 |                            273.476 |
        | 2025-12-30 |                       273.462 |                            273.324 |
        | 2025-12-31 |                       272.969 |                            273.031 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the moving average.
                The number of periods (time intervals) over which to calculate the MA.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the MA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        - EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA

        Also known as: EMA.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for EMA calculation.
                The number of periods (time intervals) over which to calculate the EMA.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the EMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        - EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA

        Also known as: DEMA, double EMA.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for moving average calculation.
                The number of periods (time intervals) over which to calculate the moving average.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the DEMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for moving average calculation.
                The number of periods (time intervals) over which to calculate the moving average.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Trix.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        Also known as: Bollinger Bands, BB, volatility bands, price channels.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for moving average calculation.
                The number of periods (time intervals) over which to calculate the moving average.
            num_std_dev (int, optional): Number of standard deviations for the bands.
                Defaults to 2.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the bands.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-19 | 272.927 |      268.779 |       276.846 |      284.913 |
        | 2025-12-22 | 270.234 |      268.558 |       275.762 |      282.966 |
        | 2025-12-23 | 271.621 |      268.807 |       274.922 |      281.038 |
        | 2025-12-24 | 273.067 |      268.933 |       274.432 |      279.93  |
        | 2025-12-26 | 272.658 |      268.866 |       274.048 |      279.231 |
        | 2025-12-29 | 273.017 |      268.846 |       273.754 |      278.663 |
        | 2025-12-30 | 272.339 |      268.741 |       273.462 |      278.183 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for TMA calculation.
                The number of periods (time intervals) over which to calculate the TMA.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the TMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the WMA.
                The number of periods (time intervals) over which to calculate the WMA.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the WMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the HMA.
                The number of periods (time intervals) over which to calculate the HMA.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the HMA.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        Also known as: VWAP.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods to consider for the VWAP.
                The number of periods (time intervals) over which to calculate the VWAP.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VWAP.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | 278.69  | 480.852 |     679.761 |
        | 2025-12-18 | 278.036 | 480.654 |     679.295 |
        | 2025-12-19 | 276.712 | 481.006 |     679.204 |
        | 2025-12-22 | 275.771 | 480.754 |     679.359 |
        | 2025-12-23 | 274.962 | 481.201 |     679.607 |
        | 2025-12-24 | 274.503 | 481.459 |     679.795 |
        | 2025-12-26 | 274.123 | 481.63  |     679.841 |
        | 2025-12-29 | 273.864 | 481.314 |     680.09  |
        | 2025-12-30 | 273.614 | 481.15  |     680.258 |
        | 2025-12-31 | 273.296 | 481.637 |     680.121 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | 286.488 | 490.376 |     688.965 |
        | 2025-12-18 | 285.3   | 470.88  |     688.254 |
        | 2025-12-19 | 283.465 | 471.254 |     687.572 |
        | 2025-12-22 | 281.813 | 471.621 |     686.917 |
        | 2025-12-23 | 280.327 | 471.981 |     671.2   |
        | 2025-12-24 | 278.989 | 472.333 |     671.54  |
        | 2025-12-26 | 277.785 | 472.679 |     672.312 |
        | 2025-12-29 | 276.702 | 473.017 |     673.472 |
        | 2025-12-30 | 275.727 | 473.349 |     674.564 |
        | 2025-12-31 | 274.849 | 474.002 |     675.59  |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Pivot Points.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 |       273.718 |        275.646 |        277.428 |        279.356 |     271.936 |     270.008 |     268.226 |
        | 2025-12-18 |       272.967 |        274.295 |        277.487 |        278.815 |     269.775 |     268.447 |     265.255 |
        | 2025-12-19 |       270.677 |        274.404 |        277.357 |        281.084 |     267.724 |     263.997 |     261.044 |
        | 2025-12-22 |       272.476 |        275.051 |        277.176 |        279.751 |     270.351 |     267.776 |     265.651 |
        | 2025-12-23 |       271.541 |        272.573 |        274.911 |        275.943 |     269.203 |     268.171 |     265.833 |
        | 2025-12-24 |       271.227 |        272.894 |        274.167 |        275.834 |     269.954 |     268.287 |     267.014 |
        | 2025-12-26 |       273.566 |        274.931 |        276.796 |        278.161 |     271.701 |     270.336 |     268.471 |
        | 2025-12-29 |       273.629 |        274.399 |        276.139 |        276.909 |     271.889 |     271.119 |     269.378 |
        | 2025-12-30 |       273.242 |        274.135 |        275.252 |        276.145 |     272.125 |     271.232 |     270.115 |
        | 2025-12-31 |       272.9   |        273.519 |        274.7   |        275.319 |     271.719 |     271.1   |     269.919 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        Note that the columns sort lexicographically by their label (so "100.0%" sorts right
        after "0.0%", ahead of "23.6%"), matching the sorting convention used by every other
        multi-column indicator in this module (e.g. Pivot Points' Resistance/Support levels).

        | Date       |   0.0% |   100.0% |   23.6% |   38.2% |   50.0% |   61.8% |   78.6% |
        |:-----------|-------:|---------:|--------:|--------:|--------:|--------:|--------:|
        | 2025-12-17 | 288.62 |   271.64 | 284.613 | 282.134 | 280.13  | 278.126 | 275.274 |
        | 2025-12-18 | 288.62 |   266.95 | 283.506 | 280.342 | 277.785 | 275.228 | 271.587 |
        | 2025-12-19 | 288.62 |   266.95 | 283.506 | 280.342 | 277.785 | 275.228 | 271.587 |
        | 2025-12-22 | 288.62 |   266.95 | 283.506 | 280.342 | 277.785 | 275.228 | 271.587 |
        | 2025-12-23 | 284.73 |   266.95 | 280.534 | 277.938 | 275.84  | 273.742 | 270.755 |
        | 2025-12-24 | 281.14 |   266.95 | 277.791 | 275.719 | 274.045 | 272.371 | 269.987 |
        | 2025-12-26 | 280.15 |   266.95 | 277.035 | 275.108 | 273.55  | 271.992 | 269.775 |
        | 2025-12-29 | 280.15 |   266.95 | 277.035 | 275.108 | 273.55  | 271.992 | 269.775 |
        | 2025-12-30 | 280.15 |   266.95 | 277.035 | 275.108 | 273.55  | 271.992 | 269.775 |
        | 2025-12-31 | 280.15 |   266.95 | 277.035 | 275.108 | 273.55  | 271.992 | 269.775 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            window (int, optional): The window size for calculating indicators.
                Defaults to 14.
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        | Date       |   Bollinger Band Upper |   Bollinger Band Middle |   Bollinger Band Lower |   True Range |
        |:-----------|-----------------------:|------------------------:|-----------------------:|-------------:|
        | 2025-12-17 |                285.486 |                 277.993 |                270.499 |       4.52   |
        | 2025-12-18 |                285.733 |                 277.518 |                269.304 |       6.68   |
        | 2025-12-19 |                284.913 |                 276.846 |                268.779 |       4.7    |
        | 2025-12-22 |                282.966 |                 275.762 |                268.558 |       3.37   |
        | 2025-12-23 |                281.038 |                 274.922 |                268.807 |       2.94   |
        | 2025-12-24 |                279.93  |                 274.432 |                268.933 |       3.8093 |
        | 2025-12-26 |                279.231 |                 274.048 |                268.866 |       2.51   |
        | 2025-12-29 |                278.663 |                 273.754 |                268.846 |       2.01   |
        | 2025-12-30 |                278.183 |                 273.462 |                268.742 |       1.8    |
        | 2025-12-31 |                277.084 |                 272.969 |                268.854 |       1.93   |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the True Range.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 | 4.52   |  6.6132 |      9.24   |
        | 2025-12-18 | 6.68   | 16.4816 |     16.5062 |
        | 2025-12-19 | 4.7    |  6.9211 |     11.8403 |
        | 2025-12-22 | 3.37   |  6.04   |     10.0446 |
        | 2025-12-23 | 2.94   |  5.967  |      8.6775 |
        | 2025-12-24 | 3.8093 |  5.3792 |      8.2018 |
        | 2025-12-26 | 2.51   |  3.1766 |      6.6305 |
        | 2025-12-29 | 2.01   |  4.17   |      4.24   |
        | 2025-12-30 | 1.8    |  5.6508 |      6.041  |
        | 2025-12-31 | 1.93   |  4.84   |      5.6745 |
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

        Also known as: ATR, volatility indicator. See `volatility_model.get_average_true_range`
        for the full formula; Wilder's smoothing constant (1/window) is slower than a
        standard EMA's (2/(window+1)) of the same window.

        Args:
            period (str): Period for which to calculate the ATR.
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int): Number of periods for ATR calculation.
                The number of periods (time intervals) over which to calculate the Average True Range.
            rounding (int | None): Number of decimal places to round the resulting ATR values to.
                If None, no rounding is performed.
            growth (bool): Flag indicating whether to return the ATR growth rate.
                If True, the ATR growth rate is calculated.
            lag (int | list[int]): Number of periods to lag the ATR values by.
                If an integer is provided, all ATR values are lagged by the same number of periods.
                If a list of integers is provided, each ATR value is lagged by the corresponding number of periods.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: ATR values per ticker or ATR growth rate (if growth is True).
                A pandas Series containing the calculated Average True Range values or growth rate for each period.

        Formula:
        The Average True Range (ATR) is calculated using the following steps:
        1. Calculate the True Range (TR) for each period:
            - True Range (TR) = max(high — low, abs(high — previous_close), abs(low — previous_close))
        2. Calculate the Average True Range (ATR) over the specified window:
            - ATR = EMA(TR, window), where EMA is the Exponential Moving Average.

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

        | Date       |   AAPL |    MSFT |   Benchmark |
        |:-----------|-------:|--------:|------------:|
        | 2025-12-17 | 4.941  |  9.5365 |     10.9804 |
        | 2025-12-18 | 5.0652 | 10.0326 |     11.3751 |
        | 2025-12-19 | 5.0391 |  9.8103 |     11.4084 |
        | 2025-12-22 | 4.9199 |  9.541  |     11.3109 |
        | 2025-12-23 | 4.7785 |  9.2857 |     11.1228 |
        | 2025-12-24 | 4.7092 |  9.0067 |     10.9142 |
        | 2025-12-26 | 4.5521 |  8.5902 |     10.6082 |
        | 2025-12-29 | 4.3706 |  8.2745 |     10.1533 |
        | 2025-12-30 | 4.1869 |  8.0871 |      9.8596 |
        | 2025-12-31 | 4.0257 |  7.8552 |      9.5607 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the underlying Average True Range
                calculation. Defaults to 10.
            multiplier (float, optional): Multiplier applied to the Average True Range to
                determine how far the bands sit from the median price. Defaults to 3.0.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Supertrend and
                Trend Direction values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        historical_data = self._historical_data[period]

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
        window: int = 14,
        atr_window: int = 14,
        atr_multiplier: int = 2,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Keltner Channels for a given price series.

        The Keltner Channels consist of three lines:
        - Upper Channel Line = Exponential Moving Average (EMA) of High Prices + ATR * ATR Multiplier
        - Middle Channel Line = Exponential Moving Average (EMA) of Closing Prices
        - Lower Channel Line = Exponential Moving Average (EMA) of Low Prices — ATR * ATR Multiplier

        The formula is a follows:

        - EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA
        - ATR = EMA(TR, ATR Window)
        - Upper Channel Line = EMA(High, Window) + ATR * ATR Multiplier
        - Middle Channel Line = EMA(Close, Window)
        - Lower Channel Line = EMA(Low, Window) — ATR * ATR Multiplier

        Also known as: ATR-based bands, volatility channels.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the moving average.
                Defaults to 14.
            atr_window (int, optional): Number of periods for ATR calculation.
                Defaults to 14.
            atr_multiplier (int, optional): Multiplier for ATR to determine channel width.
                Defaults to 2.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the channels.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 |      265.641 |       275.523 |      285.405 |
        | 2025-12-18 |      264.85  |       274.98  |      285.11  |
        | 2025-12-19 |      264.628 |       274.706 |      284.784 |
        | 2025-12-22 |      264.27  |       274.11  |      283.95  |
        | 2025-12-23 |      264.221 |       273.778 |      283.335 |
        | 2025-12-24 |      264.265 |       273.683 |      283.102 |
        | 2025-12-26 |      264.442 |       273.547 |      282.651 |
        | 2025-12-29 |      264.735 |       273.476 |      282.217 |
        | 2025-12-30 |      264.95  |       273.324 |      281.698 |
        | 2025-12-31 |      264.979 |       273.031 |      281.082 |
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
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the Donchian Channels.
                Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the channels.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
        | 2025-12-17 |          265.32 |          276.97  |          288.62 |
        | 2025-12-18 |          265.5  |          277.06  |          288.62 |
        | 2025-12-19 |          265.67 |          277.145 |          288.62 |
        | 2025-12-22 |          265.67 |          277.145 |          288.62 |
        | 2025-12-23 |          266.95 |          277.785 |          288.62 |
        | 2025-12-24 |          266.95 |          277.785 |          288.62 |
        | 2025-12-26 |          266.95 |          277.785 |          288.62 |
        | 2025-12-29 |          266.95 |          277.785 |          288.62 |
        | 2025-12-30 |          266.95 |          277.785 |          288.62 |
        | 2025-12-31 |          266.95 |          277.785 |          288.62 |
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

        Also known as: volatility cone, realized volatility term structure.

        Args:
            windows (list[int] | None, optional): The rolling windows (in periods) to
                calculate realized volatility for. Defaults to [10, 20, 30, 60, 90, 120].
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
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

        for ticker in historical_data[close_column].columns:
            volatility_cone[ticker] = volatility_model.get_volatility_cone(
                historical_data[close_column][ticker], windows=windows
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

        Also known as: Arms Index, TRIN.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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

        Also known as: new highs minus new lows, record high percent.

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the new high / new low lookback.
                Defaults to 252.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
                Defaults to 1.

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
