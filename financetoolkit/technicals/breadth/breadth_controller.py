"""Breadth Module"""

__docformat__ = "google"

from typing import Literal

import pandas as pd

from financetoolkit.helpers import get_non_trading_days, handle_portfolio
from financetoolkit.technicals.breadth import breadth_model
from financetoolkit.technicals.helpers import (
    calculate_per_ticker,
    report_missing_values,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    PERIODS_PER_YEAR,
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

logger = get_logger()

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


@report_missing_values
@filter_shown_columns
@track_sources
class BreadthTechnicals:
    """
    Breadth and volume indicators: the McClellan oscillator, advancers and decliners, on-balance volume (and Archer's version), the price volume trend, the accumulation/distribution line (Chaikin's and Williams'), the Chaikin oscillator and money flow (and Twiggs money flow), the Klinger volume oscillator, the market facilitation index, the ease of movement, the negative and positive volume index (`get_volume_index`), relative volume, the volume flow indicator, finite volume elements, the volume zone oscillator, the cumulative force index (`get_force_index(method="cumulative")`), price-volume and its rank, buying and selling pressure, the volume profile, TRIN, new highs and lows and the market breadth.
    """

    @handle_errors
    def collect_breadth_indicators(
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
        Collect the breadth and volume indicators of every ticker in one table: how much volume and how many tickers
        take part in the moves of the prices.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
            pd.Series or pd.DataFrame: Breadth indicators calculated based on the specified parameters.

        Notes:
        - The method calculates various breadth indicators for each asset in the Toolkit instance.
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

        toolkit.technicals.collect_breadth_indicators().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        | Date       |   McClellan Oscillator |   Advancers - Decliners |   On-Balance Volume |   Price Volume Trend |   Accumulation/Distribution Line |   Chaikin Oscillator |     TRIN |   New Highs - New Lows |   Chaikin Money Flow |   Ease of Movement |   Negative Volume Index |   Positive Volume Index |   Klinger Volume Oscillator |   Klinger Volume Oscillator Signal Line |   Market Facilitation Index |   Williams Accumulation/Distribution |   Relative Volume |   Volume Flow Indicator |   Finite Volume Elements |   Twiggs Money Flow |   Volume Zone Oscillator |   Cumulative Force Index |   Price Volume |   Price Volume Rank |   Buying Pressure |   Selling Pressure |   Archer On-Balance Volume Fast Average |   Archer On-Balance Volume Slow Average |   Archer On-Balance Volume Minimum |   Archer On-Balance Volume Maximum |   Archer On-Balance Volume Long Run |   Archer On-Balance Volume Short Run |
        |:-----------|-----------------------:|------------------------:|--------------------:|---------------------:|---------------------------------:|---------------------:|---------:|-----------------------:|---------------------:|-------------------:|------------------------:|------------------------:|----------------------------:|----------------------------------------:|----------------------------:|-------------------------------------:|------------------:|------------------------:|-------------------------:|--------------------:|-------------------------:|-------------------------:|---------------:|--------------------:|------------------:|-------------------:|----------------------------------------:|----------------------------------------:|-----------------------------------:|-----------------------------------:|------------------------------------:|-------------------------------------:|
        | 2025-12-17 |                -0.0981 |                      -1 |         2.26688e+09 |          6.59406e+07 |                      3.99771e+09 |         -1.74849e+07 | nan      |                      0 |              -0.0358 |            -4.7166 |                 2735.54 |                 788.839 |                -2.03391e+08 |                            -4.28732e+07 |                      8.9837 |                              165.337 |            1.171  |                 25.8886 |                  -0.0337 |             -0.0429 |                  -0.194  |              1.01185e+10 |    1.36297e+10 |                   3 |            0.7602 |             0.9939 |                             2.29242e+09 |                             2.31813e+09 |                        2.26688e+09 |                        2.31702e+09 |                                   0 |                                    1 |
        | 2025-12-18 |                -0.0378 |                       1 |         2.31851e+09 |          6.60071e+07 |                      4.02708e+09 |         -1.16789e+07 | nan      |                      0 |               0.0024 |            -7.4093 |                 2735.54 |                 789.854 |                -2.50907e+08 |                            -7.25923e+07 |                     12.8931 |                              170.559 |            1.1996 |                 25.9203 |                   0.0135 |             -0.0099 |                  -0.0055 |              1.0083e+10  |    1.40534e+10 |                   1 |            0.9191 |             0.961  |                             2.30286e+09 |                             2.31819e+09 |                        2.26688e+09 |                        2.31851e+09 |                                   0 |                                    1 |
        | 2025-12-19 |                 0.0139 |                       1 |         2.46314e+09 |          6.67935e+07 |                      4.11448e+09 |          1.95656e+07 | nan      |                      0 |               0.1361 |            -9.4903 |                 2735.54 |                 794.149 |                -1.89072e+08 |                            -8.92323e+07 |                      3.2384 |                              174.316 |            3.3172 |                 27.1858 |                   0.149  |              0.0742 |                   0.3343 |              1.00928e+10 |    3.95814e+10 |                   1 |            1.2421 |             0.9738 |                             2.36697e+09 |                             2.34049e+09 |                        2.31851e+09 |                        2.46314e+09 |                                   1 |                                    0 |
        | 2025-12-22 |                -0.0419 |                      -1 |         2.42657e+09 |          6.64327e+07 |                      4.08789e+09 |          2.21089e+07 | nan      |                      0 |               0.079  |           -12.8266 |                 2708.55 |                 794.149 |                -1.99306e+08 |                            -1.04957e+08 |                      9.1827 |                              171.416 |            0.7534 |                 27.6286 |                   0.1149 |              0.0461 |                   0.2145 |              1.00829e+10 |    9.90986e+09 |                   4 |            1.1372 |             0.9639 |                             2.39081e+09 |                             2.35373e+09 |                        2.42657e+09 |                        2.46314e+09 |                                   1 |                                    0 |
        | 2025-12-23 |                 0.0106 |                       1 |         2.45621e+09 |          6.65848e+07 |                      4.11471e+09 |          2.96727e+07 | nan      |                      0 |               0.0651 |           -14.4653 |                 2722.44 |                 794.149 |                -2.24884e+08 |                            -1.22089e+08 |                      9.8839 |                              174.206 |            0.6251 |                 27.8548 |                   0.1894 |              0.071  |                   0.2753 |              1.00822e+10 |    8.07329e+09 |                   2 |            1.0956 |             0.8755 |                             2.41697e+09 |                             2.3695e+09  |                        2.42657e+09 |                        2.45621e+09 |                                   1 |                                    0 |
        | 2025-12-24 |                 0.0554 |                       1 |         2.47412e+09 |          6.66801e+07 |                      4.11466e+09 |          3.00518e+07 | nan      |                      0 |               0.0846 |            -6.6454 |                 2736.94 |                 794.149 |                -2.1628e+08  |                            -1.35545e+08 |                     17.9713 |                              175.81  |            0.3926 |                 27.2474 |                   0.156  |              0.0697 |                   0.3125 |              1.00864e+10 |    4.9041e+09  |                   2 |            1.0156 |             0.8174 |                             2.43983e+09 |                             2.38559e+09 |                        2.45621e+09 |                        2.47412e+09 |                                   1 |                                    0 |
        | 2025-12-26 |                -0.0066 |                      -1 |         2.4526e+09  |          6.66479e+07 |                      4.10239e+09 |          2.35736e+07 | nan      |                      0 |               0.0857 |            -5.4395 |                 2736.94 |                 792.96  |                -1.90374e+08 |                            -1.43378e+08 |                     11.6221 |                              173.847 |            0.4872 |                 26.5611 |                   0.0943 |              0.0554 |                   0.2254 |              1.00882e+10 |    5.88406e+09 |                   3 |            0.9294 |             0.7778 |                             2.44494e+09 |                             2.3959e+09  |                        2.4526e+09  |                        2.47412e+09 |                                   1 |                                    0 |
        | 2025-12-29 |                 0.0405 |                       1 |         2.47632e+09 |          6.66791e+07 |                      4.11195e+09 |          2.18213e+07 |   0.4593 |                      0 |               0.0756 |            -4.7954 |                 2736.94 |                 794.004 |                -1.915e+08   |                            -1.50253e+08 |                      8.4461 |                              175.252 |            0.5442 |                 26.2841 |                   0.0461 |              0.0642 |                   0.2857 |              1.00912e+10 |    6.49227e+09 |                   1 |            0.8723 |             0.7174 |                             2.45749e+09 |                             2.40827e+09 |                        2.4526e+09  |                        2.47632e+09 |                                   1 |                                    0 |
        | 2025-12-30 |                -0.0195 |                      -1 |         2.45418e+09 |          6.66241e+07 |                      4.10949e+09 |          1.83379e+07 |   1.5877 |                      0 |               0.0249 |            -5.2835 |                 2730.14 |                 794.004 |                -2.06929e+08 |                            -1.58349e+08 |                      8.1019 |                              174.256 |            0.506  |                 25.8195 |                   0.0467 |              0.06   |                   0.1863 |              1.0091e+10  |    6.04588e+09 |                   4 |            0.8069 |             0.6714 |                             2.45616e+09 |                             2.41534e+09 |                        2.45418e+09 |                        2.47632e+09 |                                   1 |                                    0 |
        | 2025-12-31 |                -0.0707 |                      -1 |         2.42688e+09 |          6.65022e+07 |                      4.08531e+09 |          7.5513e+06  | nan      |                      0 |              -0.0357 |            -5.2467 |                 2730.14 |                 790.457 |                -2.31239e+08 |                            -1.68762e+08 |                      7.0467 |                              172.442 |            0.6417 |                 25.8501 |                  -0.005  |              0.0314 |                   0.0688 |              1.00847e+10 |    7.42004e+09 |                   3 |            0.7332 |             0.659  |                             2.44445e+09 |                             2.41711e+09 |                        2.42688e+09 |                        2.45418e+09 |                                   0 |                                    1 |
        """
        self._check_arguments(period, close_column)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday" and self._historical_data[period].empty:
            raise InvalidArgumentError(
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
        breadth_indicators["Price Volume Trend"] = self.get_price_volume_trend(
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

        breadth_indicators["Negative Volume Index"] = self.get_volume_index(
            period=period, close_column=close_column
        )

        breadth_indicators["Positive Volume Index"] = self.get_volume_index(
            period=period, close_column=close_column, method="positive"
        )

        klinger_volume_oscillator = self.get_klinger_volume_oscillator(
            period=period, close_column=close_column
        )
        breadth_indicators["Klinger Volume Oscillator"] = klinger_volume_oscillator[
            "Klinger Volume Oscillator"
        ]
        breadth_indicators["Klinger Volume Oscillator Signal Line"] = (
            klinger_volume_oscillator["Signal Line"]
        )

        breadth_indicators["Market Facilitation Index"] = (
            self.get_market_facilitation_index(period=period, close_column=close_column)
        )
        breadth_indicators["Williams Accumulation/Distribution"] = (
            self.get_williams_accumulation_distribution(
                period=period, close_column=close_column
            )
        )
        breadth_indicators["Relative Volume"] = self.get_relative_volume(period=period)
        breadth_indicators["Volume Flow Indicator"] = self.get_volume_flow_indicator(
            period=period, close_column=close_column
        )
        breadth_indicators["Finite Volume Elements"] = self.get_finite_volume_elements(
            period=period, close_column=close_column
        )
        breadth_indicators["Twiggs Money Flow"] = self.get_chaikin_money_flow(
            period=period, close_column=close_column, method="twiggs"
        )
        breadth_indicators["Volume Zone Oscillator"] = self.get_volume_zone_oscillator(
            period=period, close_column=close_column
        )
        breadth_indicators["Cumulative Force Index"] = self.get_force_index(
            period=period, close_column=close_column, method="cumulative"
        )
        # The value traded is a market value, so it keeps the traded close.
        breadth_indicators["Price Volume"] = self.get_price_volume(period=period)
        breadth_indicators["Price Volume Rank"] = self.get_price_volume_rank(
            period=period, close_column=close_column
        )

        buying_selling_pressure = self.get_buying_selling_pressure(
            period=period, close_column=close_column
        )
        breadth_indicators["Buying Pressure"] = buying_selling_pressure[
            "Buying Pressure"
        ]
        breadth_indicators["Selling Pressure"] = buying_selling_pressure[
            "Selling Pressure"
        ]

        # The On-Balance Volume itself is already in the collection.
        archer_on_balance_volume = self.get_archer_on_balance_volume(
            period=period, close_column=close_column
        )
        for column in [
            "Fast Average",
            "Slow Average",
            "Minimum",
            "Maximum",
            "Long Run",
            "Short Run",
        ]:
            breadth_indicators[f"Archer On-Balance Volume {column}"] = (
                archer_on_balance_volume[column]
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
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the McClellan Oscillator: the momentum of the market's breadth, as the difference between a fast
        and a slow average of the advancing minus declining tickers.

        The McClellan Oscillator is the difference between a short-term and a long-term
        exponential moving average of the daily advance/decline signal. Traditionally it is
        calculated from the number of advancing minus declining stocks across a market; here
        it is applied to each ticker's own signal (+1 on a day it closed higher, -1 on a day it
        closed lower, 0 otherwise), so it measures how persistently that ticker has been rising
        or falling.

        The formula is as follows:

        - McClellan Oscillator = EMA(Advance/Decline Signal, short_ema_window) — EMA(Advance/Decline
          Signal, long_ema_window)

        For more information, see the following sources:

        - McClellan, S. and McClellan, M. (1970). "Patterns for Profit: The McClellan Oscillator and Summation
          Index." Trade Levels Inc.

        Also known as: McClellan oscillator, market breadth.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            short_ema_window (int, optional): The window size for the short-term EMA.
                Defaults to 19.
            long_ema_window (int, optional): The window size for the long-term EMA.
                Defaults to 39.
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
            pd.Series or pd.DataFrame: McClellan Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the McClellan Oscillator for each asset in the Toolkit instance.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["EWJ", "EWG", "EWU", "EWQ", "EWZ", "INDA", "FXI", "EWC", "EWA", "EWY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_mcclellan_oscillator().loc[:, ["EWJ", "EWG", "EWZ", "INDA", "FXI", "Benchmark"]].tail(5)
        ```

        Which returns:

        | Date       |     EWJ |    EWG |     EWZ |    INDA |    FXI |   Benchmark |
        |:-----------|--------:|-------:|--------:|--------:|-------:|------------:|
        | 2025-12-24 | -0.0357 | 0.1365 | -0.0693 |  0.0631 | 0.0419 |      0.0718 |
        | 2025-12-26 | -0.0875 | 0.1604 | -0.0203 | -0.002  | 0.0863 |      0.0008 |
        | 2025-12-29 | -0.0314 | 0.0801 | -0.0783 | -0.0576 | 0.0239 |     -0.06   |
        | 2025-12-30 | -0.0832 | 0.1111 | -0.0275 | -0.0049 | 0.0703 |     -0.1116 |
        | 2025-12-31 | -0.1272 | 0.037  | -0.0839 |  0.0402 | 0.0097 |     -0.1552 |
        """
        self._check_arguments(
            period,
            close_column,
            short_ema_window=short_ema_window,
            long_ema_window=long_ema_window,
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

        mcclellan_oscillator = calculate_per_ticker(
            historical_data,
            [close_column],
            breadth_model.get_mcclellan_oscillator,
            short_ema_window,
            long_ema_window,
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
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the advance/decline signal: whether each ticker closed higher (1) or lower (-1) than the period
        before, the building block of the market breadth measures.

        Traditionally advancers versus decliners is a market-wide count of rising against
        falling stocks; applied to a single ticker it becomes a directional signal: +1 on a
        day the ticker closed higher than the day before, -1 on a day it closed lower and 0
        on an unchanged day. It feeds the McClellan Oscillator; get_trin and
        get_new_highs_new_lows measure breadth across all tickers.

        The formula is as follows:

        - Signal = +1 if Close(t) > Close(t-1), -1 if Close(t) < Close(t-1), else 0

        Also known as: advance decline ratio, market breadth.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
            pd.Series or pd.DataFrame: Advance/decline signal values (+1, 0 or -1).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the advance/decline signal for each asset in the Toolkit instance.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["EWJ", "EWG", "EWU", "EWQ", "EWZ", "INDA", "FXI", "EWC", "EWA", "EWY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_advancers_decliners().loc[:, ["EWJ", "EWG", "EWZ", "INDA", "FXI", "Benchmark"]].tail(5)
        ```

        Which returns:

        | Date       |   EWJ |   EWG |   EWZ |   INDA |   FXI |   Benchmark |
        |:-----------|------:|------:|------:|-------:|------:|------------:|
        | 2025-12-24 |    -1 |     1 |    -1 |     -1 |     1 |           1 |
        | 2025-12-26 |    -1 |     1 |     1 |     -1 |     1 |          -1 |
        | 2025-12-29 |     1 |    -1 |    -1 |     -1 |    -1 |          -1 |
        | 2025-12-30 |    -1 |     1 |     1 |      1 |     1 |          -1 |
        | 2025-12-31 |    -1 |    -1 |    -1 |      1 |    -1 |          -1 |
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

        historical_data = self._historical_data[period]

        advancers_decliners = calculate_per_ticker(
            historical_data, [close_column], breadth_model.get_advancers_decliners
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
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the On-Balance Volume (OBV): a running total of volume that adds the volume of up periods and
        subtracts that of down periods, to show whether volume flows into or out of a security.

        The On-Balance Volume (OBV) is a technical indicator that uses volume flow to predict changes in stock price.
        It accumulates the volume on up days and subtracts the volume on down days. The resulting OBV line provides
        insights into the buying and selling pressure behind price movements.

        The formula is as follows:

        - OBV = Previous OBV + Volume if Close > Previous Close, Previous OBV — Volume if Close < Previous Close,
          and Previous OBV otherwise.

        The level of OBV has no meaning by itself, since it depends on the first date of the data; only its
        direction and its divergences from price matter. A rising OBV while price moves sideways is read as
        accumulation ahead of a breakout, a falling OBV as distribution. A single day of very high volume, such
        as an index rebalance or an earnings day, can move the line for a long time.

        For more information, see the following sources:

        - Granville, J.E. (1963). "Granville's New Key to Stock Market Profits." Prentice-Hall.

        Referenced against:
        - TA-Lib, OBV: https://ta-lib.org/functions/obv.html

        Also known as: OBV, volume momentum.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the OBV.
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
            pd.Series or pd.DataFrame: On-Balance Volume values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates On-Balance Volume
          for each asset in the Toolkit instance.
        - TA-Lib's OBV starts its running total at the first period's volume instead of 0, so it is
          the On-Balance Volume here plus that volume; every change is equal.
        - When growth is requested, the method calculates the growth of the OBV over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_on_balance_volume().tail(5)
        ```

        Which returns:

        | Date       |       BTCUSD |     Benchmark |
        |:-----------|-------------:|--------------:|
        | 2025-12-27 | -1.15162e+12 | nan           |
        | 2025-12-28 | -1.13646e+12 | nan           |
        | 2025-12-29 | -1.18487e+12 |   1.32079e+09 |
        | 2025-12-30 | -1.14929e+12 |   1.27363e+09 |
        | 2025-12-31 | -1.18312e+12 |   1.19948e+09 |
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

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._historical_data[period].loc[self._start_date :]

        on_balance_volume = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_on_balance_volume,
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
    def get_price_volume_trend(
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
        Calculate the Price Volume Trend (PVT): a running total of volume weighted by the percentage change of the
        close, which shows how strongly buying or selling pressure backs a trend.

        The Price Volume Trend is a running total of volume weighted by the period's percentage
        change in price. On-Balance Volume adds or subtracts the whole volume depending on the
        direction of the close; the PVT adds only the share of the volume that matches the size of
        the move, so a small change in price moves the line only a little.

        The formula is as follows:

        - PVT = Previous PVT + Volume × (Close — Previous Close) / Previous Close
        - PVT = 0 on the first date, which has no previous close

        The level of the PVT has no meaning by itself, since it starts at 0 on the start date; only
        its direction and its divergences from price matter. A rising PVT confirms an uptrend carried
        by volume, and a PVT that fails to follow price to a new high warns that the move lacks
        participation. The line is in shares, so it cannot be compared across stocks.

        For more information, see the following sources:

        - Granville, J.E. (1963). "Granville's New Key to Stock Market Profits." Prentice-Hall, the On-Balance
          Volume the PVT refines.

        Referenced against:
        - TA-Lib, PVT: https://ta-lib.org/functions/pvt.html

        Also known as: PVT, Price and Volume Trend, Volume Price Trend.

        See `breadth_model.get_price_volume_trend` for the references.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", so that the change in price is the return including dividends.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the PVT.
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
            pd.Series or pd.DataFrame: Price Volume Trend values, in shares.

        Notes:
        - The running total starts at 0 on each ticker's first date with a close and a volume on or
          after the start date of the Toolkit, as in TA-Lib's PVT, which it equals.
        - When growth is requested, the method calculates the growth of the PVT over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_price_volume_trend().tail(5)
        ```

        Which returns:

        | Date       |        AAPL |        MSFT |    Benchmark |
        |:-----------|------------:|------------:|-------------:|
        | 2025-12-24 | 6.67008e+07 | 2.02096e+07 | -9.46523e+06 |
        | 2025-12-26 | 6.66686e+07 | 2.02039e+07 | -9.46945e+06 |
        | 2025-12-29 | 6.66998e+07 | 2.01903e+07 | -9.69239e+06 |
        | 2025-12-30 | 6.66449e+07 | 2.02012e+07 | -9.74998e+06 |
        | 2025-12-31 | 6.65229e+07 | 2.00776e+07 | -1.02993e+07 |
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

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._historical_data[period].loc[self._start_date :]

        price_volume_trend = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_price_volume_trend,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=price_volume_trend,
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
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Accumulation/Distribution Line: a running total of volume weighted by where each close lies
        within its high-low range, to show whether a security is being accumulated or distributed.

        The Accumulation/Distribution Line is a technical indicator that evaluates the flow of money
        into or out of an asset. It takes into account both price and volume information to identify
        whether an asset is being accumulated (bought) or distributed (sold) by investors.

        The formula is as follows:

        - Money Flow Multiplier = ((Close — Low) — (High — Close)) / (High — Low)
        - Money Flow Volume = Money Flow Multiplier * Volume
        - ADL = Previous ADL + Money Flow Volume

        For more information, see the following sources:

        - Chaikin, M., who published no formal paper on it; the standard textbook treatment is Murphy, J.J.
          (1999). "Technical Analysis of the Financial Markets." New York Institute of Finance.

        Referenced against:
        - TA-Lib, AD: https://ta-lib.org/functions/ad.html

        Also known as: ADL, Chaikin ADL, volume-price trend.

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
            growth (bool, optional): Whether to calculate the growth of the Accumulation/Distribution Line.
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
            pd.Series or pd.DataFrame: Accumulation/Distribution Line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Accumulation/Distribution Line for each asset in the Toolkit instance.
        - The Accumulation/Distribution Line agrees with TA-Lib's AD.
        - When growth is requested, the method calculates the growth of the Accumulation/Distribution Line
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_accumulation_distribution_line().tail(5)
        ```

        Which returns:

        | Date       |         GLD |   Benchmark |
        |:-----------|------------:|------------:|
        | 2025-12-24 | 7.95203e+08 | 8.01111e+09 |
        | 2025-12-26 | 7.95996e+08 | 8.00572e+09 |
        | 2025-12-29 | 7.9136e+08  | 8.01431e+09 |
        | 2025-12-30 | 7.82463e+08 | 7.98763e+09 |
        | 2025-12-31 | 7.75502e+08 | 7.919e+09   |
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

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._get_historical_data(period, close_column).loc[
            self._start_date :
        ]

        accumulation_distribution_line = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_accumulation_distribution_line,
        ).loc[self._start_date : self._end_date]
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
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chaikin Oscillator: the momentum of the Accumulation/Distribution Line, as the difference
        between its fast and slow exponential moving averages.

        The Chaikin Oscillator is a momentum-based indicator that combines price and volume
        to help identify potential trends and reversals in the market. It is calculated as the
        difference between the 3-day and 10-day Accumulation/Distribution Line.

        The formula is as follows:

        - Chaikin Oscillator = EMA(short-window ADL) — EMA(long-window ADL)

        For more information, see the following sources:

        - Chaikin, M., who published no formal paper on it; the standard textbook treatment is Murphy, J.J.
          (1999). "Technical Analysis of the Financial Markets." New York Institute of Finance.

        Referenced against:
        - TA-Lib, ADOSC: https://ta-lib.org/functions/adosc.html

        Also known as: Chaikin oscillator, volume accumulation.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            short_window (int, optional): Number of periods for the short-term moving average.
                Defaults to 3.
            long_window (int, optional): Number of periods for the long-term moving average.
                Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Chaikin Oscillator.
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
            pd.Series or pd.DataFrame: Chaikin Oscillator values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Chaikin Oscillator for each asset in the Toolkit instance.
        - Both EMAs are seeded with the simple average of their first values, as on StockCharts,
          while TA-Lib's ADOSC starts them from the first Accumulation/Distribution value, so the
          first values differ from TA-Lib's until the two converge (within 0.01% after about 65
          periods with the default windows).
        - When growth is requested, the method calculates the growth of the Chaikin Oscillator
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chaikin_oscillator().tail(5)
        ```

        Which returns:

        | Date       |          QQQ |   Benchmark |
        |:-----------|-------------:|------------:|
        | 2025-12-24 |  1.33735e+07 | 4.48979e+07 |
        | 2025-12-26 |  1.24876e+07 | 4.67752e+07 |
        | 2025-12-29 |  1.15765e+07 | 4.60255e+07 |
        | 2025-12-30 |  1.62538e+06 | 3.30466e+07 |
        | 2025-12-31 | -1.44174e+07 | 2.89497e+06 |
        """
        self._check_arguments(
            period, close_column, short_window=short_window, long_window=long_window
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

        chaikin_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_chaikin_oscillator,
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
        window: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        method: Literal["standard", "twiggs"] = "standard",
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chaikin Money Flow (CMF) or Colin Twiggs' Money Flow (TMF): the share of volume over a window
        that came with closes near the highs rather than the lows, a measure of buying versus selling pressure.

        The Chaikin Money Flow sums the same Money Flow Volume used by the Accumulation/
        Distribution Line over a rolling window and normalizes it by the window's total
        volume, turning the running (unbounded) Accumulation/Distribution Line into a
        bounded oscillator. Sustained readings above zero indicate buying pressure
        (accumulation) is dominating over the window, while sustained readings below zero
        indicate selling pressure (distribution).

        Twiggs Money Flow refines the Chaikin Money Flow. It places the close
        within the true range (which includes a gap from the previous close) instead of the
        high-low range, so a gap does not show up as a close at the top or bottom of a narrow
        range, and it smooths the money flow volume and the volume with Wilder's moving average
        instead of summing them over a window, so a large volume does not distort the result when
        it leaves the window. Twiggs used crossings of zero and divergences from price as signals.

        The formula is as follows, per method:

        - Chaikin Money Flow: Money Flow Multiplier = ((Close — Low) — (High — Close)) / (High — Low),
          Money Flow Volume = Money Flow Multiplier * Volume and
          CMF = Sum(Money Flow Volume, window) / Sum(Volume, window)
        - Twiggs Money Flow: True High = max(High, Close(t-1)), True Low = min(Low, Close(t-1)),
          Money Flow Volume = Volume * ((Close — True Low) — (True High — Close)) /
          (True High — True Low) and TMF = Wilder MA(Money Flow Volume, window) /
          Wilder MA(Volume, window)

        For more information, see the following sources:

        - Twiggs, C. (2003). "Twiggs Money Flow." Incredible Charts
          (https://www.incrediblecharts.com/indicators/twiggs_money_flow.php).

        Referenced against:
        - TA-Lib, CMF: https://ta-lib.org/functions/cmf.html

        Also known as: CMF, Chaikin Money Flow, TMF, Twiggs Money Flow.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int | None, optional): Number of periods to sum the Money Flow Volume and
                volume over, or of Wilder's moving average with method="twiggs". Defaults to None,
                which uses 20 for "standard" and 21 for "twiggs".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Chaikin Money Flow.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): The variant. Options are: "standard" (the Chaikin Money Flow),
                "twiggs" (Twiggs Money Flow). Defaults to "standard".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame: Chaikin Money Flow (or Twiggs Money Flow) values, bounded
            between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Chaikin Money Flow for each asset in the Toolkit instance.
        - There is no formal journal citation for the Chaikin Money Flow; the standard
          textbook treatment is Murphy, J.J. (1999). "Technical Analysis of the Financial
          Markets." New York Institute of Finance.
        - The Chaikin Money Flow agrees with TA-Lib's CMF.
        - Twiggs Money Flow: TA-Lib, pandas-ta-classic, ta and Tulip Indicators do not have it, and finta's
          TMF is an unfinished stub, so it follows Twiggs' definition. Wilder's moving average starts
          from the simple average of its first `window` values, and the first period, without a
          previous close, uses its own high-low range.
        - When growth is requested, the method calculates the growth of the Chaikin Money
          Flow over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["ETHUSD", "SOLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_chaikin_money_flow().tail(5)
        ```

        Which returns:

        | Date       |   ETHUSD |   SOLUSD |   Benchmark |
        |:-----------|---------:|---------:|------------:|
        | 2025-12-27 |  -0.0938 |  -0.143  |    nan      |
        | 2025-12-28 |  -0.1006 |  -0.0945 |    nan      |
        | 2025-12-29 |  -0.1815 |  -0.1419 |      0.0671 |
        | 2025-12-30 |  -0.1279 |  -0.0749 |      0.0614 |
        | 2025-12-31 |  -0.1646 |  -0.1521 |      0.017  |
        """
        self._check_arguments(period, close_column, window=window)
        methods = ("standard", "twiggs")
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
            window = 21 if method == "twiggs" else 20

        historical_data = self._get_historical_data(period, close_column)

        chaikin_money_flow = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            (
                breadth_model.get_twiggs_money_flow
                if method == "twiggs"
                else breadth_model.get_chaikin_money_flow
            ),
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
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Ease of Movement (EMV): how far the price moves per unit of volume, so a large value shows a
        price rising or falling on little volume.

        The Ease of Movement indicator relates how far price moved (the change in the
        midpoint of the high-low range from one period to the next) to the volume required
        to move it (via the "Box Ratio", volume scaled down and divided by the period's
        high-low range). High positive readings mean price is moving up easily on relatively
        little volume; high negative readings mean price is moving down easily on relatively
        little volume. The raw daily reading is smoothed with a Simple Moving Average to
        reduce noise.

        The formula is as follows:

        - Distance Moved = (High(t) + Low(t)) / 2 — (High(t-1) + Low(t-1)) / 2
        - Box Ratio = (Volume / volume_divisor) / (High — Low)
        - Raw EMV = Distance Moved / Box Ratio
        - EMV = SMA(Raw EMV, window)

        Also known as: EMV, Ease of Movement.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): Number of periods used to smooth the raw Ease of
                Movement values with a Simple Moving Average. Defaults to 14.
            volume_divisor (float, optional): Scaling constant applied to volume so that the
                Box Ratio (and therefore EMV) stays in a readable range regardless of an
                asset's typical share volume. Defaults to 100,000,000.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Ease of Movement.
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
            pd.Series or pd.DataFrame: Ease of Movement values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ease of Movement for each asset in the Toolkit instance.
        - Reference: Arms, R.W. (1989). "The Arms Index (TRIN): An Introduction to the
          Volume Analysis of Stock and Bond Markets." Dow Jones-Irwin.
        - The Ease of Movement agrees with TA-Lib's EMV with its `volumedivisor` set to `volume_divisor`
          (TA-Lib's default is 10,000).
        - When growth is requested, the method calculates the growth of the Ease of
          Movement over the given lag.

        Referenced against:
        - TA-Lib, EMV: https://ta-lib.org/functions/emv.html

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_ease_of_movement().tail(5)
        ```

        Which returns:

        | Date       |   ^GSPC |   Benchmark |
        |:-----------|--------:|------------:|
        | 2025-12-24 |  1.404  |      2.8161 |
        | 2025-12-26 |  0.6053 |      2.227  |
        | 2025-12-29 |  0.6173 |      2.7602 |
        | 2025-12-30 |  0.6371 |      2.7794 |
        | 2025-12-31 | -4.2755 |      0.4354 |
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

        ease_of_movement = calculate_per_ticker(
            historical_data,
            ["High", "Low", "Volume"],
            breadth_model.get_ease_of_movement,
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
    def get_volume_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        start_value: float = 1000.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        method: Literal["negative", "positive"] = "negative",
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Negative or the Positive Volume Index (NVI or PVI), which separates the trades of informed
        investors from those of the crowd.

        It is a running total of the price changes on the periods volume falls (NVI) or rises (PVI).

        The Negative Volume Index is a cumulative index that only updates on days where
        volume decreases from the prior period, compounding that day's percentage price
        change onto the running index; on days where volume increases (or stays flat), the
        index is carried forward unchanged. The premise, per Fosback, is that "smart money"
        tends to be active on low-volume (quiet) days, so tracking price behaviour
        specifically on those days isolates informed trading from the noise of high-volume,
        crowd-driven days. The Positive Volume Index mirrors it: it only updates
        on days where volume increases from the prior period, and is carried forward unchanged on
        days where volume decreases (or stays flat). Per Fosback, it isolates price behaviour on
        high-volume (crowd-driven) days, which is traditionally read as tracking less-informed,
        sentiment-driven trading.

        The formula is as follows, per method:

        - Negative Volume Index: Index(t) = Index(t-1) * (1 + (Close(t) / Close(t-1) — 1)) if
          Volume(t) < Volume(t-1), and Index(t) = Index(t-1) otherwise
        - Positive Volume Index: Index(t) = Index(t-1) * (1 + (Close(t) / Close(t-1) — 1)) if
          Volume(t) > Volume(t-1), and Index(t) = Index(t-1) otherwise

        For more information, see the following sources:

        - Fosback, N.G. (1976). "Stock Market Logic: A Sophisticated Approach to Profits on Wall
          Street." The Institute for Econometric Research.

        Referenced against:
        - TA-Lib, NVI: https://ta-lib.org/functions/nvi.html
        - TA-Lib, PVI: https://ta-lib.org/functions/pvi.html

        Also known as: NVI, Negative Volume Index, PVI, Positive Volume Index, volume index.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            start_value (float, optional): The index value to start the series at.
                Defaults to 1000.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the volume index.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            method (str, optional): The index. Options are: "negative" (the Negative Volume Index),
                "positive" (the Positive Volume Index). Defaults to "negative".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series or pd.DataFrame: Negative (or Positive) Volume Index values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the volume index for each asset in the Toolkit instance.
        - With the default `start_value` of 1,000, the Negative Volume Index agrees with TA-Lib's NVI
          and the Positive Volume Index with TA-Lib's PVI.
        - When growth is requested, the method calculates the growth of the volume index over
          the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["SPY", "IWM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.technicals.get_volume_index().tail(5)
        ```

        Which returns:

        | Date       |     SPY |     IWM |
        |:-----------|--------:|--------:|
        | 2025-12-24 | 5440.48 | 4501.78 |
        | 2025-12-26 | 5440.48 | 4501.78 |
        | 2025-12-29 | 5440.48 | 4501.78 |
        | 2025-12-30 | 5433.84 | 4468.59 |
        | 2025-12-31 | 5433.84 | 4468.59 |
        """
        self._check_arguments(period, close_column)
        methods = ("negative", "positive")
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

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._historical_data[period].loc[self._start_date :]

        volume_index = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            (
                breadth_model.get_positive_volume_index
                if method == "positive"
                else breadth_model.get_negative_volume_index
            ),
            start_value,
        ).loc[self._start_date : self._end_date]
        return finalize_dataset(
            dataset=volume_index,
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
    def get_trin(
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
        Calculate the TRIN (Arms Index): the ratio of advancing to declining tickers divided by the ratio of their
        volumes, which shows whether volume favours the rising or the falling tickers.

        TRIN compares the ratio of advancing to declining issues against the ratio of
        volume in advancing issues to volume in declining issues. It is a market-wide
        breadth reading computed across all tickers in the Toolkit instance (excluding
        the synthetic "Portfolio" and "Benchmark" columns), and the resulting single
        reading is broadcast to every ticker column so it lines up with the other
        breadth indicators.

        The formula is as follows:

        - TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)

        A TRIN below 1 means the volume in rising stocks outweighs their number, which is read
        as buying pressure, and above 1 as selling pressure; readings far above 2 tend to come
        with capitulation days. It is meant for a broad universe, such as all stocks of an
        exchange. Calculated across a handful of tickers it jumps between extremes and is
        undefined on days when none of them fell.

        A ticker counts on the dates it traded, with its change since its own previous close:
        on a date only a cryptocurrency in the Toolkit traded, such as a weekend, only the
        cryptocurrency counts, and a stock is never counted as unchanged on it.

        For more information, see the following sources:

        - Arms, R.W. (1967), who introduced the index in Barron's as the Short-Term Trading Index.
        - Arms, R.W. (1989). "The Arms Index (TRIN): An Introduction to the Volume Analysis of Stock and Bond
          Markets." Dow Jones-Irwin.

        Also known as: Arms Index, TRIN.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
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
            pd.Series or pd.DataFrame: TRIN values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the TRIN across all tickers in the Toolkit instance.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # TRIN is one market-wide reading, repeated in every ticker column
        toolkit.technicals.get_trin().loc[:, ["XLK"]].tail(5)
        ```

        Which returns:

        | Date       |      XLK |
        |:-----------|---------:|
        | 2025-12-24 |   2.485  |
        | 2025-12-26 |   1.6498 |
        | 2025-12-29 |   0.6578 |
        | 2025-12-30 |   0.6376 |
        | 2025-12-31 | nan      |
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

        historical_data = self._historical_data[period]

        constituents = [
            ticker
            for ticker in historical_data[close_column].columns
            if ticker not in ("Portfolio", "Benchmark")
        ]

        # NaN on the dates a constituent did not trade, rather than its carried-forward close, so it is not counted as unchanged on another ticker's trading days.  # noqa: E501
        closes = historical_data[close_column].mask(
            get_non_trading_days(historical_data, close_column)
        )

        trin_series = breadth_model.get_trin(
            closes[constituents],
            historical_data["Volume"][constituents],
        ).loc[self._start_date : self._end_date]

        # The ratios divide by the declining tickers and their volume, so a date without both
        # an advancing and a declining ticker has no TRIN.
        if not trin_series.empty and trin_series.isna().all():
            logger.warning(
                "The TRIN has no values: it needs a ticker that advances and one that declines on "
                "the same date (the Benchmark is not counted), %s.",
                (
                    "which fewer than two tickers cannot give"
                    if len(constituents) < 2  # noqa: PLR2004
                    else "and no date has both, as with a constant price"
                ),
            )

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
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the New Highs - New Lows: the number of tickers at a new high over a window minus the number at a
        new low, which shows how broad a rise or a fall is.

        New Highs - New Lows measures the number of tickers reaching a new high over
        the specified window minus the number of tickers reaching a new low over the
        same window. It is a market-wide breadth reading computed across all tickers in
        the Toolkit instance (excluding the synthetic "Portfolio" and "Benchmark"
        columns), and the resulting single reading is broadcast to every ticker column
        so it lines up with the other breadth indicators.

        The formula is as follows:

        - New Highs - New Lows = (Number of tickers at a window-period high) - (Number of tickers at a window-period low)

        The reading counts tickers at a high or low of the lookback window, so it needs a broad
        universe and a history of at least one window before the start of the period of interest: the
        reading is NaN until a ticker has a full window of closes. The default window of 252 periods is
        a year of daily closes; weekly data needs a shorter window, such as 52 periods for a year.
        Persistently more new lows than new highs while an index still rises is a classic sign of
        narrowing breadth.

        Each ticker's window counts its own closes and the ticker counts on the dates it traded,
        so the weekends of a cryptocurrency in the Toolkit do not shorten a stock's window.

        For more information, see the following sources:

        - Colby, R.W. (2003). "The Encyclopedia of Technical Market Indicators." 2nd ed. McGraw-Hill.

        Also known as: new highs minus new lows, record high percent.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            window (int, optional): Number of periods for the new high / new low lookback.
                Defaults to 252, a year of daily closes (52 is a year of weekly closes).
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
            pd.Series or pd.DataFrame: New Highs - New Lows values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the New Highs - New Lows across all tickers in the Toolkit instance.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-10-01",
            end_date="2025-12-31",
        )

        # New Highs - New Lows is one market-wide reading, repeated in every ticker column
        toolkit.technicals.get_new_highs_new_lows(window=20).loc[:, ["XLK"]].tail(5)
        ```

        Which returns:

        | Date       |   XLK |
        |:-----------|------:|
        | 2025-12-24 |     5 |
        | 2025-12-26 |     1 |
        | 2025-12-29 |     0 |
        | 2025-12-30 |     2 |
        | 2025-12-31 |     0 |
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

        constituents = [
            ticker
            for ticker in historical_data[close_column].columns
            if ticker not in ("Portfolio", "Benchmark")
        ]

        # NaN on the dates a constituent did not trade, so its window counts its own closes.
        closes = historical_data[close_column].mask(
            get_non_trading_days(historical_data, close_column)
        )

        new_highs_new_lows_series = breadth_model.get_new_highs_new_lows(
            closes[constituents],
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

    @handle_portfolio
    @handle_errors
    def get_klinger_volume_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 34,
        long_window: int = 55,
        signal_window: int = 13,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Klinger Volume Oscillator (KVO): the difference between a fast and a slow average of the
        volume force, which compares the volume of up and down trends to show long-term money flow.

        The Klinger Volume Oscillator turns each period's volume into a signed "volume force": positive
        when the sum of the high, low and close rose from the period before (an up trend), negative when
        it fell, and weighted by how the period's range compares with the range accumulated since the
        trend last turned. The oscillator is the difference between a short and a long exponential moving
        average of that force, with a signal line to show when it turns.

        The formula is as follows:

        - Trend = +1 if High + Low + Close > the previous High + Low + Close, —1 if lower, else the previous Trend
        - dm = High — Low
        - cm = cm(t-1) + dm if the Trend is unchanged, else dm(t-1) + dm
        - Volume Force = Volume * |2 * dm / cm — 1| * Trend * 100
        - KVO = EMA(Volume Force, short_window) — EMA(Volume Force, long_window)
        - Signal Line = EMA(KVO, signal_window)

        Klinger designed it to follow the long-term flow of money while staying sensitive to short-term
        reversals: the KVO crossing its signal line, or diverging from price at a new high or low, is the
        usual signal. Its level is in units of volume and depends on how many shares trade, so it is not
        comparable across tickers.

        For more information, see the following sources:

        - Klinger, S.J. (1997). "Identifying Trends With Volume Analysis." Technical Analysis of Stocks & Commodities, 15(12).

        Referenced against:
        - TA-Lib, KVO: https://ta-lib.org/functions/kvo.html
        - Tulip Indicators, kvo: https://tulipindicators.org/kvo

        Also known as: KVO, Klinger Oscillator, Klinger Volume Force.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            short_window (int, optional): The number of periods for the short-term EMA of the volume force.
                Defaults to 34.
            long_window (int, optional): The number of periods for the long-term EMA of the volume force.
                Defaults to 55.
            signal_window (int, optional): The number of periods for the signal line. Defaults to 13.
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
            pd.Series or pd.DataFrame: The Klinger Volume Oscillator and its Signal Line, in units of volume.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the KVO for
          each asset in the Toolkit instance.
        - The volume force agrees with that of TA-Lib (and Tulip Indicators). Every EMA is seeded with the simple average
          of its first values, so the KVO starts on the 56th period, as TA-Lib's, and the signal line 12
          periods later. TA-Lib seeds the averages with the first volume force instead; on daily data of
          AAPL and MSFT the two agree to 1% of the typical KVO after about 210 periods and to 0.0001% after
          about 470. pandas-ta-classic's `kvo` uses a simplified force (the volume signed by the change in
          the typical price) and differs.
        - When growth is requested, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_klinger_volume_oscillator().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Klinger Volume Oscillator |   Signal Line |
        |:-----------|----------------------------:|--------------:|
        | 2025-12-24 |                -2.16282e+08 |  -1.35548e+08 |
        | 2025-12-26 |                -1.90376e+08 |  -1.43381e+08 |
        | 2025-12-29 |                -1.91502e+08 |  -1.50255e+08 |
        | 2025-12-30 |                -2.06931e+08 |  -1.58352e+08 |
        | 2025-12-31 |                -2.31241e+08 |  -1.68765e+08 |
        """
        self._check_arguments(
            period,
            close_column,
            short_window=short_window,
            long_window=long_window,
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

        klinger_volume_oscillator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_klinger_volume_oscillator,
            short_window,
            long_window,
            signal_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=klinger_volume_oscillator,
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
    def get_market_facilitation_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        volume_divisor: float = 100_000_000,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Bill Williams' Market Facilitation Index (BW MFI): the range of a period per unit of volume, which
        shows how efficiently volume moves the price.

        The Market Facilitation Index measures how far price moved per unit of volume: the period's range
        divided by its volume. Williams read it together with the change in volume: a rising index on
        rising volume ("green") shows a market that moves easily with new participants, a falling index on
        falling volume ("fade") a market that loses interest, a falling index on rising volume ("squat") a
        struggle between buyers and sellers that often comes before a breakout, and a rising index on
        falling volume ("fake") a move without participation.

        The formula is as follows:

        - Market Facilitation Index = (High — Low) / (Volume / volume_divisor)

        The index depends on the price and the number of shares traded, so it is read against its own
        history rather than compared across tickers.

        For more information, see the following sources:

        - Williams, B.M. (1995). "Trading Chaos: Applying Expert Techniques to Maximize Your Profits." John Wiley & Sons.

        Referenced against:
        - TA-Lib, MARKETFI: https://ta-lib.org/functions/marketfi.html
        - pandas-ta-classic, marketfi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/marketfi.py#L16
        - Tulip Indicators, marketfi: https://tulipindicators.org/marketfi

        Also known as: BW MFI, Market Facilitation Index, MARKETFI.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            volume_divisor (float, optional): Scaling constant the volume is divided by, so the index stays in
                a readable range at the Toolkit's rounding. Defaults to 100,000,000, as for the Ease of Movement.
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
            pd.Series or pd.DataFrame: Market Facilitation Index values, the range per 100 million shares
            traded by default.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the Market
          Facilitation Index for each asset in the Toolkit instance.
        - With a volume divisor of 1 it agrees with TA-Lib's MARKETFI, pandas-ta-classic's `marketfi` and Tulip
          Indicators' `marketfi`, whose index for a stock trading millions of shares is a fraction of a
          millionth. A period without volume has no index (NaN) rather than an infinite one.
        - When growth is requested, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_market_facilitation_index().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 17.9714 | 73.3347 |      7.6219 |
        | 2025-12-26 | 11.6221 | 24.2275 |      5.6988 |
        | 2025-12-29 |  8.4461 | 37.9654 |      4.9645 |
        | 2025-12-30 |  8.102  | 29.7295 |      4.1659 |
        | 2025-12-31 |  7.0467 | 30.7675 |      7.5612 |
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

        market_facilitation_index = calculate_per_ticker(
            historical_data,
            ["High", "Low", "Volume"],
            breadth_model.get_market_facilitation_index,
            volume_divisor,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=market_facilitation_index,
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
    def get_williams_accumulation_distribution(
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
        Calculate Larry Williams' Accumulation/Distribution (WAD): a running total of how far each close moved from
        the opposite end of its true range, which shows accumulation and distribution from prices alone.

        Williams' Accumulation/Distribution is a running total of how far each close moved away from the
        opposite end of the period's true range: on an up close the distance from the true low
        (accumulation), on a down close the distance from the true high (distribution). Unlike Chaikin's
        Accumulation/Distribution Line it uses no volume.

        The formula is as follows:

        - True High = max(High, Close(t-1)), True Low = min(Low, Close(t-1))
        - A/D(t) = Close — True Low if Close > Close(t-1), Close — True High if Close < Close(t-1), else 0
        - WAD(t) = WAD(t-1) + A/D(t), starting at 0 on the start date

        Williams traded the divergences: price making a new high while the line does not points to
        distribution, price making a new low while the line holds up to accumulation. The level of the line
        depends on the start date; only its direction and divergences carry meaning.

        For more information, see the following sources:

        - Williams, L.R. (1973). "How I Made One Million Dollars Last Year Trading Commodities." Windsor Books.

        Referenced against:
        - TA-Lib, WAD: https://ta-lib.org/functions/wad.html
        - pandas-ta-classic, wad: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/wad.py#L13
        - Tulip Indicators, wad: https://tulipindicators.org/wad

        Also known as: WAD, Williams A/D, Williams Accumulation Distribution.

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
            pd.Series or pd.DataFrame: Williams' Accumulation/Distribution values, in price units.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates Williams'
          Accumulation/Distribution for each asset in the Toolkit instance, as a running total from the
          start date.
        - It agrees with TA-Lib's WAD (which also starts at 0), Tulip Indicators' `wad` (which starts on the
          second period) and pandas-ta-classic's `wad` (NaN on the first period).
        - When growth is requested, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_williams_accumulation_distribution().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 175.81  | 316.48  |     368.326 |
        | 2025-12-26 | 173.847 | 316.073 |     366.986 |
        | 2025-12-29 | 175.252 | 314.834 |     364.546 |
        | 2025-12-30 | 174.256 | 316.797 |     363.007 |
        | 2025-12-31 | 172.442 | 312.315 |     357.61  |
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

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._get_historical_data(period, close_column).loc[
            self._start_date :
        ]

        williams_accumulation_distribution = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            breadth_model.get_williams_accumulation_distribution,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=williams_accumulation_distribution,
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
    def get_relative_volume(
        self,
        period: str = "daily",
        window: int = 20,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Volume (RVOL): the volume of a period divided by its average over a window, which
        shows how unusually busy the period is.

        Relative Volume compares a period's volume with the average volume of the periods before it. A
        value of 2 means twice the usual volume: a breakout or a gap on high relative volume carries more
        conviction than one on thin trading, and traders use it to screen for stocks "in play".

        The formula is as follows:

        - RVOL = Volume(t) / SMA(Volume, window)(t-1)

        The average covers the `window` periods before the current one, so a spike does not raise its own
        benchmark. As a ratio it can be compared across tickers whatever the number of shares they trade.

        For more information, see the following sources:

        - Bulkowski, T.N. (2005). "Encyclopedia of Chart Patterns." 2nd ed. John Wiley & Sons.

        Referenced against:
        - TA-Lib, RVOL: https://ta-lib.org/functions/rvol.html

        Also known as: RVOL, Relative Volume, volume ratio.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            window (int, optional): The number of earlier periods to average the volume over. Defaults to 20.
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
            pd.Series or pd.DataFrame: Relative Volume values, as a ratio (1 is the average volume).

        Notes:
        - The method retrieves the volumes of the specified `period` and calculates the Relative Volume for
          each asset in the Toolkit instance.
        - It agrees with TA-Lib's RVOL. A ticker without volume, such as an index or a currency pair, has no
          Relative Volume (NaN).
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["NVDA", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_relative_volume().tail(5)
        ```

        Which returns:

        | Date       |   NVDA |   TSLA |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.353  | 0.5386 |      0.4951 |
        | 2025-12-26 | 0.8084 | 0.7825 |      0.5363 |
        | 2025-12-29 | 0.7032 | 0.8849 |      0.8223 |
        | 2025-12-30 | 0.5726 | 0.7756 |      0.6145 |
        | 2025-12-31 | 0.7232 | 0.6418 |      0.975  |
        """
        self._check_arguments(period, window=window)
        check_period(
            period, ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
        )
        if period == "intraday" and self._historical_data[period].empty:
            raise InvalidArgumentError(
                "Please define the 'intraday_period' parameter when initializing the Toolkit."
            )

        historical_data = self._historical_data[period]

        relative_volume = calculate_per_ticker(
            historical_data, ["Volume"], breadth_model.get_relative_volume, window
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=relative_volume,
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
    def get_volume_flow_indicator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 130,
        coefficient: float = 0.2,
        volume_coefficient: float = 2.5,
        smooth_window: int = 3,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Markos Katsanos' Volume Flow Indicator (VFI): the volume of the periods with a meaningful move of
        the typical price, signed by the direction of the move, which shows whether money flows in or out.

        The Volume Flow Indicator refines On-Balance Volume in three ways: it uses the typical price instead
        of the close, it ignores moves too small to matter (below a cutoff that scales with the recent
        volatility of the typical price) and it caps the volume of a single period at a multiple of the
        average volume, so one spike cannot dominate. The sum of that signed volume over the window,
        divided by the average volume, is smoothed with a short EMA.

        The formula is as follows:

        - Typical Price = (High + Low + Close) / 3
        - Volatility = Population Standard Deviation of ln(Typical Price / Typical Price(t-1)) over 30 periods
        - Cutoff = coefficient * Volatility * Close
        - Average Volume = SMA(Volume, window)(t-1)
        - Capped Volume = min(Volume, volume_coefficient * Average Volume)
        - Signed Volume = +Capped Volume if Typical Price — Typical Price(t-1) > Cutoff, —Capped Volume if
          it is below —Cutoff, else 0
        - VFI = EMA(Sum(Signed Volume, window) / Average Volume, smooth_window)

        Katsanos read a VFI above zero as a bullish flow of money and below zero as a bearish one, a
        crossing of zero as a change in trend and a divergence from price as an early warning. The VFI is
        in units of the average volume (a sum of up to `window` periods), so it is comparable across
        tickers.

        For more information, see the following sources:

        - Katsanos, M. (2004). "Using Money Flow to Stay with the Trend." Technical Analysis of Stocks & Commodities, 22(6).
        - Katsanos, M. (2004). "Volume Flow Indicator Performance." Technical Analysis of Stocks & Commodities, 22(7).

        Referenced against:
        - pandas-ta-classic, vfi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/vfi.py#L17

        Shares similar logic:
        - finta, VFI: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L2026

        Also known as: VFI, Volume Flow Indicator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for the average volume and the sum of signed volume.
                Defaults to 130, about six months of trading days.
            coefficient (float, optional): The multiple of the volatility times the close that a move of the
                typical price needs to count. Defaults to 0.2.
            volume_coefficient (float, optional): The multiple of the average volume that a single period's
                volume is capped at. Defaults to 2.5.
            smooth_window (int, optional): The number of periods for the EMA that smooths the result.
                Defaults to 3.
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
            pd.Series or pd.DataFrame: Volume Flow Indicator values, in units of the average volume.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the VFI for
          each asset in the Toolkit instance. With the default window it needs 261 periods before its first
          value, so start a year earlier than the dates of interest.
        - The volatility is the population standard deviation (ddof=0), as in Katsanos' MetaStock formula:
          it only sets the cutoff of the indicator's definition, so it follows the primary source, as the
          Bollinger Bands do. The VFI agrees with pandas-ta-classic's `vfi` exactly. finta uses the sample
          deviation (ddof=1), a 1.7% larger cutoff, and differs wherever a move falls between the two
          cutoffs.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_volume_flow_indicator().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 27.2483 | -5.4611 |     18.9447 |
        | 2025-12-26 | 26.5619 | -6.4634 |     19.1226 |
        | 2025-12-29 | 26.285  | -7.5029 |     18.3944 |
        | 2025-12-30 | 25.8204 | -8.4473 |     17.6238 |
        | 2025-12-31 | 25.851  | -9.7919 |     16.2389 |
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

        volume_flow_indicator = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_volume_flow_indicator,
            window,
            coefficient,
            volume_coefficient,
            smooth_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=volume_flow_indicator,
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
    def get_finite_volume_elements(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 22,
        cutoff: float = 0.003,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate Markos Katsanos' Finite Volume Elements (FVE): the share of volume over a window classified as
        buying rather than selling, leaving out moves too small to matter, a measure of money flow.

        Finite Volume Elements is a money flow indicator that classifies each period's whole volume as
        buying, selling or neutral. It looks at both the intraday move (the close against the middle of the
        range) and the interday move (the change in the typical price), and leaves out periods whose
        combined move is smaller than a cutoff, so noise does not count as flow. The result is the net
        signed volume as a share of all volume over the window.

        The formula is as follows:

        - Typical Price = (High + Low + Close) / 3
        - Money Flow = Close — (High + Low) / 2 + Typical Price — Typical Price(t-1)
        - Signed Volume = +Volume if Money Flow > cutoff * Close, —Volume if Money Flow < —cutoff * Close, else 0
        - FVE = Sum(Signed Volume, window) / Sum(Volume, window)

        A rising FVE while price moves sideways points to accumulation ahead of a breakout, a falling one to
        distribution. As a share of volume it lies between -1 and 1 and is comparable across tickers.

        For more information, see the following sources:

        - Katsanos, M. (2003). "Detecting Breakouts." Technical Analysis of Stocks & Commodities, 21(4).
        - Katsanos, M. (2003). "Detecting Breakouts In Intraday Charts." Technical Analysis of Stocks & Commodities, 21(9).

        Referenced against:
        - finta, FVE: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1998

        Also known as: FVE, Finite Volume Elements.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods to sum the signed and total volume over. Defaults to
                22, a month of trading days.
            cutoff (float, optional): The share of the close the money flow needs to exceed to count, as a
                decimal. Defaults to 0.003 (0.3%).
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
            pd.Series or pd.DataFrame: Finite Volume Elements values, as decimals between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the FVE for
          each asset in the Toolkit instance.
        - It agrees with finta's FVE divided by 100 (Katsanos and finta express it in percent, with the cutoff
          in percent as well), from the second value on: finta counts the first period, which has no
          change in the typical price, as neutral volume and starts one period earlier.
        - When growth is requested, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_finite_volume_elements().tail(5)
        ```

        Which returns:

        | Date       |    AAPL |   MSFT |   Benchmark |
        |:-----------|--------:|-------:|------------:|
        | 2025-12-24 |  0.156  | 0.3175 |      0.1378 |
        | 2025-12-26 |  0.0943 | 0.333  |      0.0929 |
        | 2025-12-29 |  0.0461 | 0.289  |      0.0072 |
        | 2025-12-30 |  0.0467 | 0.244  |     -0.0367 |
        | 2025-12-31 | -0.005  | 0.183  |     -0.1106 |
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

        finite_volume_elements = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_finite_volume_elements,
            window,
            cutoff,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=finite_volume_elements,
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
    def get_volume_zone_oscillator(
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
        Calculate the Volume Zone Oscillator (VZO): the volume of up closes minus that of down closes as a share of
        the total volume, smoothed, which shows whether buying or selling volume dominates.

        The Volume Zone Oscillator divides the volume into the volume of up closes and that of down closes
        and expresses their difference, smoothed with an exponential moving average, as a share of the
        smoothed total volume.

        The formula is as follows:

        - Signed Volume = sign(Close — Close(t-1)) * Volume
        - VZO = EMA(Signed Volume, window) / EMA(Volume, window)

        Khalil and Steckler read it in zones: above 0.05 the trend is bullish and above 0.40 overbought
        (0.60 extremely), below -0.05 bearish and below -0.40 oversold (-0.60 extremely), with crossings of
        zero and of the 0.40 lines as signals. As a share of volume it is comparable across tickers.

        For more information, see the following sources:

        - Khalil, W. and Steckler, D. (2011). "In The Volume Zone." Technical Analysis of Stocks & Commodities, 29(5).

        Referenced against:
        - finta, VZO: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1521

        Also known as: VZO, Volume Zone Oscillator.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods for both EMAs. Defaults to 14.
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
            pd.Series or pd.DataFrame: Volume Zone Oscillator values, as decimals between -1 and 1.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the VZO for
          each asset in the Toolkit instance.
        - The VZO is a decimal; Khalil and Steckler and finta multiply it by 100, so their zones are at 5,
          40 and 60. Both EMAs start from the simple average of their first values on the second period
          (the first has no change), so the VZO starts on period `window` + 1. finta averages from the
          first value and agrees after about 90 periods.
        - When growth is requested, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_volume_zone_oscillator().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.3125 | 0.3228 |      0.2742 |
        | 2025-12-26 | 0.2254 | 0.2524 |      0.1808 |
        | 2025-12-29 | 0.2857 | 0.1642 |      0.0476 |
        | 2025-12-30 | 0.1863 | 0.2429 |     -0.046  |
        | 2025-12-31 | 0.0688 | 0.1082 |     -0.1791 |
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

        volume_zone_oscillator = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_volume_zone_oscillator,
            window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=volume_zone_oscillator,
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
    def get_price_volume(
        self,
        period: str = "daily",
        close_column: str = "Close",
        signed: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Price-Volume (dollar volume): the close times the volume, the amount of money that traded in
        each period.

        The Price-Volume is the value traded in a period: the close times the volume. Unlike the number of
        shares, it can be compared across stocks with very different prices and is the usual measure of
        liquidity for position sizing. Signed, it carries the direction of the close, as a money flow per
        period.

        The formula is as follows:

        - Price-Volume = Close * Volume
        - Signed Price-Volume = sign(Close — Close(t-1)) * Close * Volume

        For more information, see the following sources:

        - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects." Journal of Financial Markets, 5(1), 31-56.

        Referenced against:
        - pandas-ta-classic, pvol: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/pvol.py#L17

        Also known as: PVOL, dollar volume, traded value.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Close".
            signed (bool, optional): Whether to sign the value with the direction of the close. Defaults to
                False.
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
            pd.Series or pd.DataFrame: Price-Volume values, in the currency of the prices.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Price-Volume for each asset in the Toolkit instance.
        - The traded close is the default, as the value traded is a market value; the dividend-adjusted
          close would understate the value of older periods.
        - It agrees with pandas-ta-classic's `pvol`, also in its signed form.
        - When growth is requested, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_price_volume().tail(5)
        ```

        Which returns:

        | Date       |        AAPL |        MSFT |   Benchmark |
        |:-----------|------------:|------------:|------------:|
        | 2025-12-24 | 4.9041e+09  | 2.8578e+09  | 2.72325e+10 |
        | 2025-12-26 | 5.88406e+09 | 4.31243e+09 | 2.87261e+10 |
        | 2025-12-29 | 6.49228e+09 | 5.30618e+09 | 4.30316e+10 |
        | 2025-12-30 | 6.04589e+09 | 6.79767e+09 | 3.23999e+10 |
        | 2025-12-31 | 7.42005e+09 | 7.54525e+09 | 5.05608e+10 |
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

        historical_data = self._historical_data[period]

        price_volume = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_price_volume,
            signed,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=price_volume,
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
    def get_price_volume_rank(
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
        Calculate the Price Volume Rank (PVR): a code from 1 (a rising price on rising volume) to 4 (a falling price
        on falling volume), which shows whether volume confirms a move.

        The Price Volume Rank classifies each period by the direction of the close and of the volume, from
        the most bullish to the most bearish combination: a rising price on rising volume (1), a rising
        price on falling volume (2), a falling price on rising volume (3) and a falling price on falling
        volume (4). An unchanged close or volume counts as rising.

        The formula is as follows:

        - PVR = 1 if Close >= Close(t-1) and Volume >= Volume(t-1)
        - PVR = 2 if Close >= Close(t-1) and Volume < Volume(t-1)
        - PVR = 3 if Close < Close(t-1) and Volume >= Volume(t-1)
        - PVR = 4 if Close < Close(t-1) and Volume < Volume(t-1)

        A moving average of the rank below 2.5 is read as bullish and above it as bearish.

        For more information, see the following sources:

        - Fulks Maxwell Laboratories (fmlabs), "Price Volume Rank," Technical Indicators Reference.

        Referenced against:
        - pandas-ta-classic, pvr: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/pvr.py#L12

        Also known as: PVR, Price Volume Rank.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
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
            pd.Series or pd.DataFrame: Price Volume Rank values (1, 2, 3 or 4).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the Price
          Volume Rank for each asset in the Toolkit instance.
        - It agrees with pandas-ta-classic's `pvr`.
        - When growth is requested, the method calculates the growth of the indicator values
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

        toolkit.technicals.get_price_volume_rank().tail(5)
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 |      2 |      2 |           2 |
        | 2025-12-26 |      3 |      3 |           3 |
        | 2025-12-29 |      1 |      3 |           3 |
        | 2025-12-30 |      4 |      1 |           4 |
        | 2025-12-31 |      3 |      3 |           3 |
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

        historical_data = self._historical_data[period]

        price_volume_rank = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_price_volume_rank,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=price_volume_rank,
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
    def get_buying_selling_pressure(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 40,
        smooth_window: int = 20,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Buying and Selling Pressure (BASP), which shows whether buyers or sellers control the market.

        It compares the distance of the close above the low (buying) and of the high above the close (selling) with
        their averages, weighted by volume.

        The Buying Pressure is the distance of the close above the low, the Selling Pressure the distance
        of the high above the close. Each is divided by its own exponential moving average, so it shows
        whether the pressure is larger than usual, and multiplied by the volume relative to its average, so
        a session with heavy volume weighs more. A short EMA smooths both lines.

        The formula is as follows:

        - Relative Volume = Volume / EMA(Volume, window)
        - Buying Pressure = EMA((Close — Low) / EMA(Close — Low, window) * Relative Volume, smooth_window)
        - Selling Pressure = EMA((High — Close) / EMA(High — Close, window) * Relative Volume, smooth_window)

        Both lines are ratios around 1 (the usual pressure on the usual volume). Buying pressure above
        selling pressure points to accumulation, and a crossing of the two to a change in who is in control.

        For more information, see the following sources:

        - Williams, L.R. (1985). "The Ultimate Oscillator." Technical Analysis of Stocks & Commodities, 3(4),
          for a similar buying pressure measured from the lower of the low and the previous close.

        Referenced against:
        - finta, BASP: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1671
        - finta, BASPN: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1691

        Also known as: BASP, BASPN, normalized buying and selling pressure.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way.
            window (int, optional): The number of periods for the averages of the pressure and the volume.
                Defaults to 40.
            smooth_window (int, optional): The number of periods for the EMA that smooths both lines.
                Defaults to 20; 1 gives the unsmoothed lines.
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
            pd.Series or pd.DataFrame: The Buying Pressure and Selling Pressure.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the buying
          and selling pressure for each asset in the Toolkit instance.
        - finta's BASPN is the default and its BASP a `smooth_window` of 1; finta cites no primary source.
          Every EMA starts from the simple average of its first values, while finta's average from the
          first value, so the two agree after about 300 periods.
        - When growth is requested, the method calculates the growth of the indicator values
          over the given lag.
        - finta defines the Buying and Selling Pressure without citing a primary source.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_buying_selling_pressure().xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Buying Pressure |   Selling Pressure |
        |:-----------|------------------:|-------------------:|
        | 2025-12-24 |            1.0156 |             0.8174 |
        | 2025-12-26 |            0.9294 |             0.7778 |
        | 2025-12-29 |            0.8723 |             0.7174 |
        | 2025-12-30 |            0.8069 |             0.6714 |
        | 2025-12-31 |            0.7332 |             0.659  |
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

        buying_selling_pressure = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column, "Volume"],
            breadth_model.get_buying_selling_pressure,
            window,
            smooth_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=buying_selling_pressure,
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
    def get_archer_on_balance_volume(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 4,
        long_window: int = 12,
        extreme_window: int = 2,
        run_window: int = 2,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Archer On-Balance Volume (AOBV): a fast and a slow average of the On-Balance Volume and
        whether the volume trend is up or down.

        The Archer On-Balance Volume adds a fast and a slow exponential moving average of On-Balance Volume
        and two flags for their trend, so OBV can be read like a moving average crossover system. A "long
        run" is a period in which the fast average rose over the last `run_window` periods while the slow
        one moved (a rising OBV trend or a potential bottom); a "short run" one in which the fast average
        fell (a falling trend or a potential top). The rolling minimum and maximum of OBV mark its recent
        range.

        The formula is as follows:

        - OBV = Cumulative Sum(sign(Close — Close(t-1)) * Volume), starting at 0 on the start date
        - Fast Average = EMA(OBV, short_window), Slow Average = EMA(OBV, long_window)
        - Minimum = Min(OBV, extreme_window), Maximum = Max(OBV, extreme_window)
        - Long Run = 1 if Fast Average rose over `run_window` periods and the Slow Average changed, else 0
        - Short Run = 1 if Fast Average fell over `run_window` periods and the Slow Average changed, else 0

        For more information, see the following sources:

        - Granville, J.E. (1963). "Granville's New Key to Stock Market Profits." Prentice-Hall.

        Referenced against:
        - pandas-ta-classic, aobv: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/aobv.py#L17

        Also known as: AOBV, Archer OBV.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): The number of periods for the fast EMA of OBV. Defaults to 4.
            long_window (int, optional): The number of periods for the slow EMA of OBV. Defaults to 12.
            extreme_window (int, optional): The number of periods for the rolling minimum and maximum of OBV.
                Defaults to 2.
            run_window (int, optional): The number of periods over which the averages must rise or fall.
                Defaults to 2.
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
            pd.Series or pd.DataFrame: The On-Balance Volume, its Minimum and Maximum, the Fast and Slow
            Average and the Long Run and Short Run flags (1 or 0).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the Archer
          On-Balance Volume for each asset in the Toolkit instance, as a running total from the start date.
        - The indicator comes from a TradingView script by Archer without a published paper. It equals
          pandas-ta-classic's `aobv`, except that its On-Balance Volume (as TA-Lib's) starts at the first
          period's volume instead of 0: the OBV, its minimum, maximum and averages are that volume higher
          there, and the flags are equal.
        - When growth is requested, the method calculates the growth of the indicator values
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

        archer_obv = toolkit.technicals.get_archer_on_balance_volume(
            show_columns=["On-Balance Volume", "Fast Average", "Slow Average", "Long Run", "Short Run"]
        )

        archer_obv.xs("AAPL", level=1, axis="columns").tail(5)
        ```

        Which returns:

        | Date       |   Fast Average |   Long Run |   On-Balance Volume |   Short Run |   Slow Average |
        |:-----------|---------------:|-----------:|--------------------:|------------:|---------------:|
        | 2025-12-24 |    2.43811e+09 |          1 |         2.47241e+09 |           0 |    2.38388e+09 |
        | 2025-12-26 |    2.44322e+09 |          1 |         2.45088e+09 |           0 |    2.39418e+09 |
        | 2025-12-29 |    2.45577e+09 |          1 |         2.4746e+09  |           0 |    2.40656e+09 |
        | 2025-12-30 |    2.45445e+09 |          1 |         2.45246e+09 |           0 |    2.41362e+09 |
        | 2025-12-31 |    2.44273e+09 |          0 |         2.42517e+09 |           1 |    2.41539e+09 |
        """
        self._check_arguments(
            period,
            close_column,
            short_window=short_window,
            long_window=long_window,
            extreme_window=extreme_window,
            run_window=run_window,
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

        # A running total starts at the start date, not at the earlier prices.
        historical_data = self._historical_data[period].loc[self._start_date :]

        archer_on_balance_volume = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_archer_on_balance_volume,
            short_window,
            long_window,
            extreme_window,
            run_window,
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=archer_on_balance_volume,
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
    def get_volume_profile(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        bins: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Volume Profile (Volume by Price) between the start and end date: how much volume traded at
        each price level, which shows the prices the market accepted most.

        The Volume Profile is not a series over time but a histogram: it divides the range of the closes
        between the start and end date into price bins of equal width and adds up the volume traded at a
        close in each bin, split into the volume of up closes and that of down closes.

        The formula is as follows:

        - Bin edges = Min(Close) + k * (Max(Close) — Min(Close)) / bins, for k = 0 to bins
        - Up Volume = Sum(Volume of the periods with Close > Close(t-1) and the Close in the bin)
        - Down Volume = Sum(Volume of the periods with Close < Close(t-1) and the Close in the bin)
        - Total Volume = Sum(Volume of all periods with the Close in the bin)

        Bins with a lot of volume are prices at which many positions were opened, which tend to act as
        support or resistance when price returns to them; the bin with the most volume is the point of
        control. More up than down volume in a bin suggests the positions opened there were mostly
        purchases.

        For more information, see the following sources:

        - Steidlmayer, J.P. and Koy, K. (1986). "Markets and Market Logic." Porcupine Press.
        - StockCharts ChartSchool, "Volume by Price" (https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/volume-by-price).

        Referenced against:
        - pandas-ta-classic, vp: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/vp.py#L13

        Also known as: VP, Volume by Price, Volume at Price, Market Profile.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            bins (int, optional): The number of price bins of equal width. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A row per price bin (1 for the lowest prices) with the Low Price, High Price,
            Up Volume, Down Volume and Total Volume of each ticker.

        Notes:
        - The profile covers the periods between the start and end date of the Toolkit and looks at every
          close in them, so it describes that window as a whole and is not meant for a backtest. Each
          ticker has its own bins, over the range of its own closes.
        - The bins are those of pandas-ta-classic's `vp` sorted by close, whose up and down volume
          it equals, with the exact range of the closes as edges and every bin kept, also those without
          volume (pandas-ta-classic widens the range by 0.1% and drops empty bins; its default
          splits the period into stretches of time instead). Its total is the up plus
          the down volume; here the Total Volume also includes the volume of unchanged closes and of the
          first period, which has no change.
        - A close on an inner edge belongs to the lower bin. A ticker whose close does not change has all
          its volume in the first bin, with every edge at that close.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            tickers=["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_volume_profile().xs("AAPL", level=1, axis="columns")
        ```

        Which returns:

        |   Price Bin |   Down Volume |   High Price |   Low Price |   Total Volume |   Up Volume |
        |------------:|--------------:|-------------:|------------:|---------------:|------------:|
        |           1 |   2.81326e+08 |      182.632 |     171.236 |    2.81326e+08 | 0           |
        |           2 |   3.54266e+08 |      194.028 |     182.632 |    3.54266e+08 | 0           |
        |           3 |   1.25692e+09 |      205.424 |     194.028 |    2.79533e+09 | 1.53841e+09 |
        |           4 |   9.40298e+08 |      216.82  |     205.424 |    2.32431e+09 | 1.38402e+09 |
        |           5 |   8.29389e+08 |      228.216 |     216.82  |    1.59005e+09 | 7.60656e+08 |
        |           6 |   1.1089e+09  |      239.612 |     228.216 |    2.11226e+09 | 1.00335e+09 |
        |           7 |   3.32909e+08 |      251.008 |     239.612 |    1.00977e+09 | 6.21122e+08 |
        |           8 |   3.4874e+08  |      262.404 |     251.008 |    9.81495e+08 | 6.32755e+08 |
        |           9 |   7.25819e+08 |      273.8   |     262.404 |    1.47212e+09 | 7.46301e+08 |
        |          10 |   2.38447e+08 |      285.196 |     273.8   |    6.23551e+08 | 3.85104e+08 |
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

        # The profile describes the closes between the start and end date.
        historical_data = self._historical_data[period].loc[
            self._start_date : self._end_date
        ]

        volume_profile = calculate_per_ticker(
            historical_data,
            [close_column, "Volume"],
            breadth_model.get_volume_profile,
            bins,
        )

        return apply_rounding(
            volume_profile, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_market_breadth(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 50,
        long_window: int = 200,
        high_low_window: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the breadth of the market the Toolkit's tickers make up: how many of them take part in a move.

        An index can rise on the strength of a few large companies while most of its members fall; breadth shows
        that divergence, which often comes before the index itself turns. Breadth is most meaningful across many
        tickers, such as the constituents of an index, which `Discovery.get_index_constituents` lists (e.g. the 500
        tickers of the S&P 500). Today's members leave out the companies that were removed from the index since
        (often after falling behind), so breadth of past dates computed from them is biased towards the survivors,
        the further back the more; its `date=` gives the members on the first date of the window instead.

        Per date it returns:

        - Advancers, Decliners and Unchanged: the number of tickers whose close rose, fell or
          stayed the same, and the Net Advances.
        - Advance-Decline Line: the running total of the net advances. A rising index with a
          falling line is the classic negative divergence.
        - Advance-Decline Ratio: the advancers divided by the decliners.
        - McClellan Oscillator: the ratio-adjusted version, the 19-period minus the 39-period
          exponential moving average (smoothing constants of 0.10 and 0.05) of the net advances
          divided by advancers plus decliners (times 1,000); unchanged issues are left out.
          Each average starts from the simple average of its first 19 or 39 values, so the
          oscillator starts once the 39-period average exists (NaN before), as does its running
          total, the McClellan Summation Index.
        - Percentage Above the short and long moving average: the share of tickers trading
          above their 50 and 200-period average, a gauge of how broad a trend is.
        - New Highs and New Lows: the tickers whose close is above the highest (below the
          lowest) close of the previous `high_low_window` - 1 periods, so the highest or
          lowest of the last `high_low_window` periods; a close that only equals the previous
          high is no new high. NaN until at least one ticker has a full window.

        The Benchmark (and a Portfolio) are left out. A constituent counts on the dates it
        traded, and its changes, moving averages and highs and lows run over its own closes: on
        a date only a cryptocurrency in the Toolkit traded, such as a weekend, only the
        cryptocurrency counts, and the McClellan averages then skip that date (see
        `breadth_model.get_market_breadth`).

        For more information, see the following book:

        - McClellan, S. and McClellan, M. (1970). "Patterns for Profit: The McClellan
          Oscillator and Summation Index." Trade Levels Inc.

        Also known as: market breadth, advance-decline line, McClellan oscillator, percentage
        above 200-day moving average, new highs new lows.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The name of the column containing the close prices.
                Defaults to "Adj Close".
            short_window (int, optional): The short moving average window. Defaults to 50.
            long_window (int, optional): The long moving average window. Defaults to 200.
            high_low_window (int | None, optional): The number of periods, including the current
                one, of the new highs and lows. Defaults to None, which uses a year of periods:
                252 daily, 52 weekly, 12 monthly, 4 quarterly and 1 yearly (too short for a new
                high or low, so those stay NaN), and 252 for intraday data.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The breadth measures per date.

        As an example:

        ```python
        from financetoolkit import Discovery, Toolkit

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        constituents = discovery.get_index_constituents("Dow Jones")

        toolkit = Toolkit(
            constituents.index.tolist(),
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.technicals.get_market_breadth().tail()
        ```

        Which returns:

        | Date       |   Advancers |   Decliners |   Unchanged |   Net Advances |   Advance-Decline Line |   Advance-Decline Ratio |   McClellan Oscillator |   McClellan Summation Index |   Percentage Above 50-Period Average |   Percentage Above 200-Period Average |   New Highs |   New Lows |
        |:-----------|------------:|------------:|------------:|---------------:|-----------------------:|------------------------:|-----------------------:|----------------------------:|-------------------------------------:|--------------------------------------:|------------:|-----------:|
        | 2025-12-24 |          26 |           3 |           1 |             23 |                    978 |                  8.6667 |                65.5048 |                    -814.088 |                               0.7333 |                                   0.8 |           3 |          0 |
        | 2025-12-26 |          14 |          16 |           0 |             -2 |                    976 |                  0.875  |                49.8424 |                    -764.246 |                               0.7    |                                   0.8 |           1 |          0 |
        | 2025-12-29 |          10 |          20 |           0 |            -10 |                    966 |                  0.5    |                22.8684 |                    -741.377 |                               0.7333 |                                   0.8 |           0 |          0 |
        | 2025-12-30 |           9 |          21 |           0 |            -12 |                    954 |                  0.4286 |                -3.642  |                    -745.019 |                               0.7    |                                   0.8 |           0 |          0 |
        | 2025-12-31 |           3 |          27 |           0 |            -24 |                    930 |                  0.1111 |               -46.2902 |                    -791.309 |                               0.5667 |                                   0.8 |           0 |          0 |
        """
        self._check_arguments(
            period,
            close_column,
            short_window=short_window,
            long_window=long_window,
            high_low_window=high_low_window,
        )
        # The Advance-Decline Line and McClellan Summation Index start at the start date.
        # NaN on the dates a constituent did not trade, so a stock is not counted as unchanged on the weekends of a cryptocurrency.  # noqa: E501
        prices = self._close_prices(period, close_column, traded_only=True).loc[
            self._start_date :
        ]
        constituents = [
            ticker
            for ticker in prices.columns
            if ticker not in ("Portfolio", "Benchmark")
        ]
        # A year of periods, so weekly data looks for 52-week highs rather than 252-week ones. Intraday bars have no fixed number per year and keep the daily count.  # noqa: E501
        if high_low_window is None:
            high_low_window = PERIODS_PER_YEAR.get(period, PERIODS_PER_YEAR["daily"])
        breadth = breadth_model.get_market_breadth(
            prices[constituents], short_window, long_window, high_low_window
        ).loc[self._start_date : self._end_date]

        return apply_rounding(
            breadth, rounding if rounding is not None else self._rounding
        )
