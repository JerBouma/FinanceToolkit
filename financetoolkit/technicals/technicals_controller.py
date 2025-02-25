"""Technicals Module"""

__docformat__ = "google"


import pandas as pd

from financetoolkit.helpers import calculate_growth, handle_portfolio
from financetoolkit.technicals import (
    breadth_model,
    momentum_model,
    overlap_model,
    volatility_model,
)
from financetoolkit.technicals.helpers import handle_errors

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-public-methods,too-many-locals,eval-used
# pylint: disable=too-many-boolean-expressions


class Technicals:
    """
    The Technicals Module contains 30+ Technical Indicators that can
    be used to analyse companies. These ratios are divided into 4
    categories which are breadth, momentum, overlap and volatility.
    Each indicator is calculated using the data from the Toolkit module.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: pd.DataFrame,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Technicals Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the calculation.
            intraday_historical (pd.DataFrame, optional): The intraday historical data to use for the calculation.
                Defaults to pd.DataFrame().
            daily_historical (pd.DataFrame, optional): The daily historical data to use for the calculation.
                Defaults to pd.DataFrame().
            weekly_historical (pd.DataFrame, optional): The weekly historical data to use for the calculation.
                Defaults to pd.DataFrame().
            monthly_historical (pd.DataFrame, optional): The monthly historical data to use for the calculation.
                Defaults to pd.DataFrame().
            quarterly_historical (pd.DataFrame, optional): The quarterly historical data to use for the calculation.
                Defaults to pd.DataFrame().
            yearly_historical (pd.DataFrame, optional): The yearly historical data to use for the calculation.
                Defaults to pd.DataFrame().
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculation.
                Defaults to None.
            end_date (str | None, optional): The end date to use for the calculation.
                Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        average_directional_index = toolkit.technicals.get_average_directional_index()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |
        |:-----------|--------:|--------:|
        | 2023-08-21 | 62.8842 | 36.7468 |
        | 2023-08-22 | 65.7063 | 36.5525 |
        | 2023-08-23 | 67.3596 | 35.5149 |
        | 2023-08-24 | 66.4527 | 35.4399 |
        | 2023-08-25 | 63.4837 | 32.3323 |
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

    def collect_all_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
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

        Returns:
            pd.Series or pd.DataFrame: Technical indicators calculated based on the specified parameters.

        Notes:
        - The method calculates various types of technical indicators for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.collect_all_indicators()
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

        self._all_indicators = self._all_indicators.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            self._all_indicators_growth = calculate_growth(
                self._all_indicators,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return self._all_indicators_growth if growth else self._all_indicators

    def collect_breadth_indicators(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.collect_breadth_indicators()
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

        self._breadth_indicators = pd.concat(breadth_indicators, axis=1)

        self._breadth_indicators = self._breadth_indicators.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            self._breadth_indicators_growth = calculate_growth(
                self._breadth_indicators,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the McClellan Oscillator for a given price series.

        The McClellan Oscillator is a breadth indicator that measures the difference
        between the exponential moving average of advancing stocks and the exponential
        moving average of declining stocks.

        The formula is a follows:

        - McClellan Oscillator = EMA(Advancers) — EMA(Decliners)

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_mcclellan_oscillator()
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

        mcclellan_oscillator = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            mcclellan_oscillator[ticker] = breadth_model.get_mcclellan_oscillator(
                historical_data[close_column][ticker], short_ema_window, long_ema_window
            ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                mcclellan_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return mcclellan_oscillator.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_advancers_decliners(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Advancers/Decliners ratio for a given price series.

        The Advancers/Decliners ratio is a breadth indicator that measures the number
        of advancing stocks (stocks with positive price changes) versus the number of
        declining stocks (stocks with negative price changes).

        The formula is a follows:

        - Advancers/Decliners = Advancers / Decliners

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_advancers_decliners()
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

        advancers_decliners = breadth_model.get_advancers_decliners(
            historical_data[close_column],
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                advancers_decliners,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return advancers_decliners.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_on_balance_volume(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the On-Balance Volume (OBV) for a given price series.

        The On-Balance Volume (OBV) is a technical indicator that uses volume flow to predict changes in stock price.
        It accumulates the volume on up days and subtracts the volume on down days. The resulting OBV line provides
        insights into the buying and selling pressure behind price movements.

        The formula is a follows:

        - OBV = Previous OBV + Current Volume if Close > Previous Close

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_on_balance_volume()
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

        on_balance_volume = breadth_model.get_on_balance_volume(
            historical_data[close_column],
            historical_data["Volume"],
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                on_balance_volume,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return on_balance_volume.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_accumulation_distribution_line(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Accumulation/Distribution Line for a given price series.

        The Accumulation/Distribution Line is a technical indicator that evaluates the flow of money
        into or out of an asset. It takes into account both price and volume information to identify
        whether an asset is being accumulated (bought) or distributed (sold) by investors.

        The formula is a follows:

        - ADL = Previous ADL + Current ADL

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_accumulation_distribution_line()
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

        if growth:
            return calculate_growth(
                accumulation_distribution_line,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return accumulation_distribution_line.round(
            rounding if rounding else self._rounding
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chaikin Oscillator for a given price series.

        The Chaikin Oscillator is a momentum-based indicator that combines price and volume
        to help identify potential trends and reversals in the market. It is calculated as the
        difference between the 3-day and 10-day Accumulation/Distribution Line.

        The formula is a follows:

        - Chaikin Oscillator = EMA(short-window ADL) — EMA(long-window ADL)

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_chaikin_oscillator()
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

        chaikin_oscillator = breadth_model.get_chaikin_oscillator(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            historical_data["Volume"],
            short_window,
            long_window,
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                chaikin_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return chaikin_oscillator.round(rounding if rounding else self._rounding)

    def collect_momentum_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.collect_momentum_indicators()
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

        self._momentum_indicators = pd.concat(momentum_indicators, axis=1)

        self._momentum_indicators = self._momentum_indicators.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            self._momentum_indicators_growth = calculate_growth(
                self._momentum_indicators,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Money Flow Index (MFI) for a given price series.

        The Money Flow Index is a momentum indicator that measures the strength and
        direction of money flowing in and out of a security by considering both price
        and volume.

        The formula is a follows:

        - MFI = 100 - (100 / (1 + (positive_money_flow / negative_money_flow)))

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_money_flow_index()
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

        money_flow_index = momentum_model.get_money_flow_index(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            historical_data["Volume"],
            window,
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                money_flow_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return money_flow_index.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Williams Percent R (Williams %R) for a given price series.

        The Williams %R is a momentum indicator that measures the level of the close price
        relative to the high-low range over a certain number of periods.

        The formula is a follows:

        - Williams %R = (Highest High — Close) / (Highest High — Lowest Low) * —100

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_williams_percent_r()
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

        williams_percent_r = momentum_model.get_williams_percent_r(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            window,
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                williams_percent_r,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return williams_percent_r.round(rounding if rounding else self._rounding)

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
    ) -> tuple[pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Aroon Indicator for a given price series.

        The Aroon Indicator is an oscillator that measures the strength of a trend and the
        likelihood of its continuation or reversal.

        The formula is a follows:

        - Aroon Up = ((Number of periods) — (Number of periods since highest high)) / (Number of periods) * 100

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            window (int, optional): The number of periods for calculating the Aroon Indicator.
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.

        Returns:
            Tuple[pd.Series, pd.Series] or Tuple[pd.DataFrame, pd.DataFrame]:
            Aroon Indicator values for the upward and downward trends.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Aroon Indicator values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_aroon_indicator()
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

        if growth:
            aroon_indicator_growth = calculate_growth(
                aroon_indicator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return aroon_indicator_growth

        return aroon_indicator.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Commodity Channel Index (CCI) for a given price series.

        The Commodity Channel Index is an oscillator that measures the current price level
        relative to an average price level over a specified period.

        The formula is a follows:

        - CCI = (Typical Price — SMA(Typical Price)) / (constant * Mean Deviation)

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_commodity_channel_index()
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

        if growth:
            return calculate_growth(
                commodity_channel_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return commodity_channel_index.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Vigor Index (RVI) for a given price series.

        The Relative Vigor Index is an oscillator that measures the conviction of a current price
        trend using the relationship between closing and opening prices.

        The formula is a follows:

        - RVI = SMA(Upward Change) / (SMA(Upward Change) + SMA(Downward Change))

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_relative_vigor_index()
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

        relative_vigor_index = momentum_model.get_relative_vigor_index(
            historical_data["Open"],
            historical_data[close_column],
            historical_data["Volume"],
            window,
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                relative_vigor_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return relative_vigor_index.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Force Index for a given price series.

        The Force Index is an indicator that measures the strength behind price movements.

        The formula is a follows:

        - Force Index = SMA(Periods) * (Close — Close(1))

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_force_index()
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

        force_index = momentum_model.get_force_index(
            historical_data[close_column],
            historical_data["Volume"],
            window,
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                force_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return force_index.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Ultimate Oscillator for a given price series.

        The Ultimate Oscillator is a momentum oscillator that combines short-term, mid-term,
        and long-term price momentum into a single value.

        The formula is a follows:

        - Ultimate Oscillator = 100 * ((4 * SMA(Periods)) / (SMA(Periods) + SMA(Periods) + SMA(Periods)))

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_ultimate_oscillator()
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

        if growth:
            ultimate_oscillator_growth = calculate_growth(
                ultimate_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            return ultimate_oscillator_growth

        return ultimate_oscillator.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Percentage Price Oscillator (PPO) for a given price series.

        The Percentage Price Oscillator (PPO) is a momentum oscillator that measures the
        difference between two moving averages as a percentage of the longer moving average.

        The formula is a follows:

        - PPO = ((Long-term EMA — Short-term EMA) / Short—term EMA) * 100

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_percentage_price_oscillator()
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

        percentage_price_oscillator = momentum_model.get_percentage_price_oscillator(
            historical_data[close_column],
            short_window,
            long_window,
        ).loc[self._start_date : self._end_date]

        if growth:
            ppo_growth = calculate_growth(
                percentage_price_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            return ppo_growth

        return percentage_price_oscillator.round(
            rounding if rounding else self._rounding
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Detrended Price Oscillator (DPO) for a given price series.

        The Detrended Price Oscillator (DPO) is an indicator that helps identify short-term cycles
        by removing longer-term trends from prices.

        The formula is a follows:

        - DPO = Close - SMA(Close, (Number of Periods / 2) + 1)

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_detrended_price_oscillator()
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

        detrended_price_oscillator = momentum_model.get_detrended_price_oscillator(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        if growth:
            dpo_growth = calculate_growth(
                detrended_price_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            return dpo_growth

        return detrended_price_oscillator.round(
            rounding if rounding else self._rounding
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX) for a given price series.

        The Average Directional Index (ADX) is an indicator that measures the strength of a trend,
        whether it's an uptrend or a downtrend.

        The formula is a follows:

        - ADX = SMA(DMI) / (SMA(DMI) + SMA(DMI))

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_average_directional_index()
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

        if growth:
            adx_growth = calculate_growth(
                average_directional_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            return adx_growth

        return average_directional_index.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chande Momentum Oscillator (CMO) for a given price series.

        The Chande Momentum Oscillator is an indicator that measures the momentum of a price
        series and identifies overbought and oversold conditions.

        The formula is a follows:

        - CMO = ((Sum of Upward Change) — (Sum of Downward Change)) / ((Sum of Upward Change)
            + (Sum of Downward Change))

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_chande_momentum_oscillator()
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

        chande_momentum_oscillator = momentum_model.get_chande_momentum_oscillator(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        if growth:
            cmo_growth = calculate_growth(
                chande_momentum_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            return cmo_growth

        return chande_momentum_oscillator.round(
            rounding if rounding else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_ichimoku_cloud(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        conversion_window: int = 9,
        base_window: int = 20,
        lead_span_b_window: int = 40,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Ichimoku Cloud indicator for a given price series.

        The Ichimoku Cloud, also known as the Ichimoku Kinko Hyo, is a versatile indicator that
        defines support and resistance, identifies trend direction, gauges momentum, and provides
        trading signals.

        The formula is a follows:

        - Conversion Line = (Highest High + Lowest Low) / 2

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            conversion_window (int, optional): The number of periods to consider for the
                Conversion Line (Tenkan-sen) calculation. Defaults to 9.
            base_window (int, optional): The number of periods to consider for the Base Line
                (Kijun-sen) calculation. Defaults to 20.
            lead_span_b_window (int, optional): The number of periods to shift forward for the
                Lead Span B calculation. Defaults to 40.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the indicator values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series] or
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Conversion Line, Base Line, Lead Span A, and Lead Span B values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ichimoku Cloud values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_ichimoku_cloud()
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

        if growth:
            ichimoku_cloud_growth = calculate_growth(
                ichimoku_cloud,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return ichimoku_cloud_growth

        return ichimoku_cloud.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_stochastic_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        smooth_widow: int = 3,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Stochastic Oscillator indicator for a given price series.

        The Stochastic Oscillator is a momentum indicator that shows the location of the close
        relative to the high-low range over a set number of periods. It consists of the %K line
        (fast) and the %D line (slow).

        The formula is a follows:

        - %K = 100 * ((Close — Lowest Low) / (Highest High — Lowest Low))
        - %D = SMA(%K)

        Args:
            period (str, optional): The time period to consider for historical data.
                Can be "daily", "weekly", "quarterly", or "yearly". Defaults to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            window (int, optional): The number of periods to consider for the %K line calculation.
                Defaults to 14.
            smooth_widow (int, optional): The number of periods to consider for the %D line
                (slow stochastic) calculation. Defaults to 3.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the %K and %D values.
                Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.

        Returns:
            Tuple[pd.Series, pd.Series] or Tuple[pd.DataFrame, pd.DataFrame]:
            %K line and %D line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the Stochastic Oscillator values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the %K and %D values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_stochastic_oscillator()
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

        stochastic_oscillator_dict = {}

        for ticker in historical_data[close_column].columns:
            stochastic_oscillator_dict[ticker] = (
                momentum_model.get_stochastic_oscillator(
                    historical_data["High"][ticker],
                    historical_data["Low"][ticker],
                    historical_data[close_column][ticker],
                    window,
                    smooth_widow,
                ).loc[self._start_date : self._end_date]
            )

        stochastic_oscillator = (
            pd.concat(stochastic_oscillator_dict, axis=1)
            .swaplevel(1, 0, axis=1)
            .sort_index(axis=1)
        )

        if growth:
            stochastic_oscillator_growth = calculate_growth(
                stochastic_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return stochastic_oscillator_growth

        return stochastic_oscillator.round(rounding if rounding else self._rounding)

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
    ) -> tuple[pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Moving Average Convergence Divergence (MACD) indicator for a given price series.

        The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator
        that shows the relationship between two moving averages of a security's price. It consists
        of the MACD line, signal line, and MACD histogram.

        The formula is a follows:

        - MACD Line = Short-term EMA — Long-term EMA
        - Signal Line = SMA(MACD Line)

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
                Defaults to 1.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] or Tuple[pd.Series, pd.Series]:
            MACD line and signal line values.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates
          the MACD and signal line values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the MACD and signal values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_moving_average_convergence_divergence()
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

        if growth:
            macd_growth = calculate_growth(
                macd,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return macd_growth

        return macd.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI) indicator for a given price series.

        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and
        change of price movements. It ranges from 0 to 100 and is used to identify overbought or
        oversold conditions in an asset's price.

        The formula is a follows:

        - RSI = 100 — (100 / (1 + RS))

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_relative_strength_index()
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

        relative_strength_index = momentum_model.get_relative_strength_index(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                relative_strength_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return relative_strength_index.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_balance_of_power(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Balance of Power (BOP) indicator for a given price series.

        The Balance of Power (BOP) indicator measures the strength of buyers versus sellers
        in the market. It relates the price change to the change in the asset's trading range.

        The formula is a follows:

        - BOP = (Close — Open) / (High — Low)

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_balance_of_power()
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

        balance_of_power = momentum_model.get_balance_of_power(
            historical_data["Open"],
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                balance_of_power,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return balance_of_power.round(rounding if rounding else self._rounding)

    def collect_overlap_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.collect_overlap_indicators()
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

        self._overlap_indicators = pd.concat(overlap_indicators, axis=1)

        self._overlap_indicators = self._overlap_indicators.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            self._overlap_indicators_growth = calculate_growth(
                self._overlap_indicators,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        if len(self._tickers) == 1:
            return (
                self._overlap_indicators_growth[self._tickers[0]]
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Moving Average (MA) for a given price series.

        The Moving Average (MA) is a commonly used technical indicator that smooths out
        price data by calculating the average price over a specified number of periods.

        The formula is a follows:

        - MA = (Sum of Prices) / (Number of Prices)

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_moving_average()
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

        moving_average = overlap_model.get_moving_average(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                moving_average,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return moving_average.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Exponential Moving Average (EMA) for a given price series.

        EMA is a technical indicator that gives more weight to recent price data,
        providing a smoothed moving average that reacts faster to price changes.

        The formula is a follows:

        - EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_exponential_moving_average()
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

        exponential_moving_average = overlap_model.get_exponential_moving_average(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                exponential_moving_average,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return exponential_moving_average.round(
            rounding if rounding else self._rounding
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Double Exponential Moving Average (DEMA) for a given price series.

        DEMA is a technical indicator that attempts to reduce the lag from traditional
        moving averages by using a combination of two exponential moving averages.

        The formula is a follows:

        - EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_double_exponential_moving_average()
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

        double_exponential_moving_average = (
            overlap_model.get_double_exponential_moving_average(
                historical_data[close_column], window
            ).loc[self._start_date : self._end_date]
        )

        if growth:
            return calculate_growth(
                double_exponential_moving_average,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return double_exponential_moving_average.round(
            rounding if rounding else self._rounding
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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_trix()
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

        trix = overlap_model.get_trix(historical_data[close_column], window).loc[
            self._start_date : self._end_date
        ]

        if growth:
            return calculate_growth(
                trix,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return trix.round(rounding if rounding else self._rounding)

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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                Defaults to 1.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] or Tuple[pd.Series, pd.Series, pd.Series]:
            Bollinger Bands (upper, middle, lower).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          Bollinger Bands for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Bollinger Bands
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_bollinger_bands()
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

        if growth:
            bollinger_bands_growth = calculate_growth(
                bollinger_bands,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return bollinger_bands_growth

        return bollinger_bands.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Triangular Moving Average (TMA) for a given price series.

        The Triangular Moving Average (TMA) is a smoothed version of the Simple Moving Average (SMA)
        that uses multiple SMAs to reduce noise and provide a smoother trendline.

        The formula is a follows:

        - TMA = SMA(SMA(Close, Window), Window)

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_triangular_moving_average()
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

        triangular_moving_average = overlap_model.get_triangular_moving_average(
            historical_data[close_column], window
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                triangular_moving_average,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return triangular_moving_average.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_support_resistance_levels(
        self,
        sensitivity: float = 0.05,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = None,
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

        Returns:
           pd.DataFrame: The support and resistance levels for each asset.

        Raises:
            ValueError: If the specified `period` is not one of the valid options.

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates the
          support and resistance levels for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        support_resistance_levels = toolkit.technicals.get_support_resistance_levels()
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

        return support_resistance_levels_df.round(
            rounding if rounding else self._rounding
        )

    def collect_volatility_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.collect_volatility_indicators()
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

        keltner_channels = self.get_keltner_channels(
            period=period, close_column=close_column, window=window
        )

        volatility_indicators["Keltner Channel Upper"] = keltner_channels["Upper Line"]
        volatility_indicators["Keltner Channel Middle"] = keltner_channels[
            "Middle Line"
        ]
        volatility_indicators["Keltner Channel Lower"] = keltner_channels["Lower Line"]

        self._volatility_indicators = pd.concat(volatility_indicators, axis=1)

        self._volatility_indicators = self._volatility_indicators.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            self._volatility_indicators_growth = calculate_growth(
                self._volatility_indicators,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        if len(self._tickers) == 1:
            return (
                self._volatility_indicators_growth[self._tickers[0]]
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
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the True Range (TR) for a given price series.

        The True Range (TR) is a measure of market volatility that considers the differences
        between the high and low prices and the previous closing price. It provides insights
        into the price movement of an asset.

        The formula is a follows:

        - TR = max(high — low, abs(high — previous_close), abs(low — previous_close))

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

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_true_range()
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

        true_range = pd.DataFrame(
            index=historical_data.loc[self._start_date : self._end_date].index
        )
        for ticker in historical_data[close_column].columns:
            true_range[ticker] = volatility_model.get_true_range(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
            ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                true_range,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return true_range.round(rounding if rounding else self._rounding)

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
    ) -> pd.Series:
        """
        Calculate the Average True Range (ATR) of a given price series.

        The Average True Range (ATR) is a technical indicator that measures the volatility
        of an asset's price movements over a specified number of periods. It provides insights
        into the potential price range of an asset, which can help traders and investors make
        more informed decisions.

        The formula is a follows:

        - TR = max(high — low, abs(high — previous_close), abs(low — previous_close))
        - ATR = EMA(TR, Window)

        Args:
            period (str): Period for which to calculate the ATR.
            window (int): Number of periods for ATR calculation.
                The number of periods (time intervals) over which to calculate the Average True Range.
            rounding (int | None): Number of decimal places to round the resulting ATR values to.
                If None, no rounding is performed.
            growth (bool): Flag indicating whether to return the ATR growth rate.
                If True, the ATR growth rate is calculated.
            lag (int | list[int]): Number of periods to lag the ATR values by.
                If an integer is provided, all ATR values are lagged by the same number of periods.
                If a list of integers is provided, each ATR value is lagged by the corresponding number of periods.

        Returns:
            pd.Series: ATR values or ATR growth rate (if growth is True).
                A pandas Series containing the calculated Average True Range values or growth rate for each period.

        Formula:
        The Average True Range (ATR) is calculated using the following steps:
        1. Calculate the True Range (TR) for each period:
            - True Range (TR) = max(high - low, abs(high - previous_close), abs(low - previous_close))
        2. Calculate the Average True Range (ATR) over the specified window:
            - ATR = EMA(TR, window), where EMA is the Exponential Moving Average.

        Notes:
        - ATR values are typically used to assess the volatility and potential price movement of an asset.
        - A higher ATR value indicates higher volatility, while a lower ATR value suggests lower volatility.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_average_true_range()
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

        if growth:
            return calculate_growth(
                average_true_range,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return average_true_range.round(rounding if rounding else self._rounding)

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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Keltner Channels for a given price series.

        The Keltner Channels consist of three lines:
        - Upper Channel Line = Exponential Moving Average (EMA) of High Prices + ATR * ATR Multiplier
        - Middle Channel Line = Exponential Moving Average (EMA) of Closing Prices
        - Lower Channel Line = Exponential Moving Average (EMA) of Low Prices - ATR * ATR Multiplier

        The formula is a follows:

        - EMA = (Close — Previous EMA) * (2 / (1 + Window)) + Previous EMA
        - ATR = EMA(TR, ATR Window)
        - Upper Channel Line = EMA(High, Window) + ATR * ATR Multiplier
        - Middle Channel Line = EMA(Close, Window)
        - Lower Channel Line = EMA(Low, Window) — ATR * ATR Multiplier

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
                Defaults to 1.

        Returns:
            pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Keltner Channels (upper, middle, lower).

        Notes:
        - The method retrieves historical data based on the specified `period` and calculates Keltner Channels
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the channels using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(tickers=["AAPL", "MSFT"])

        toolkit.technicals.get_keltner_channels()
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

        if growth:
            kelter_channels_growth = calculate_growth(
                kelter_channels,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return kelter_channels_growth

        return kelter_channels.round(rounding if rounding else self._rounding)
