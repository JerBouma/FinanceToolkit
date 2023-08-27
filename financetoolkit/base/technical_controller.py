"""Ratios Module"""
__docformat__ = "google"


import pandas as pd

from financetoolkit.base.helpers import calculate_growth, handle_errors
from financetoolkit.technical import breadth, momentum, overlap, volatility, volume

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-public-methods,too-many-locals,eval-used


class Technical:
    """
    Technical Controller Class
    """

    def __init__(
        self,
        tickers: str | list[str],
        daily_historical: pd.DataFrame = pd.DataFrame(),
        weekly_historical: pd.DataFrame = pd.DataFrame(),
        monthly_historical: pd.DataFrame = pd.DataFrame(),
        quarterly_historical: pd.DataFrame = pd.DataFrame(),
        yearly_historical: pd.DataFrame = pd.DataFrame(),
        rounding: int | None = 4,
    ):
        """
        Initializes the Technical Controller Class.
        """
        if (
            daily_historical.empty
            and weekly_historical.empty
            and monthly_historical.empty
            and quarterly_historical.empty
            and yearly_historical.empty
        ):
            raise ValueError("At least one historical DataFrame is required.")

        self._tickers = tickers
        self._daily_historical = daily_historical
        self._weekly_historical = weekly_historical
        self._monthly_historical = monthly_historical
        self._quarterly_historical = quarterly_historical
        self._yearly_historical = yearly_historical
        self._rounding: int | None = rounding

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
        self._volume_indicators: pd.DataFrame = pd.DataFrame()
        self._volume_indicators_growth: pd.DataFrame = pd.DataFrame()

    def collect_all_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = 4,
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

        Note:
        - The method calculates various types of technical indicators for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
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
                self.collect_volume_indicators(
                    period=period, close_column=close_column
                ),
            ],
            axis=1,
        )

        self._all_indicators = self._all_indicators.round(
            rounding if rounding else self._rounding
        )

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
        rounding: int | None = 4,
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

        Note:
        - The method calculates various breadth indicators for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        breadth_indicators: dict = {}

        breadth_indicators["McClellan Oscillator"] = self.get_mcclellan_oscillator(
            period=period, close_column=close_column
        )

        breadth_indicators["Advancers - Decliners"] = self.get_advancers_decliners(
            period=period, close_column=close_column
        )

        self._breadth_indicators = pd.concat(breadth_indicators, axis=1)

        self._breadth_indicators = self._breadth_indicators.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._breadth_indicators_growth = calculate_growth(
                self._breadth_indicators,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        if len(self._tickers) == 1:
            return (
                self._breadth_indicators_growth[self._tickers[0]]
                if growth
                else self._breadth_indicators.loc[self._tickers[0]]
            )

        return self._breadth_indicators_growth if growth else self._breadth_indicators

    def collect_momentum_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = 4,
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

        Note:
        - The method calculates various momentum indicators for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        momentum_indicators: dict = {}

        momentum_indicators["Money Flow Index"] = self.get_money_flow_index(
            period=period, close_column=close_column, window=window
        )

        momentum_indicators["Williams %R"] = self.get_williams_percent_r(
            period=period, close_column=close_column, window=window
        )

        (
            momentum_indicators["Aroon Indicator Up"],
            momentum_indicators["Aroon Indicator Down"],
        ) = self.get_aroon_indicator(period=period, window=window)

        momentum_indicators[
            "Commodity Channel Index"
        ] = self.get_commodity_channel_index(
            period=period, close_column=close_column, window=window
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
        momentum_indicators[
            "Percentage Price Oscillator"
        ] = self.get_percentage_price_oscillator(
            period=period, close_column=close_column
        )
        momentum_indicators[
            "Detrended Price Oscillator"
        ] = self.get_detrended_price_oscillator(
            period=period, close_column=close_column, window=window
        )
        momentum_indicators[
            "Average Directional Index"
        ] = self.get_average_directional_index(
            period=period, close_column=close_column, window=window
        )
        momentum_indicators[
            "Chande Momentum Oscillator"
        ] = self.get_chande_momentum_oscillator(
            period=period, close_column=close_column, window=window
        )
        (
            momentum_indicators["Ichimoku Conversion Line"],
            momentum_indicators["Ichimoku Base Line"],
            momentum_indicators["Ichimoku Leading Span A"],
            momentum_indicators["Ichimoku Leading Span B"],
        ) = self.get_ichimoku_cloud(period=period)
        (
            momentum_indicators["Stochastic %K"],
            momentum_indicators["Stochastic %D"],
        ) = self.get_stochastic_oscillator(
            period=period, close_column=close_column, window=window
        )
        (
            momentum_indicators["MACD Line"],
            momentum_indicators["MACD Signal Line"],
        ) = self.get_moving_average_convergence_divergence(
            period=period, close_column=close_column
        )
        momentum_indicators[
            "Relative Strength Index"
        ] = self.get_relative_strength_index(
            period=period, close_column=close_column, window=window
        )
        momentum_indicators["Balance of Power"] = self.get_balance_of_power(
            period=period, close_column=close_column
        )

        self._momentum_indicators = pd.concat(momentum_indicators, axis=1)

        self._momentum_indicators = self._momentum_indicators.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._momentum_indicators_growth = calculate_growth(
                self._momentum_indicators,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        if len(self._tickers) == 1:
            return (
                self._momentum_indicators_growth[self._tickers[0]]
                if growth
                else self._momentum_indicators.loc[self._tickers[0]]
            )

        return self._momentum_indicators_growth if growth else self._momentum_indicators

    def collect_overlap_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = 4,
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

        Note:
        - The method calculates several overlap-based indicators for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        overlap_indicators: dict = {}

        overlap_indicators["Simple Moving Average (SMA)"] = self.get_moving_average(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators[
            "Exponential Moving Average (EMA)"
        ] = self.get_exponential_moving_average(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators[
            "Double Exponential Moving Average (DEMA)"
        ] = self.get_double_exponential_moving_average(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators["TRIX"] = self.get_trix(
            period=period, close_column=close_column, window=window
        )

        (
            overlap_indicators["Bollinger Band Upper"],
            overlap_indicators["Bollinger Band Middle"],
            overlap_indicators["Bollinger Band Lower"],
        ) = self.get_bollinger_bands(
            period=period, close_column=close_column, window=window
        )

        overlap_indicators[
            "Triangular Moving Average"
        ] = self.get_triangular_moving_average(
            period=period, close_column=close_column, window=window
        )

        self._overlap_indicators = pd.concat(overlap_indicators, axis=1)

        self._overlap_indicators = self._overlap_indicators.round(
            rounding if rounding else self._rounding
        )

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
                else self._overlap_indicators.loc[self._tickers[0]]
            )

        return self._overlap_indicators_growth if growth else self._overlap_indicators

    def collect_volatility_indicators(
        self,
        period: str = "daily",
        window: int = 14,
        close_column: str = "Adj Close",
        rounding: int | None = 4,
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

        Note:
        - The method calculates several volatility-based indicators for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        volatility_indicators: dict = {}

        volatility_indicators["True Range"] = self.get_true_range(
            period=period, close_column=close_column
        )

        volatility_indicators["Average True Range"] = self.get_average_true_range(
            period=period, close_column=close_column, window=window
        )

        (
            volatility_indicators["Keltner Channel Upper"],
            volatility_indicators["Keltner Channel Middle"],
            volatility_indicators["Keltner Channel Lower"],
        ) = self.get_keltner_channels(
            period=period, close_column=close_column, window=window
        )

        self._volatility_indicators = pd.concat(volatility_indicators, axis=1)

        self._volatility_indicators = self._volatility_indicators.round(
            rounding if rounding else self._rounding
        )

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
                else self._volatility_indicators.loc[self._tickers[0]]
            )

        return (
            self._volatility_indicators_growth
            if growth
            else self._volatility_indicators
        )

    def collect_volume_indicators(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects various volume indicators based on the provided data.

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
            pd.Series or pd.DataFrame: Volume indicators calculated based on the specified parameters.

        Note:
        - The method calculates several volume-based indicators for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        volume_indicators: dict = {}

        volume_indicators["On-Balance Volume"] = self.get_on_balance_volume(
            period=period, close_column=close_column
        )

        volume_indicators[
            "Accumulation/Distribution Line"
        ] = self.get_accumulation_distribution_line(
            period=period, close_column=close_column
        )

        volume_indicators["Chaikin Oscillator"] = self.get_chaikin_oscillator(
            period=period, close_column=close_column
        )

        self._volume_indicators = pd.concat(volume_indicators, axis=1)

        self._volume_indicators = self._volume_indicators.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._volume_indicators_growth = calculate_growth(
                self._volume_indicators,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        if len(self._tickers) == 1:
            return (
                self._volume_indicators_growth[self._tickers[0]]
                if growth
                else self._volume_indicators.loc[self._tickers[0]]
            )

        return self._volume_indicators_growth if growth else self._volume_indicators

    @handle_errors
    def get_mcclellan_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_ema_window: int = 19,
        long_ema_window: int = 39,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the McClellan Oscillator for a given price series.

        The McClellan Oscillator is a breadth indicator that measures the difference
        between the exponential moving average of advancing stocks and the exponential
        moving average of declining stocks.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the McClellan Oscillator for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        mcclellan_oscillator = pd.DataFrame(index=historical_data.index)
        for ticker in self._tickers:
            mcclellan_oscillator[ticker] = breadth.get_mcclellan_oscillator(
                historical_data[close_column][ticker], short_ema_window, long_ema_window
            )

        if growth:
            return calculate_growth(
                mcclellan_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return mcclellan_oscillator.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_advancers_decliners(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Advancers/Decliners ratio for a given price series.

        The Advancers/Decliners ratio is a breadth indicator that measures the number
        of advancing stocks (stocks with positive price changes) versus the number of
        declining stocks (stocks with negative price changes).

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the Advancers/Decliners ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        advancers_decliners = breadth.get_advancers_decliners(
            historical_data[close_column],
        )

        if growth:
            return calculate_growth(
                advancers_decliners,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return advancers_decliners.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_money_flow_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Money Flow Index (MFI) for a given price series.

        The Money Flow Index is a momentum indicator that measures the strength and
        direction of money flowing in and out of a security by considering both price
        and volume.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the MFI values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        money_flow_index = momentum.get_money_flow_index(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            historical_data["Volume"],
            window,
        )

        if growth:
            return calculate_growth(
                money_flow_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return money_flow_index.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_williams_percent_r(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Williams Percent R (Williams %R) for a given price series.

        The Williams %R is a momentum indicator that measures the level of the close price
        relative to the high-low range over a certain number of periods.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the Williams %R values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        williams_percent_r = momentum.get_williams_percent_r(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            window,
        )

        if growth:
            return calculate_growth(
                williams_percent_r,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return williams_percent_r.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_aroon_indicator(
        self,
        period: str = "daily",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> tuple[pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Aroon Indicator for a given price series.

        The Aroon Indicator is an oscillator that measures the strength of a trend and the
        likelihood of its continuation or reversal.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the Aroon Indicator values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        aroon_indicator_up, aroon_indicator_down = momentum.get_aroon_indicator(
            historical_data["High"], historical_data["Low"], window
        )

        if growth:
            aroon_indicator_up_growth = calculate_growth(
                aroon_indicator_up,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            aroon_indicator_down_growth = calculate_growth(
                aroon_indicator_down,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            return aroon_indicator_up_growth, aroon_indicator_down_growth

        return (
            aroon_indicator_up.round(rounding if rounding else self._rounding),
            aroon_indicator_down.round(rounding if rounding else self._rounding),
        )

    @handle_errors
    def get_commodity_channel_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        constant: float = 0.015,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Commodity Channel Index (CCI) for a given price series.

        The Commodity Channel Index is an oscillator that measures the current price level
        relative to an average price level over a specified period.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the CCI values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        commodity_channel_index = pd.DataFrame(index=historical_data.index)

        for ticker in self._tickers:
            commodity_channel_index[ticker] = momentum.get_commodity_channel_index(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
                constant,
            )

        if growth:
            return calculate_growth(
                commodity_channel_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return commodity_channel_index.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_relative_vigor_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Vigor Index (RVI) for a given price series.

        The Relative Vigor Index is an oscillator that measures the conviction of a current price
        trend using the relationship between closing and opening prices.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the RVI values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        relative_vigor_index = momentum.get_relative_vigor_index(
            historical_data["Open"],
            historical_data[close_column],
            historical_data["Volume"],
            window,
        )

        if growth:
            return calculate_growth(
                relative_vigor_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return relative_vigor_index.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_force_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Force Index for a given price series.

        The Force Index is an indicator that measures the strength behind price movements.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the Force Index values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        force_index = momentum.get_force_index(
            historical_data[close_column],
            historical_data["Volume"],
            window,
        )

        if growth:
            return calculate_growth(
                force_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return force_index.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_ultimate_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window_1: int = 7,
        window_2: int = 14,
        window_3: int = 28,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Ultimate Oscillator for a given price series.

        The Ultimate Oscillator is a momentum oscillator that combines short-term, mid-term,
        and long-term price momentum into a single value.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ultimate Oscillator values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        ultimate_oscillator = pd.DataFrame(index=historical_data.index)
        for ticker in self._tickers:
            ultimate_oscillator[ticker] = momentum.get_ultimate_oscillator(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window_1,
                window_2,
                window_3,
            )

        if growth:
            ultimate_oscillator_growth = calculate_growth(
                ultimate_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            return ultimate_oscillator_growth

        return ultimate_oscillator.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_percentage_price_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 7,
        long_window: int = 28,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Percentage Price Oscillator (PPO) for a given price series.

        The Percentage Price Oscillator (PPO) is a momentum oscillator that measures the
        difference between two moving averages as a percentage of the longer moving average.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the PPO values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        percentage_price_oscillator = momentum.get_percentage_price_oscillator(
            historical_data[close_column],
            short_window,
            long_window,
        )

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

    @handle_errors
    def get_detrended_price_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Detrended Price Oscillator (DPO) for a given price series.

        The Detrended Price Oscillator (DPO) is an indicator that helps identify short-term cycles
        by removing longer-term trends from prices.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the DPO values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        detrended_price_oscillator = momentum.get_detrended_price_oscillator(
            historical_data[close_column], window
        )

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

    @handle_errors
    def get_average_directional_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX) for a given price series.

        The Average Directional Index (ADX) is an indicator that measures the strength of a trend,
        whether it's an uptrend or a downtrend.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the ADX values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        average_directional_index = pd.DataFrame(index=historical_data.index)
        for ticker in self._tickers:
            average_directional_index[ticker] = momentum.get_average_directional_index(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
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

    @handle_errors
    def get_chande_momentum_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chande Momentum Oscillator (CMO) for a given price series.

        The Chande Momentum Oscillator is an indicator that measures the momentum of a price
        series and identifies overbought and oversold conditions.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the Chande Momentum Oscillator values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        chande_momentum_oscillator = momentum.get_chande_momentum_oscillator(
            historical_data[close_column], window
        )

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

    @handle_errors
    def get_ichimoku_cloud(
        self,
        period: str = "daily",
        conversion_window: int = 9,
        base_window: int = 20,
        lead_span_b_window: int = 40,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Ichimoku Cloud indicator for a given price series.

        The Ichimoku Cloud, also known as the Ichimoku Kinko Hyo, is a versatile indicator that
        defines support and resistance, identifies trend direction, gauges momentum, and provides
        trading signals.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the Ichimoku Cloud values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the indicator values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        (
            conversion_line,
            base_line,
            lead_span_a,
            lead_span_b,
        ) = momentum.get_ichimoku_cloud(
            historical_data["High"],
            historical_data["Low"],
            conversion_window,
            base_window,
            lead_span_b_window,
        )

        if growth:
            conversion_line_growth = calculate_growth(
                conversion_line,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            base_line_growth = calculate_growth(
                base_line,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            lead_span_a_growth = calculate_growth(
                lead_span_a,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            lead_span_b_growth = calculate_growth(
                lead_span_b,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return (
                conversion_line_growth,
                base_line_growth,
                lead_span_a_growth,
                lead_span_b_growth,
            )

        return (
            conversion_line.round(rounding if rounding else self._rounding),
            base_line.round(rounding if rounding else self._rounding),
            lead_span_a.round(rounding if rounding else self._rounding),
            lead_span_b.round(rounding if rounding else self._rounding),
        )

    @handle_errors
    def get_stochastic_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        smooth_widow: int = 3,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Stochastic Oscillator indicator for a given price series.

        The Stochastic Oscillator is a momentum indicator that shows the location of the close
        relative to the high-low range over a set number of periods. It consists of the %K line
        (fast) and the %D line (slow).

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the Stochastic Oscillator values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the %K and %D values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        stochastic_d, stochastic_k = momentum.get_stochastic_oscillator(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            window,
            smooth_widow,
        )

        if growth:
            stochastic_k_growth = calculate_growth(
                stochastic_k,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            stochastic_d_growth = calculate_growth(
                stochastic_d,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return stochastic_k_growth, stochastic_d_growth

        return (
            stochastic_k.round(rounding if rounding else self._rounding),
            stochastic_d.round(rounding if rounding else self._rounding),
        )

    @handle_errors
    def get_moving_average_convergence_divergence(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> tuple[pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Moving Average Convergence Divergence (MACD) indicator for a given price series.

        The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator
        that shows the relationship between two moving averages of a security's price. It consists
        of the MACD line, signal line, and MACD histogram.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates
          the MACD and signal line values for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the MACD and signal values
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        macd, signal = momentum.get_moving_average_convergence_divergence(
            historical_data[close_column], short_window, long_window, signal_window
        )

        if growth:
            macd_growth = calculate_growth(
                macd,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            signal_growth = calculate_growth(
                signal,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return macd_growth, signal_growth

        return (
            macd.round(rounding if rounding else self._rounding),
            signal.round(rounding if rounding else self._rounding),
        )

    @handle_errors
    def get_relative_strength_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI) indicator for a given price series.

        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and
        change of price movements. It ranges from 0 to 100 and is used to identify overbought or
        oversold conditions in an asset's price.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          RSI for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the RSI
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        relative_strength_index = momentum.get_relative_strength_index(
            historical_data[close_column], window
        )

        if growth:
            return calculate_growth(
                relative_strength_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return relative_strength_index.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_balance_of_power(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Balance of Power (BOP) indicator for a given price series.

        The Balance of Power (BOP) indicator measures the strength of buyers versus sellers
        in the market. It relates the price change to the change in the asset's trading range.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          BOP for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the BOP
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        balance_of_power = momentum.get_balance_of_power(
            historical_data["Open"],
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
        )

        if growth:
            return calculate_growth(
                balance_of_power,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return balance_of_power.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Moving Average (MA) for a given price series.

        The Moving Average (MA) is a commonly used technical indicator that smooths out
        price data by calculating the average price over a specified number of periods.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          MA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the MA
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        moving_average = overlap.get_moving_average(
            historical_data[close_column], window
        )

        if growth:
            return calculate_growth(
                moving_average,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return moving_average.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_exponential_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Exponential Moving Average (EMA) for a given price series.

        EMA is a technical indicator that gives more weight to recent price data,
        providing a smoothed moving average that reacts faster to price changes.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          EMA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the EMA
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        exponential_moving_average = overlap.get_exponential_moving_average(
            historical_data[close_column], window
        )

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

    @handle_errors
    def get_double_exponential_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Double Exponential Moving Average (DEMA) for a given price series.

        DEMA is a technical indicator that attempts to reduce the lag from traditional
        moving averages by using a combination of two exponential moving averages.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          DEMA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the DEMA
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        double_exponential_moving_average = (
            overlap.get_double_exponential_moving_average(
                historical_data[close_column], window
            )
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

    @handle_errors
    def get_trix(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Trix (Triple Exponential Moving Average) for a given price series.

        Trix is a momentum oscillator that calculates the percentage rate of change of a triple
        exponentially smoothed moving average. It helps identify overbought and oversold conditions
        in a market.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          Trix for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Trix
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        trix = overlap.get_trix(historical_data[close_column], window)

        if growth:
            return calculate_growth(
                trix,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return trix.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_bollinger_bands(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        num_std_dev: int = 2,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Bollinger Bands for a given price series.

        Bollinger Bands are a volatility indicator that consists of three lines: an upper band,
        a middle band (simple moving average), and a lower band. The upper and lower bands are
        calculated as the moving average plus and minus a specified number of standard deviations,
        respectively.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          Bollinger Bands for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Bollinger Bands
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        upper_band, middle_band, lower_band = overlap.get_bollinger_bands(
            historical_data[close_column], window, num_std_dev
        )

        if growth:
            upper_band_growth = calculate_growth(
                upper_band,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            middle_band_growth = calculate_growth(
                middle_band,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            lower_band_growth = calculate_growth(
                lower_band,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return upper_band_growth, middle_band_growth, lower_band_growth

        upper_band = upper_band.round(rounding if rounding else self._rounding)
        middle_band = middle_band.round(rounding if rounding else self._rounding)
        lower_band = lower_band.round(rounding if rounding else self._rounding)

        return upper_band, middle_band, lower_band

    @handle_errors
    def get_triangular_moving_average(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Triangular Moving Average (TMA) for a given price series.

        The Triangular Moving Average (TMA) is a smoothed version of the Simple Moving Average (SMA)
        that uses multiple SMAs to reduce noise and provide a smoother trendline.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          Triangular Moving Average for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Triangular Moving Average
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        triangular_moving_average = overlap.get_triangular_moving_average(
            historical_data[close_column], window
        )

        if growth:
            return calculate_growth(
                triangular_moving_average,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return triangular_moving_average.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_true_range(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the True Range (TR) for a given price series.

        The True Range (TR) is a measure of market volatility that considers the differences
        between the high and low prices and the previous closing price. It provides insights
        into the price movement of an asset.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          True Range for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the True Range
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        true_range = pd.DataFrame(index=historical_data.index)
        for ticker in self._tickers:
            true_range[ticker] = volatility.get_true_range(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
            )

        if growth:
            return calculate_growth(
                true_range,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return true_range.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_average_true_range(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series:
        """
        Calculate the Average True Range (ATR) of a given price series.

        The Average True Range (ATR) is a technical indicator that measures the volatility
        of an asset's price movements over a specified number of periods. It provides insights
        into the potential price range of an asset, which can help traders and investors make
        more informed decisions.

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

        Note:
        - ATR values are typically used to assess the volatility and potential price movement of an asset.
        - A higher ATR value indicates higher volatility, while a lower ATR value suggests lower volatility.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly or yearly.")

        average_true_range = pd.DataFrame(index=historical_data.index)
        for ticker in self._tickers:
            average_true_range[ticker] = volatility.get_average_true_range(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
            )

        if growth:
            return calculate_growth(
                average_true_range,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return average_true_range.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_keltner_channels(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        atr_window: int = 14,
        atr_multiplier: int = 2,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate the Keltner Channels for a given price series.

        The Keltner Channels consist of three lines:
        - Upper Channel Line = Exponential Moving Average (EMA) of High Prices + ATR * ATR Multiplier
        - Middle Channel Line = Exponential Moving Average (EMA) of Closing Prices
        - Lower Channel Line = Exponential Moving Average (EMA) of Low Prices - ATR * ATR Multiplier

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates Keltner Channels
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the channels using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        upper_line = pd.DataFrame(index=historical_data.index)
        middle_line = pd.DataFrame(index=historical_data.index)
        lower_line = pd.DataFrame(index=historical_data.index)

        for ticker in self._tickers:
            (
                upper_line[ticker],
                middle_line[ticker],
                lower_line[ticker],
            ) = volatility.get_keltner_channels(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
                atr_window,
                atr_multiplier,
            )

        if growth:
            upper_line_growth = calculate_growth(
                upper_line,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            middle_line_growth = calculate_growth(
                middle_line,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            lower_line_growth = calculate_growth(
                lower_line,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

            return upper_line_growth, middle_line_growth, lower_line_growth

        upper_line = upper_line.round(rounding if rounding else self._rounding)
        middle_line = middle_line.round(rounding if rounding else self._rounding)
        lower_line = lower_line.round(rounding if rounding else self._rounding)

        return upper_line, middle_line, lower_line

    @handle_errors
    def get_on_balance_volume(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the On-Balance Volume (OBV) for a given price series.

        The On-Balance Volume (OBV) is a technical indicator that uses volume flow to predict changes in stock price.
        It accumulates the volume on up days and subtracts the volume on down days. The resulting OBV line provides
        insights into the buying and selling pressure behind price movements.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates On-Balance Volume
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the OBV using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        on_balance_volume = volume.get_on_balance_volume(
            historical_data[close_column],
            historical_data["Volume"],
        )

        if growth:
            return calculate_growth(
                on_balance_volume,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return on_balance_volume.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_accumulation_distribution_line(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Accumulation/Distribution Line for a given price series.

        The Accumulation/Distribution Line is a technical indicator that evaluates the flow of money
        into or out of an asset. It takes into account both price and volume information to identify
        whether an asset is being accumulated (bought) or distributed (sold) by investors.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          Accumulation/Distribution Line for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Accumulation/Distribution Line
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        accumulation_distribution_line = pd.DataFrame(index=historical_data.index)
        for ticker in self._tickers:
            accumulation_distribution_line[
                ticker
            ] = volume.get_accumulation_distribution_line(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                historical_data["Volume"][ticker],
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

    @handle_errors
    def get_chaikin_oscillator(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 3,
        long_window: int = 10,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate the Chaikin Oscillator for a given price series.

        The Chaikin Oscillator is a momentum-based indicator that combines price and volume
        to help identify potential trends and reversals in the market. It is calculated as the
        difference between the 3-day and 10-day Accumulation/Distribution Line.

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

        Note:
        - The method retrieves historical data based on the specified `period` and calculates the
          Chaikin Oscillator for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Chaikin Oscillator
          using the specified `lag`.
        """
        if period == "daily":
            historical_data = self._daily_historical
        elif period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be daily, weekly, quarterly, or yearly.")

        chaikin_oscillator = volume.get_chaikin_oscillator(
            historical_data["High"],
            historical_data["Low"],
            historical_data[close_column],
            historical_data["Volume"],
            short_window,
            long_window,
        )

        if growth:
            return calculate_growth(
                chaikin_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return chaikin_oscillator.round(rounding if rounding else self._rounding)
