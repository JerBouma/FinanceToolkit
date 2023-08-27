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
    ):
        """
        Calculates all Technical Indicators based on the data provided.

        Note that due to the nature of the calculations, the dataset can get quite extensive.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "daily".
            window (int, optional): The number of days to use for the calculation. Defaults to 14.
            close_column (str, optional): The column to use for the calculation. Defaults to "Adj Close".
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_all_ratios()
        ```
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
    ):
        """
        Calculates all Efficiency Ratios based on the data provided.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_efficiency_ratios()
        ```
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
    ):
        """
        Calculates all Efficiency Ratios based on the data provided.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_efficiency_ratios()
        ```
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
            "Deterended Price Oscillator"
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
    ):
        """
        Calculates all Efficiency Ratios based on the data provided.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_efficiency_ratios()
        ```
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
    ):
        """
        Calculates all Efficiency Ratios based on the data provided.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_efficiency_ratios()
        ```
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
            volatility_indicators["Keltner Channels Lower"],
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
    ):
        """
        Calculates all Efficiency Ratios based on the data provided.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_efficiency_ratios()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        mcclelland_oscillator = pd.DataFrame(index=historical_data.index)
        for ticker in self._tickers:
            mcclelland_oscillator[ticker] = breadth.get_mcclellan_oscillator(
                historical_data[close_column][ticker], short_ema_window, long_ema_window
            )

        if growth:
            return calculate_growth(
                mcclelland_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return mcclelland_oscillator.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_advancers_decliners(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        advanced_decliners = breadth.get_advancers_decliners(
            historical_data[close_column],
        )

        if growth:
            return calculate_growth(
                advanced_decliners,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return advanced_decliners.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_money_flow_index(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        return aroon_indicator_up.round(
            rounding if rounding else self._rounding
        ), aroon_indicator_down.round(rounding if rounding else self._rounding)

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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
            return calculate_growth(
                ultimate_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        percentage_price_oscillator = momentum.get_percentage_price_oscillator(
            historical_data[close_column],
            short_window,
            long_window,
        )

        if growth:
            return calculate_growth(
                percentage_price_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        detrended_price_oscillator = momentum.get_detrended_price_oscillator(
            historical_data[close_column], window
        )

        if growth:
            return calculate_growth(
                detrended_price_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        average_directional_index = pd.DataFrame(index=historical_data.index)
        for ticker in self._tickers:
            average_directional_index[ticker] = momentum.get_average_directional_index(
                historical_data["High"][ticker],
                historical_data["Low"][ticker],
                historical_data[close_column][ticker],
                window,
            )

        if growth:
            return calculate_growth(
                average_directional_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        chande_momentum_oscillator = momentum.get_chande_momentum_oscillator(
            historical_data[close_column], window
        )

        if growth:
            return calculate_growth(
                chande_momentum_oscillator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        conversion_line = conversion_line.round(
            rounding if rounding else self._rounding
        )
        base_line = base_line.round(rounding if rounding else self._rounding)
        lead_span_a = lead_span_a.round(rounding if rounding else self._rounding)
        lead_span_b = lead_span_b.round(rounding if rounding else self._rounding)

        return conversion_line, base_line, lead_span_a, lead_span_b

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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        stochastic_k = stochastic_k.round(rounding if rounding else self._rounding)
        stochastic_d = stochastic_d.round(rounding if rounding else self._rounding)

        return stochastic_k, stochastic_d

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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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

        return macd.round(rounding if rounding else self._rounding), signal.round(
            rounding if rounding else self._rounding
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
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
