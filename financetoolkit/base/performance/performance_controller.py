"""Performance Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import calculate_growth, handle_errors
from financetoolkit.base.performance.helpers import (
    handle_return_data_periods,
    handle_risk_free_data_periods,
)
from financetoolkit.performance import performance

# pylint: disable=too-many-instance-attributes,too-few-public-methods


class Performance:
    """
    Performance Controller Class
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: dict[str, pd.DataFrame],
        risk_free_rate_data: pd.DataFrame,
        benchmark_ticker: str | None = None,
        quarterly: bool | None = None,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Performance Controller Class.
        """
        self._tickers = tickers
        self._benchmark_ticker = benchmark_ticker
        self._quarterly: bool | None = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date

        # Historical Data
        self._daily_historical_data = historical_data["daily"]
        self._weekly_historical_data = historical_data["weekly"]
        self._monthly_historical_data = historical_data["monthly"]
        self._quarterly_historical_data = historical_data["quarterly"]
        self._yearly_historical_data = historical_data["yearly"]

        # Risk Free Data
        self._daily_risk_free_rate_data = risk_free_rate_data["daily"]
        self._weekly_risk_free_rate_data = risk_free_rate_data["weekly"]
        self._monthly_risk_free_rate_data = risk_free_rate_data["monthly"]
        self._quarterly_risk_free_rate_data = risk_free_rate_data["quarterly"]
        self._yearly_risk_free_rate_data = risk_free_rate_data["yearly"]

        # Within Period Calculations
        self._weekly_within_historical_data = self._daily_historical_data.groupby(
            pd.Grouper(freq="W")
        ).apply(lambda x: x)
        self._monthly_within_historical_data = self._daily_historical_data.groupby(
            pd.Grouper(freq="M")
        ).apply(lambda x: x)
        self._quarterly_within_historical_data = self._daily_historical_data.groupby(
            pd.Grouper(freq="Q")
        ).apply(lambda x: x)
        self._yearly_within_historical_data = self._daily_historical_data.groupby(
            pd.Grouper(freq="Y")
        ).apply(lambda x: x)

    @handle_errors
    def get_beta(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Sharpe ratio, a measure of risk-adjusted return that evaluates the excess return
        of an investment portfolio or asset per unit of risk taken.

        The Sharpe ratio is calculated as the difference between the expected return of the asset or portfolio
        and the risk-free rate of return, divided by the standard deviation of the asset or portfolio's excess return.
        It quantifies the amount of return generated for each unit of risk assumed, providing insights into the
        investment's performance relative to the risk taken.

        Formula is as follows:

            Beta = Covariance of Asset Returns and Benchmark Returns / Variance of Benchmark Returns

        For a given period, for example monthly, this translates into the following:

            Beta = Monthly Covariance of Asset Returns and Benchmark Returns
            / Monthly Variance of Benchmark Returns

        See definition: https://en.wikipedia.org/wiki/Beta_(finance)

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling period to use for the calculation. If you select
            period = 'monthly' and set rolling to 12 you obtain the rolling 12-month Sharpe Ratio.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Sharpe ratio values.

        Notes:
        - Daily Sharpe Ratio is not an option as the standard deviation for 1 days is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the Sharpe ratio for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_sharpe_ratio()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period == "daily":
            raise ValueError(
                "Daily Beta is not an option as standard deviation for 1 day "
                "is close to zero. Therefore, it does not give any useful insights."
            )

        historical_data = handle_return_data_periods(
            self, period, within_period=not rolling
        )
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_ticker]

        if rolling:
            beta = performance.get_rolling_beta(returns, benchmark_returns, rolling)
        else:
            beta = performance.get_beta(returns, benchmark_returns)

        beta = beta.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        if growth:
            return calculate_growth(
                beta,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return beta

    @handle_errors
    def get_capital_asset_pricing_model(
        self,
        period: str | None = None,
        show_full_results: bool = False,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Sharpe ratio, a measure of risk-adjusted return that evaluates the excess return
        of an investment portfolio or asset per unit of risk taken.

        The Sharpe ratio is calculated as the difference between the expected return of the asset or portfolio
        and the risk-free rate of return, divided by the standard deviation of the asset or portfolio's excess return.
        It quantifies the amount of return generated for each unit of risk assumed, providing insights into the
        investment's performance relative to the risk taken.

        Formula is as follows:

            Capital Asset Pricing Model = Risk Free Rate + Beta * (Benchmark Returns - Risk Free Rate)

        See definition: https://en.wikipedia.org/wiki/Capital_asset_pricing_model

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            show_full_results (bool, optional): Whether to show the full results. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Sharpe ratio values.

        Notes:
        - Daily Sharpe Ratio is not an option as the standard deviation for 1 days is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the Sharpe ratio for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_sharpe_ratio()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(self, period, within_period=True)
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_ticker]

        beta = performance.get_beta(returns, benchmark_returns)

        risk_free_rate = handle_risk_free_data_periods(self, period)
        benchmark_returns = handle_return_data_periods(
            self, period, within_period=False
        ).loc[:, "Return"][self._benchmark_ticker]

        capm = performance.get_capital_asset_pricing_model(
            risk_free_rate, beta, benchmark_returns
        )

        capm = capm.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        if show_full_results:
            full_results = pd.concat(
                [risk_free_rate, beta, benchmark_returns, capm], axis=1
            ).loc[self._start_date : self._end_date]
            full_results.columns = (
                ["Risk Free Rate"]
                + [f"Beta {ticker}" for ticker in self._tickers]
                + ["Benchmark Returns"]
                + [f"CAPM {ticker}" for ticker in self._tickers]
            )

            return full_results

        if growth:
            return calculate_growth(
                capm,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return capm

    @handle_errors
    def get_sharpe_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Sharpe ratio, a measure of risk-adjusted return that evaluates the excess return
        of an investment portfolio or asset per unit of risk taken.

        The Sharpe ratio is calculated as the difference between the expected return of the asset or portfolio
        and the risk-free rate of return, divided by the standard deviation of the asset or portfolio's excess return.
        It quantifies the amount of return generated for each unit of risk assumed, providing insights into the
        investment's performance relative to the risk taken.

        Formula is as follows:

            Sharpe Ratio = Excess Return / Excess Standard Deviation

        For a given period, for example monthly, this translates into the following:

            Sharpe Ratio = Average Monthly Excess Return / Standard Deviation of Monthly Excess Returns

        For a rolling period, this translates into the following:

            Sharpe Ratio = Average Rolling Excess Return / Standard Deviation of Rolling Excess Returns

        Note that this is explicitly already subtracts the Risk Free Rate.

        See definition: https://en.wikipedia.org/wiki/Sharpe_ratio

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling period to use for the calculation. If you select
            period = 'monthly' and set rolling to 12 you obtain the rolling 12-month Sharpe Ratio.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Sharpe ratio values.

        Notes:
        - Daily Sharpe Ratio is not an option as the standard deviation for 1 days is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the Sharpe ratio for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_sharpe_ratio()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period == "daily":
            raise ValueError(
                "Daily Sharpe Ratio is not an option as standard deviation for 1 day "
                "is close to zero. Therefore, it does not give any useful insights."
            )

        historical_data = handle_return_data_periods(
            self, period, within_period=not rolling
        )
        excess_return = historical_data.loc[:, "Excess Return"][self._tickers]

        if rolling:
            sharpe_ratio = performance.get_rolling_sharpe_ratio(excess_return, rolling)
        else:
            sharpe_ratio = performance.get_sharpe_ratio(excess_return)

        sharpe_ratio = sharpe_ratio.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        if growth:
            return calculate_growth(
                sharpe_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return sharpe_ratio

    @handle_errors
    def get_sortino_ratio(
        self,
        period: str | None = None,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Sharpe ratio, a measure of risk-adjusted return that evaluates the excess return
        of an investment portfolio or asset per unit of risk taken.

        The Sharpe ratio is calculated as the difference between the expected return of the asset or portfolio
        and the risk-free rate of return, divided by the standard deviation of the asset or portfolio's excess return.
        It quantifies the amount of return generated for each unit of risk assumed, providing insights into the
        investment's performance relative to the risk taken.

        Formula is as follows:

            Sortino Ratio = Excess Return / Excess Downside Risk

        For a given period, for example monthly, this translates into the following:

            Sortino Ratio = Average Monthly Excess Return / Average Monthly Excess Downside Risk

        For a rolling period, this translates into the following:

            Sortino Ratio = Average Rolling Excess Return / Rolling Downside Risk

        Note that this is explicitly already subtracts the Risk Free Rate.

        See definition: https://en.wikipedia.org/wiki/Sortino_ratio

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Sharpe ratio values.

        Notes:
        - Daily Sortino Ratio is not an option as the standard deviation for 1 days is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the Sortino ratio for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_sortino_ratio()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period == "daily":
            raise ValueError(
                "Daily Sortino Ratio is not an option as standard deviation for 1 day "
                "is close to zero. Therefore, it does not give any useful insights."
            )

        historical_data = handle_return_data_periods(self, period, within_period=True)
        excess_return = historical_data.loc[:, "Excess Return"][self._tickers]

        sortino_ratio = performance.get_sortino_ratio(excess_return)

        sortino_ratio = sortino_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                sortino_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return sortino_ratio
