"""Performance Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import calculate_growth
from financetoolkit.base.performance.helpers import (
    handle_errors,
    handle_return_data_periods,
    handle_risk_free_data_periods,
)
from financetoolkit.performance import performance
from financetoolkit.risk.risk import get_ui

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines


class Performance:
    """
    The Performance module is meant to calculate important performance metrics such
    as Sharpe Ratio, Sortino Ratio, Treynor Ratio, Information Ratio, Jensen’s
    Alpha, Beta, Capital Asset Pricing Model, R-Squared and more.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: dict[str, pd.DataFrame],
        risk_free_rate_data: pd.DataFrame,
        quarterly: bool | None = None,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Performance Controller Class.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_capital_asset_pricing_model(period='quarterly')
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2022Q3 | -0.0684 | -0.1047 |
        | 2022Q4 |  0.0857 |  0.0828 |
        | 2023Q1 |  0.075  |  0.1121 |
        | 2023Q2 |  0.0922 |  0.1342 |
        | 2023Q3 |  0.0052 | -0.0482 |
        """
        self._tickers = tickers
        self._benchmark_name = "Benchmark"
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
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Beta, a measurement that assess the systematic risk of a stock or investment.

        Beta is a financial metric used to assess the systematic risk of a stock or investment in
        relation to the overall market. It provides valuable insights into how a particular asset's
        returns tend to move in response to fluctuations in the broader market. A stock's Beta is
        calculated by analyzing its historical price movements and their correlation with the
        movements of a market index, typically the benchmark index like the S&P 500.

        The formula is as follows:

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
            pd.DataFrame: Beta values.

        Notes:
        - Daily Beta is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the Beta for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "AMZN"], api_key=FMP_KEY)

        toolkit.performance.get_beta()
        ```
        """
        if period == "daily":
            print(
                "Daily Beta is not an option as the standard deviation for 1 day is close to zero. "
                "Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(
            self, period, within_period=not rolling
        )
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

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
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        CAPM, or the Capital Asset Pricing Model, is a financial model used to estimate the expected return
        on an investment, such as a stock or portfolio of stocks. It provides a framework for evaluating the
        risk and return trade-off of an asset or portfolio in relation to the overall market. CAPM is based
        on the following key components:

            - Risk-Free Rate (Rf): This is the theoretical return an investor could earn from an investment
            with no risk of financial loss. It is typically based on the yield of a government bond.
            - Market Risk Premium (Rm - Rf): This represents the additional return that investors expect
            to earn for taking on the risk of investing in the overall market as opposed to a risk-free asset.
            It is calculated as the difference between the expected return of the market (Rm) and the risk-free
            rate (Rf).
            - Beta (β): Beta is a measure of an asset's or portfolio's sensitivity to market movements. It
            quantifies how much an asset's returns are expected to move in relation to changes in the
            overall market. A beta of 1 indicates that the asset moves in line with the market, while a
            beta greater than 1 suggests higher volatility, and a beta less than 1 indicates lower volatility.

        The Capital Asset Pricing Model (CAPM) is a widely used financial model that helps in
        determining the expected return of an asset or portfolio based on its systematic risk and
        the prevailing risk-free rate in the market. CAPM provides insights into how an asset or
        investment should be priced in order to offer an appropriate rate of return, given its
        level of risk compared to the overall market.

        The formula is as follows:

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
            pd.DataFrame: CAPM values.

        Notes:
        - Daily CAPM is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the CAPM for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.gget_capital_asset_pricing_model()
        ```
        """
        if period == "daily":
            print(
                "Daily CAPM is not an option as the standard deviation for 1 day is close to zero. "
                "Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(self, period, within_period=True)
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        beta = performance.get_beta(returns, benchmark_returns)

        risk_free_rate = handle_risk_free_data_periods(self, period)
        benchmark_returns = handle_return_data_periods(
            self, period, within_period=False
        ).loc[:, "Return"][self._benchmark_name]

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

            full_results = full_results.round(rounding if rounding else self._rounding)

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
    def get_alpha(
        self,
        period: str | None = None,
        show_full_results: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Alpha, in a general sense, represents the excess return an investment generates relative to
        a benchmark or a risk-adjusted return. It can be positive (indicating the investment
        outperformed the benchmark) or negative (indicating underperformance).

        The formula is as follows:

            Alpha = Asset's Actual Return - Benchmark's Actual Return

        See definition: https://en.wikipedia.org/wiki/Alpha_(finance)

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Alpha values.

        Notes:
        - The method retrieves historical data and calculates the Alpha for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_alpha()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(self, period, within_period=False)
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        alpha = performance.get_alpha(returns, benchmark_returns)

        alpha = alpha.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        if show_full_results:
            full_results = pd.concat([returns, benchmark_returns, alpha], axis=1).loc[
                self._start_date : self._end_date
            ]
            full_results.columns = (
                [f"Actual Return {ticker}" for ticker in self._tickers]
                + ["Benchmark Returns"]
                + [f"Alpha {ticker}" for ticker in self._tickers]
            )

            return full_results

        if growth:
            return calculate_growth(
                alpha,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return alpha

    @handle_errors
    def get_jensens_alpha(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate Jensen's Alpha, a measure of an asset's performance relative to its expected return
        based on the Capital Asset Pricing Model (CAPM).

        Jensen's Alpha is used to assess whether an investment has outperformed or underperformed
        its expected return given its systematic risk, as represented by the asset's Beta.

        The formula for Jensen's Alpha is as follows:

            Jensen's Alpha = Asset's Actual Return - [Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate)]

        See definition: https://en.wikipedia.org/wiki/Jensen%27s_alpha

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Jensen's Alpha values.

        Notes:
        - Daily Jensen's Alpha is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the CAPM for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_jensens_alpha()
        ```
        """
        if period == "daily":
            print(
                "Daily Jensen's Alpha is not an option as the standard deviation for 1 day is close to zero. "
                "Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(self, period, within_period=True)
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        beta = performance.get_beta(returns, benchmark_returns)

        historical_period_data = handle_return_data_periods(
            self, period, within_period=False
        )
        period_returns = historical_period_data.loc[:, "Return"][self._tickers]

        risk_free_rate = handle_risk_free_data_periods(self, period)
        benchmark_returns = handle_return_data_periods(
            self, period, within_period=False
        ).loc[:, "Return"][self._benchmark_name]

        jensens_alpha = performance.get_jensens_alpha(
            period_returns, risk_free_rate, beta, benchmark_returns
        )

        jensens_alpha = jensens_alpha.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                jensens_alpha,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return jensens_alpha

    @handle_errors
    def get_treynor_ratio(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        The Treynor Ratio, also known as Treynor's Measure or the Reward-to-Variability Ratio, is a
        financial metric used to assess the risk-adjusted performance of an investment portfolio or
        asset. It measures the excess return generated by the portfolio per unit of systematic or market
        risk, often represented by Beta. The Treynor Ratio is a valuable tool for evaluating the
        performance of investments in relation to their market risk exposure.

        The formula is as follows:

            Treynor Ratio = (Portfolio's Return - Risk-Free Rate) / Portfolio Beta

        See definition: https://en.wikipedia.org/wiki/Treynor_ratio

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Treynor Ratio values.

        Notes:
        - Daily Treynor Ratio is not an option as the standard deviation for 1 day is close to zero. Therefore,
        it does not give any useful insights.
        - The method retrieves historical data and calculates the TR for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_treynor_ratio()
        ```
        """
        if period == "daily":
            print(
                "Daily Treynor Ratio is not an option as the standard deviation for 1 day is close to zero. "
                "Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(self, period, within_period=True)
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        beta = performance.get_beta(returns, benchmark_returns)

        historical_period_data = handle_return_data_periods(
            self, period, within_period=False
        )
        period_returns = historical_period_data.loc[:, "Return"][self._tickers]
        risk_free_rate = handle_risk_free_data_periods(self, period)

        treynor_ratio = performance.get_treynor_ratio(
            period_returns, risk_free_rate, beta
        )

        treynor_ratio = treynor_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                treynor_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return treynor_ratio

    @handle_errors
    def get_sharpe_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
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

        The formula is as follows:

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
        - Daily Sharpe Ratio is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
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
        if period == "daily":
            print(
                "Daily Sharpe Ratio is not an option as standard deviation for 1 day "
                "is close to zero. Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

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
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """

        The Sortino Ratio is a financial metric used to assess the risk-adjusted performance of an
        investment portfolio or asset by considering only the downside risk. It measures the excess
        return generated by the portfolio per unit of downside risk, specifically, the standard
        deviation of negative returns. The Sortino Ratio is particularly useful for investors who
        are primarily concerned with minimizing the downside risk of their investments.

        The formula is as follows:

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
            pd.DataFrame: Sortino ratio values.

        Notes:
        - Daily Sortino Ratio is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
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
        if period == "daily":
            print(
                "Daily Sortino Ratio is not an option as standard deviation for 1 day "
                "is close to zero. Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

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

    @handle_errors
    def get_ulcer_performance_index(
        self,
        period: str | None = None,
        rolling: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Ulcer Performance Index (UPI), alternatively called Martin ratio, a measure of risk-adjusted
        return that evaluates the excess return of an investment portfolio or asset per unit of risk taken.

        It can be used to compare volatilities in different stocks or show stocks go into Ulcer territory.
        Similair to the Sharpe Ratio, a higher UPI is better than a lower one (since investors prefer more return
        for less risk).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int): The rolling period to use to calculate the Ulcer Index. Defaults to 14.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Ulcer Performance Index values.

        Notes:
        - The method retrieves historical data and calculates the UPI for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_ulcer_performance_index()
        ```
        """

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(self, period, True)
        returns = historical_data.loc[:, "Return"][self._tickers]
        historical_data_within_period = handle_return_data_periods(self, period, False)
        excess_return = historical_data_within_period.loc[:, "Excess Return"][
            self._tickers
        ]

        ulcer_index = get_ui(returns, rolling)

        ulcer_performance_index = performance.get_ulcer_performance_index(
            excess_return, ulcer_index
        )
        ulcer_performance_index = ulcer_performance_index.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                ulcer_performance_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return ulcer_performance_index

    @handle_errors
    def get_m2_ratio(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        The M2 Ratio, also known as the Modigliani-Modigliani Measure, is a financial
        metric used to evaluate the risk-adjusted performance of an investment portfolio
        or strategy. It assesses the excess return generated by the portfolio relative
        to a risk-free investment, taking into account the portfolio's volatility or risk.
        The M2 Ratio helps investors and portfolio managers determine whether the portfolio
        is delivering returns that justify its level of risk.

        The formula is as follows:

            M2 Ratio = (Portfolio's Return - Risk-Free Rate) / Portfolio Standard Deviation

        See definition: https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: M2 ratio values.

        Notes:
        - Daily M2 is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the M2 for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_m2_ratio()
        ```
        """
        if period == "daily":
            print(
                "Daily M@ Ratio is not an option as the standard deviation for 1 day is close to zero. "
                "Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_period_data = handle_return_data_periods(
            self, period, within_period=False
        )
        period_returns = historical_period_data.loc[:, "Return"][self._tickers]
        period_standard_deviation = historical_period_data.loc[:, "Volatility"][
            self._tickers
        ]
        risk_free_rate = handle_risk_free_data_periods(self, period)

        m2_ratio = performance.get_m2_ratio(
            period_returns, risk_free_rate, period_standard_deviation
        )

        m2_ratio = m2_ratio.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        if growth:
            return calculate_growth(
                m2_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return m2_ratio

    @handle_errors
    def get_tracking_error(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Tracking Error is a financial metric that quantifies the volatility or dispersion of the
        difference between the returns of an investment portfolio or asset and the returns of a
        benchmark index. It measures how closely the portfolio tracks its benchmark and provides
        insights into the consistency of the portfolio's performance relative to the benchmark.
        A higher Tracking Error indicates greater divergence from the benchmark, while a lower
        Tracking Error suggests that the portfolio closely follows the benchmark.

        The formula is as follows:

            Tracking Error (TE) = Standard Deviation of (Portfolio Returns - Benchmark Returns)

        See definition: https://en.wikipedia.org/wiki/Tracking_error

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Tracking error values.

        Notes:
        - Daily Tracking Error is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the TE for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_tracking_error()
        ```
        """
        if period == "daily":
            print(
                "Daily Tracking Error is not an option as the standard deviation for 1 day is close to zero. "
                "Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(self, period, within_period=True)
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        tracking_error = performance.get_tracking_error(returns, benchmark_returns)

        tracking_error = tracking_error.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                tracking_error,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return tracking_error

    @handle_errors
    def get_information_ratio(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        The Information Ratio (IR), also known as the Information Coefficient, is a financial
        metric that assesses the risk-adjusted performance of a portfolio or investment
        strategy relative to a benchmark index. It quantifies how much excess return the
        portfolio generates for each unit of tracking error (volatility of tracking error).
        The Information Ratio is commonly used by portfolio managers, financial analysts, and
        investors to evaluate the skill of a portfolio manager in generating returns beyond
        what would be expected based on the risk taken.

        - IR > 0: A positive Information Ratio indicates that the portfolio has generated
        excess returns compared to the benchmark, suggesting that the portfolio manager has added value.
        - IR = 0: An Information Ratio of zero implies that the portfolio's excess return
        is in line with the benchmark, meaning the portfolio manager has not added or lost value relative
        to the benchmark.
        - IR < 0: A negative Information Ratio suggests that the portfolio has underperformed
        the benchmark, potentially indicating that the portfolio manager has detracted value.

        The formula is as follows:

            Information Ratio (IR) = (Portfolio's Excess Return - Benchmark's Excess Return) / Tracking Error

        See definition: https://en.wikipedia.org/wiki/Information_ratio

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Information ratio values.

        Notes:
        - Daily Information Ratio is not an option as the standard deviation for 1 day is close to zero.
        Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates the IR for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_information_ratio()
        ```
        """
        if period == "daily":
            print(
                "Daily Tracking Error is not an option as the standard deviation for 1 day is close to zero. "
                "Therefore, it does not give any useful insights."
            )
            return pd.Series(dtype="object")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = handle_return_data_periods(self, period, within_period=True)
        returns = historical_data.loc[:, "Return"][self._tickers]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        information_ratio = performance.get_information_ratio(
            returns, benchmark_returns
        )

        information_ratio = information_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                information_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return information_ratio
