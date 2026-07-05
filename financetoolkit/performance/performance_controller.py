"""Performance Module"""

__docformat__ = "google"

import warnings

import pandas as pd

from financetoolkit.helpers import calculate_growth, filter_columns, handle_portfolio
from financetoolkit.performance import performance_model
from financetoolkit.performance.helpers import (
    determine_within_dataset,
    determine_within_historical_data,
    handle_errors,
)
from financetoolkit.risk.risk_model import get_max_drawdown, get_ui, get_volatility
from financetoolkit.utilities.logger_model import get_logger

# Runtime errors are ignored on purpose given the nature of the calculations
# sometimes leading to division by zero or other mathematical errors. This is however
# for financial analysis purposes not an issue and should not be considered as a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals

logger = get_logger()


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
        intraday_period: str | None = None,
        progress_bar: bool = True,
    ):
        """
        Initializes the Performance Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the calculations.
            historical_data (dict[str, pd.DataFrame]): The historical data to use for the calculations.
            risk_free_rate_data (pd.DataFrame): The risk free rate data to use for the calculations.
            quarterly (bool | None, optional): Whether to use quarterly data. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculations. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculations. Defaults to None.
            intraday_period (str | None, optional): The intraday period used for within-period calculations.
                Defaults to None.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

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
        self._tickers_without_portfolio = [
            ticker for ticker in tickers if ticker != "Portfolio"
        ]
        self._benchmark_name = "Benchmark"
        self._quarterly: bool | None = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._progress_bar: bool = progress_bar
        self._portfolio_weights: dict | None = None

        # Historical Data
        self._historical_data = historical_data
        self._risk_free_rate_data = risk_free_rate_data

        # Fama and French
        self._fama_and_french_dataset: pd.DataFrame = pd.DataFrame()
        self._fama_and_french_model: pd.DataFrame = pd.DataFrame()
        self._fama_and_french_residuals: pd.DataFrame = pd.DataFrame()
        self._factor_asset_correlations: pd.DataFrame = pd.DataFrame()
        self._factor_correlations: pd.DataFrame = pd.DataFrame()

        # Within Period Calculations
        daily_historical_data = self._historical_data["daily"].copy()

        excess_return = performance_model.get_excess_return(
            daily_historical_data["Return"], self._risk_free_rate_data["daily"]
        )
        excess_return.columns = pd.MultiIndex.from_product(
            [["Excess Return"], excess_return.columns]
        )
        daily_historical_data = pd.concat(
            [daily_historical_data, excess_return], axis=1
        )

        intraday_historical_data = self._historical_data["intraday"].copy()

        daily_historical_data.index = pd.DatetimeIndex(
            daily_historical_data.to_timestamp().index
        )

        if not self._historical_data["intraday"].empty:
            intraday_historical_data.index = pd.DatetimeIndex(
                intraday_historical_data.to_timestamp().index
            )

        self._within_historical_data = determine_within_historical_data(
            daily_historical_data=daily_historical_data,
            intraday_historical_data=intraday_historical_data,
            intraday_period=intraday_period,
        )

        # Risk Free Rate of Intraday Historical Data is set to be equal to the last value of the daily risk free rate
        self._intraday_risk_free_rate_data = pd.Series(
            self._historical_data["daily"].iloc[-1],
            index=self._historical_data["intraday"].index,
        )

    @handle_errors
    def collect_all_metrics(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates and collects all performance metrics.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.Series or pd.DataFrame: Performance metrics calculated based on the specified parameters.

        Notes:
        - The method calculates various performance metrics for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.collect_all_metrics()
        ```

        Which returns:

        |      |   Win Rate |   Upside Capture Ratio |   Downside Capture Ratio |   M2 Ratio |   Tracking Error |
        |:-----|-----------:|-----------------------:|-------------------------:|-----------:|-----------------:|
        | 2021 |     0.4921 |                 1.3754 |                   1.4016 |     1.2868 |           0.0118 |
        | 2022 |     0.4821 |                 1.3044 |                   1.3043 |    -0.8603 |           0.0115 |
        | 2023 |     0.576  |                 1.1783 |                   0.9486 |     2.1832 |           0.009  |
        | 2024 |     0.504  |                 1.1158 |                   1.0337 |     1.1242 |           0.0121 |
        | 2025 |     0.472  |                 1.0162 |                   1.0842 |     0.1354 |           0.0139 |
        | 2026 |     0.504  |                 0.766  |                   0.5691 |     0.3377 |           0.0154 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        rounding = rounding if rounding else self._rounding

        performance_metrics = {
            "Alpha": self.get_alpha(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Beta": self.get_beta(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "CAPM": self.get_capital_asset_pricing_model(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Jensen's Alpha": self.get_jensens_alpha(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Treynor Ratio": self.get_treynor_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Sharpe Ratio": self.get_sharpe_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Sortino Ratio": self.get_sortino_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Ulcer Index": self.get_ulcer_performance_index(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Calmar Ratio": self.get_calmar_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Sterling Ratio": self.get_sterling_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Burke Ratio": self.get_burke_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Omega Ratio": self.get_omega_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Kappa Ratio": self.get_kappa_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Gain to Pain Ratio": self.get_gain_to_pain_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Win Rate": self.get_win_rate(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Upside Capture Ratio": self.get_upside_capture_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Downside Capture Ratio": self.get_downside_capture_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "M2 Ratio": self.get_m2_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Tracking Error": self.get_tracking_error(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
            "Information Ratio": self.get_information_ratio(
                period=period, rounding=rounding, growth=growth, lag=lag
            ),
        }

        if period != "daily":
            performance_metrics["Returns"] = self.get_returns(
                period=period, rounding=rounding, growth=growth, lag=lag
            )
            performance_metrics["Excess Return"] = self.get_excess_return(
                period=period, rounding=rounding, growth=growth, lag=lag
            )

        performance_metrics = pd.concat(performance_metrics, axis=1)

        if len(self._tickers) == 1:
            performance_metrics = performance_metrics.xs(
                self._tickers[0], level=1, axis=1
            )

        return performance_metrics

    @handle_portfolio
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

            - Beta = Covariance of Asset Returns and Benchmark Returns / Variance of Benchmark Returns

        For a given period, for example monthly, this translates into the following:

            - Beta = Monthly Covariance of Asset Returns and Benchmark Returns
            / Monthly Variance of Benchmark Returns

        See definition: https://en.wikipedia.org/wiki/Beta_(finance)

        Also known as: market sensitivity, systematic risk.

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

        toolkit = Toolkit(["AAPL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_beta()
        ```

        Which returns:

        | Date   |   AAPL |   AMZN |
        |:-------|-------:|-------:|
        | 2021   | 1.3093 | 1.0276 |
        | 2022   | 1.2989 | 1.6292 |
        | 2023   | 1.1    | 1.5133 |
        | 2024   | 0.9656 | 1.5442 |
        | 2025   | 1.2485 | 1.3264 |
        | 2026   | 0.7887 | 1.281  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = (
            self._within_historical_data[period]
            if not rolling
            else self._historical_data[period]
        )

        returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        if rolling:
            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
            )
        else:
            beta = performance_model.get_beta(returns, benchmark_returns)

        beta = beta.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        beta = beta.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                beta,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return beta

    @handle_portfolio
    @handle_errors
    def get_capital_asset_pricing_model(
        self,
        period: str | None = None,
        rolling: int | None = None,
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

        - Capital Asset Pricing Model = Risk Free Rate + Beta * (Benchmark Returns — Risk Free Rate)

        See definition: https://en.wikipedia.org/wiki/Capital_asset_pricing_model

        Also known as: CAPM, expected return model.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_capital_asset_pricing_model()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  0.3494 |  0.4914 |
        | 2022   | -0.2646 | -0.3666 |
        | 2023   |  0.2633 |  0.4905 |
        | 2024   |  0.2266 |  0.4924 |
        | 2025   |  0.1938 |  0.3135 |
        | 2026   |  0.0822 |  0.1364 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            historical_data = self._historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
            )
        else:
            historical_data = self._within_historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            beta = performance_model.get_beta(returns, benchmark_returns)

        risk_free_rate = self._risk_free_rate_data[period]
        benchmark_returns = self._historical_data[period].loc[:, "Return"][
            self._benchmark_name
        ]

        capm = performance_model.get_capital_asset_pricing_model(
            risk_free_rate, beta, benchmark_returns
        )

        capm = capm.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        capm = capm.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                capm,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return capm

    @handle_errors
    def get_factor_asset_correlations(
        self,
        period: str | None = None,
        factors_to_calculate: list[str] | None = None,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        Calculates factor exposures for each asset.

        The major difference between the Fama and French Model here is that the correlation
        is taken as opposed to a Linear Regression in which the R-squared or Slope can be used to
        understand the exposure to each factor.

        For assessing the exposure or influence of a stock to external factors, it's often preferable
        to use R-squared (R²) or Beta because it explicitly measures how well the factors explain the stock's
        returns. A higher R² indicates that the stock's returns are more closely related to the factors,
        and thus, the factors have a greater influence on the stock's performance.

        However, since the results are closely related and tend to point into the same direction it could
        be fine to use correlations as well depending on the level of accuracy required.

        Also known as: factor exposure, asset correlations.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: Factor Asset Correlations.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_factor_asset_correlations()
        ```

        Which returns:

        |      |   AAPL |   TSLA |
        |:-----|-------:|-------:|
        | 2021 | 0.6688 | 0.5063 |
        | 2022 | 0.8365 | 0.6432 |
        | 2023 | 0.6986 | 0.5615 |
        | 2024 | 0.518  | 0.4874 |
        | 2025 | 0.7403 | 0.6978 |
        | 2026 | 0.4834 | 0.6055 |
        """

        factors_to_calculate = (
            factors_to_calculate
            if factors_to_calculate
            else ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        )

        for factor in factors_to_calculate:
            if factor not in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                raise ValueError(
                    f"Factor {factor} is not a valid factor. Please select from: Mkt-RF, SMB, HML, RMW, CMA."
                )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data_within = self._within_historical_data[period]
        returns = historical_data_within.loc[:, "Return"][
            self._tickers_without_portfolio
        ]

        if self._fama_and_french_dataset.empty:
            self._fama_and_french_dataset = (
                performance_model.obtain_fama_and_french_dataset()
            )

        fama_and_french_period = determine_within_dataset(
            dataset=self._fama_and_french_dataset, period=period, correlation=False
        )

        merged_df = fama_and_french_period.merge(
            returns, left_index=True, right_index=True
        )

        factor_correlations: dict = {}

        logger.info("Calculating Factor Asset Correlations")
        for ticker in self._tickers_without_portfolio:
            factor_correlations[ticker] = {}
            for dataset_period in merged_df.index.get_level_values(0):
                factor_data = merged_df.loc[dataset_period][factors_to_calculate]
                excess_returns = (
                    merged_df.loc[dataset_period][ticker]
                    - merged_df.loc[dataset_period]["RF"]
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    factor_correlations[ticker][dataset_period] = (
                        performance_model.get_factor_asset_correlations(
                            factors=factor_data, excess_return=excess_returns
                        )
                    )
        factor_asset_correlations = pd.DataFrame.from_dict(
            {
                (ticker, dataset_period): value
                for ticker, factor_scores_ticker in factor_correlations.items()
                for dataset_period, value in factor_scores_ticker.items()
            },
        )

        factor_order = factor_asset_correlations.index

        factor_asset_correlations = (
            factor_asset_correlations.stack(level=1, future_stack=True)
            .unstack(level=0)
            .reindex(factor_order, level=1, axis=1)
            .reindex(self._tickers_without_portfolio, level=0, axis=1)
        )

        self._factor_asset_correlations = factor_asset_correlations.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        return filter_columns(self._factor_asset_correlations, show_columns)

    @handle_errors
    def get_factor_correlations(
        self,
        period: str | None = None,
        factors_to_calculate: list[str] | None = None,
        exclude_risk_free: bool = True,
        rounding: int | None = None,
    ):
        """
        Calculates factor correlations between each factor. This is useful to understand how correlated
        each factor is to each other. This is based off the Fama and French 5 Factor model which includes:

            - Market Risk Premium (Mkt-RF): Represents the additional return that investors expect to earn for taking
            on the risk of investing in the overall market as opposed to a risk-free asset.
            - Size Premium (SMB): Reflects the historical excess return of small-cap stocks over large-cap stocks.
            - Value Premium (HML): Captures the historical excess return of value stocks over growth stocks.
            - Profitability (RMW): Measures the historical excess return of high profitability stocks over
            low profitability stocks.
            - Investment (CMA): Quantifies the historical excess return of low investment stocks over
            high investment stocks.

        Optionally, it is also possible to see the correlation between the risk-free rate and each factor.

        Also known as: factor model correlations.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            exclude_risk_free (bool, optional): Whether to exclude the risk-free rate from the results. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: Factor Correlations.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_factor_correlations()
        ```

        Which returns:

        |        |   Mkt-RF |     SMB |     HML |     RMW |     CMA |
        |:-------|---------:|--------:|--------:|--------:|--------:|
        | Mkt-RF |   1      | -0.4121 |  0.332  |  0.014  | -0.4682 |
        | SMB    |  -0.4121 |  1      | -0.1718 | -0.2326 |  0.1379 |
        | HML    |   0.332  | -0.1718 |  1      | -0.4551 |  0.184  |
        | RMW    |   0.014  | -0.2326 | -0.4551 |  1      | -0.4106 |
        | CMA    |  -0.4682 |  0.1379 |  0.184  | -0.4106 |  1      |
        """
        factors_to_calculate = (
            factors_to_calculate
            if factors_to_calculate
            else ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        )

        for factor in factors_to_calculate:
            if factor not in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                raise ValueError(
                    f"Factor {factor} is not a valid factor. Please select from: Mkt-RF, SMB, HML, RMW, CMA."
                )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        factors_to_calculate = (
            factors_to_calculate + ["RF"]
            if not exclude_risk_free
            else factors_to_calculate
        )

        if self._fama_and_french_dataset.empty:
            self._fama_and_french_dataset = (
                performance_model.obtain_fama_and_french_dataset()
            )

        fama_and_french_period = determine_within_dataset(
            self._fama_and_french_dataset[factors_to_calculate],
            period,
            correlation=True,
        )

        self._factor_correlations = fama_and_french_period.round(
            rounding if rounding else self._rounding
        )

        return self._factor_correlations

    @handle_errors
    def get_fama_and_french_model(
        self,
        period: str | None = None,
        method: str = "multi",
        factors_to_calculate: list[str] | None = None,
        include_daily_residuals: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        show_columns: list[str] | None = None,
    ):
        """
        Calculate Fama and French 5 Factor model scores and residuals for a set of financial assets.

        The Fama and French 5 Factor model is a widely used financial model that helps estimate the expected return
        of financial assets, such as stocks or portfolios, based on five key factors:

            - Market Risk Premium (Mkt-RF): Represents the additional return that investors expect to earn for taking
            on the risk of investing in the overall market as opposed to a risk-free asset.
            - Size Premium (SMB): Reflects the historical excess return of small-cap stocks over large-cap stocks.
            - Value Premium (HML): Captures the historical excess return of value stocks over growth stocks.
            - Profitability (RMW): Measures the historical excess return of high profitability stocks over
            low profitability stocks.
            - Investment (CMA): Quantifies the historical excess return of low investment stocks over
            high investment stocks.

        The model can perform both a Simple Linear Regression on each factor as well as a Multi Linear Regression
        which includes all factors. Generally, a multi linear regression is applied but if you wish to see individual
        R-squared values for each factor you can select the simple linear regression method.

        The model performs a Linear Regression on each factor and defines the regression parameters and residuals
        for each asset over time based on its exposure to these factors.

        These results can be validated by comparing them to the period returns obtained from the historical data. E.g.
        the regression formula is as follows for the Multi Linear Regression:

            - Excess Return = Intercept + Beta1 * Mkt-RF + Beta2 * SMB + Beta3 * HML + Beta4 * RMW
                + Beta5 * CMA + Residuals

        And the following for the Simple Linear Regression:

            - Excess Return = Intercept + Slope * Factor Value + Residuals

        So for a given factor, it should hold that the Excess Return equals the entire regression. Note that in this
        calculation the Excess Return refers to the Asset Return minus the Risk Free Rate as reported in the Fama and
        French dataset and will not be the same as the defined Excess Return in the historical data given that this is
        based on the Risk Free Rate defined in the initialization.

        What is relevant to look at is the influence these factors have on each stock and how much each factor explains
        the stock return. E.g. you will generally see a pretty high influence (Beta or Slope) for the Market Risk Premium
        (Mkt-RF) factor as this is the main factor that explains the stock return (as also prevalent in the CAPM).
        The other factors can fluctuate greatly between stocks depending on which stocks you look at.

        Also known as: Fama-French model, three-factor model, five-factor model, FF3, FF5.

        Args:
            period (str, optional): The period for the calculation (e.g., "weekly", "monthly", "quarterly", "yearly").
                Defaults to None, using class-defined quarterly or yearly period.
            method (str, optional): The regression method to use for the calculation. Defaults to 'multi'.
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            include_residuals (bool, optional): Whether to include residuals in the results. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio values. Defaults to False.
            lag (int or list of int, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Fama and French 5 Factor model scores for the specified assets.

        Notes:
        - The dataset from Fama and French is not always fully up to date. Therefore, some periods could be excluded.
        - Daily Fama and French results is not an option as it would attempt to do a linear regression on a
        single data point which will not give any meaningful results.
        - The method retrieves historical data and calculates regression parameters and residuals for each asset.
        - The risk-free rate is typically represented by the return of a risk-free investment, such as a Treasury bond.
        In this case, the Risk Free Rate from the Fama and French dataset is used.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # Calculate Fama and French 5 Factor model scores
        toolkit.performance.get_fama_and_french_model()
        ```

        Which returns:

        |      |    AAPL |    TSLA |
        |:-----|--------:|--------:|
        | 2021 | -0.0051 | -0.0195 |
        | 2022 | -0.0196 | -0.02   |
        | 2023 | -0.013  |  0.0045 |
        | 2024 |  0.0022 | -0.0059 |
        | 2025 | -0.0191 | -0.0191 |
        | 2026 | -0.0204 | -0.0153 |
        """
        if method not in ["simple", "multi"]:
            raise ValueError(
                f"Method {method} is not a valid method. Please select from: simple or multi (default). "
                "This refers to a simple linear regression (regression of each factor on each asset return series) "
                "or a multi linear regression (regression of all factors on each asset return series) respectively"
            )

        factors_to_calculate = (
            factors_to_calculate
            if factors_to_calculate
            else ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        )

        for factor in factors_to_calculate:
            if factor not in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                raise ValueError(
                    f"Factor {factor} is not a valid factor. Please select from: Mkt-RF, SMB, HML, RMW, CMA."
                )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data_within = self._within_historical_data[period]
        returns = historical_data_within.loc[:, "Return"][
            self._tickers_without_portfolio
        ]

        historical_data = self._historical_data[period]
        returns_total = historical_data.loc[:, "Return"][
            self._tickers_without_portfolio
        ]

        self._fama_and_french_dataset = (
            performance_model.obtain_fama_and_french_dataset()
        )
        fama_and_french_period = determine_within_dataset(
            self._fama_and_french_dataset, period, correlation=False
        )

        merged_df = fama_and_french_period.merge(
            returns, left_index=True, right_index=True
        )
        factor_scores: dict = {}
        daily_residuals: dict = {}

        logger.info(
            "Calculating %s Factor Exposures",
            "Multi" if method == "multi" else "Individual",
        )
        for ticker in self._tickers_without_portfolio:
            factor_scores[ticker] = {}
            daily_residuals[ticker] = {}

            if method == "multi":
                for dataset_period in merged_df.index.get_level_values(0):
                    factor_data = merged_df.loc[dataset_period][factors_to_calculate]
                    excess_returns = (
                        merged_df.loc[dataset_period][ticker]
                        - merged_df.loc[dataset_period]["RF"]
                    )

                    (
                        factor_scores[ticker][dataset_period],
                        daily_residuals[ticker][dataset_period],
                        error_message,
                    ) = performance_model.get_fama_and_french_model_multi(
                        excess_returns=excess_returns, factor_dataset=factor_data
                    )

                    if error_message:
                        logger.warning(
                            "%s for %s in %s.", error_message, ticker, dataset_period
                        )

                    fama_and_french_model = pd.DataFrame.from_dict(
                        {
                            (ticker, factor): value
                            for ticker, factor_scores_ticker in factor_scores.items()
                            for factor, value in factor_scores_ticker.items()
                        },
                        orient="index",
                    )

                fama_and_french_model = fama_and_french_model.unstack(
                    level=0, sort=False
                ).swaplevel(0, 1, axis=1)

                # Sort the DataFrame with respect to the original column order
                tickers_column_order = fama_and_french_model.columns.get_level_values(
                    0
                ).unique()
                parameters_column_order = (
                    fama_and_french_model.columns.get_level_values(1).unique()
                )

                fama_and_french_model = (
                    fama_and_french_model.sort_index(axis=1)
                    .reindex(tickers_column_order, level=0, axis=1)
                    .reindex(parameters_column_order, level=1, axis=1)
                )

            elif method == "simple":
                for factor in factors_to_calculate:
                    factor_scores[ticker][factor] = {}
                    daily_residuals[ticker][factor] = {}
                    for dataset_period in merged_df.index.get_level_values(0):
                        factor_data = merged_df.loc[dataset_period][factor]
                        excess_returns = (
                            merged_df.loc[dataset_period][ticker]
                            - merged_df.loc[dataset_period]["RF"]
                        )

                        (
                            factor_scores[ticker][factor][dataset_period],
                            daily_residuals[ticker][factor][dataset_period],
                        ) = performance_model.get_fama_and_french_model_single(
                            excess_returns=excess_returns, factor=factor_data
                        )

                        factor_scores[ticker][factor][dataset_period][
                            "Factor Value"
                        ] = factor_data.iloc[-1]

                        factor_scores[ticker][factor][dataset_period][
                            "Residuals"
                        ] = returns_total.loc[dataset_period][ticker] - (
                            factor_scores[ticker][factor][dataset_period]["Slope"]
                            * factor_data.iloc[-1]
                            + factor_scores[ticker][factor][dataset_period]["Intercept"]
                        )

                fama_and_french_model = pd.DataFrame.from_dict(
                    {
                        (period, factor, ticker): value
                        for ticker, factor_scores_ticker in factor_scores.items()
                        for factor, factor_scores_factor in factor_scores_ticker.items()
                        for period, value in factor_scores_factor.items()
                    },
                    orient="index",
                )

                # Sort the DataFrame with respect to the original column order
                parameters_column_order = fama_and_french_model.columns.unique()
                factor_column_order = fama_and_french_model.index.get_level_values(
                    1
                ).unique()
                ticker_column_order = fama_and_french_model.index.get_level_values(
                    2
                ).unique()

                fama_and_french_model = fama_and_french_model.stack().unstack(
                    level=[2, 1, 3]
                )

                fama_and_french_model = (
                    fama_and_french_model.sort_index(axis=1)
                    .reindex(parameters_column_order, level=2, axis=1)
                    .reindex(factor_column_order, level=1, axis=1)
                    .reindex(ticker_column_order, level=0, axis=1)
                )

        self._fama_and_french_model = fama_and_french_model.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if include_daily_residuals:
            if method == "multi":
                daily_residuals_df = pd.DataFrame.from_dict(
                    {
                        (ticker, factor): value
                        for ticker, residuals_ticker in daily_residuals.items()
                        for factor, value in residuals_ticker.items()
                    },
                    orient="index",
                )

                daily_residuals_df = (
                    daily_residuals_df.stack()
                    .unstack(level=0)
                    .sort_index(axis=1, sort_remaining=False)
                )
                daily_residuals_df = daily_residuals_df.reset_index(level=0, drop=True)
            else:
                daily_residuals_df = pd.DataFrame.from_dict(
                    {
                        (period, factor, ticker): value
                        for ticker, residuals_ticker in daily_residuals.items()
                        for factor, residuals_factor in residuals_ticker.items()
                        for period, value in residuals_factor.items()
                    },
                    orient="index",
                )

                daily_residuals_df = (
                    daily_residuals_df.unstack(level=[1, 2])
                    .stack(level=0)
                    .sort_index(axis=1, sort_remaining=False)
                )

                daily_residuals_df = daily_residuals_df.reset_index(level=0, drop=True)

            self._fama_and_french_residuals = daily_residuals_df.round(
                rounding if rounding else self._rounding
            ).loc[self._start_date : self._end_date]

            return (
                filter_columns(self._fama_and_french_model, show_columns),
                self._fama_and_french_residuals,
            )

        if growth:
            return filter_columns(
                calculate_growth(
                    self._fama_and_french_model,
                    lag=lag,
                    rounding=rounding if rounding else self._rounding,
                    axis="index",
                ),
                show_columns,
            )

        return filter_columns(self._fama_and_french_model, show_columns)

    @handle_portfolio
    @handle_errors
    def get_alpha(
        self,
        period: str | None = None,
        show_full_results: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Alpha, in a general sense, represents the excess return an investment generates relative to
        a benchmark or a risk-adjusted return. It can be positive (indicating the investment
        outperformed the benchmark) or negative (indicating underperformance).

        The formula is as follows:

            - Alpha = Asset's Actual Return — Benchmark's Actual Return

        See definition: https://en.wikipedia.org/wiki/Alpha_(finance)

        Also known as: excess return, outperformance, active return.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Alpha is calculated as the rolling mean excess return over this many periods across
            the full return history instead of per `period`. Defaults to None.
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_alpha()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  0.0678 |  0.2272 |
        | 2022   | -0.0735 | -0.4555 |
        | 2023   |  0.2389 |  0.7743 |
        | 2024   |  0.0677 |  0.3922 |
        | 2025   | -0.0779 | -0.0499 |
        | 2026   |  0.0431 | -0.2173 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = self._historical_data[period]
        returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        if rolling:
            alpha = performance_model.get_rolling_alpha(
                returns, benchmark_returns, rolling
            )
        else:
            alpha = performance_model.get_alpha(returns, benchmark_returns)

        alpha = alpha.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        alpha = alpha.dropna(how="all", axis=0)

        if show_full_results:
            full_results = pd.concat([returns, benchmark_returns, alpha], axis=1).loc[
                self._start_date : self._end_date
            ]
            full_results.columns = (
                [
                    f"Actual Return {ticker}"
                    for ticker in self._tickers_without_portfolio
                ]
                + ["Benchmark Returns"]
                + [f"Alpha {ticker}" for ticker in self._tickers_without_portfolio]
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

    @handle_portfolio
    @handle_errors
    def get_jensens_alpha(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate Jensen's Alpha, a measure of an asset's performance relative to its expected return
        based on the Capital Asset Pricing Model (CAPM).

        Jensen's Alpha is used to assess whether an investment has outperformed or underperformed
        its expected return given its systematic risk, as represented by the asset's Beta.

        The formula is as follows:

            - Jensen's Alpha = Asset's Actual Return — [Risk-Free Rate + Beta * (Benchmark Return — Risk-Free Rate)]

        See definition: https://en.wikipedia.org/wiki/Jensen%27s_alpha

        Also known as: Jensen alpha, risk-adjusted excess return.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_jensens_alpha()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   | -0.0112 |  0.0062 |
        | 2022   | -0.0037 | -0.2837 |
        | 2023   |  0.2185 |  0.5267 |
        | 2024   |  0.0741 |  0.1328 |
        | 2025   | -0.1082 | -0.1999 |
        | 2026   |  0.0531 | -0.2615 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            historical_data = self._historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
            )
        else:
            historical_within_data = self._within_historical_data[period]
            returns = historical_within_data.loc[:, "Return"][
                self._tickers_without_portfolio
            ]
            benchmark_returns = historical_within_data.loc[:, "Return"][
                self._benchmark_name
            ]

            beta = performance_model.get_beta(returns, benchmark_returns)

        historical_data = self._historical_data[period]

        period_returns = historical_data.loc[:, "Return"][
            self._tickers_without_portfolio
        ]

        risk_free_rate = self._risk_free_rate_data[period]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        jensens_alpha = performance_model.get_jensens_alpha(
            period_returns, risk_free_rate, beta, benchmark_returns
        )

        jensens_alpha = jensens_alpha.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        jensens_alpha = jensens_alpha.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                jensens_alpha,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return jensens_alpha

    @handle_portfolio
    @handle_errors
    def get_treynor_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
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

        - Treynor Ratio = (Portfolio's Return — Risk-Free Rate) / Portfolio Beta

        See definition: https://en.wikipedia.org/wiki/Treynor_ratio

        Also known as: reward-to-volatility ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_treynor_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  0.2468 |  0.2586 |
        | 2022   | -0.2364 | -0.3971 |
        | 2023   |  0.4028 |  0.4422 |
        | 2024   |  0.2641 |  0.243  |
        | 2025   |  0.0352 |  0.0323 |
        | 2026   |  0.1146 | -0.0879 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            historical_data = self._historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
            )
        else:
            historical_within_data = self._within_historical_data[period]
            returns = historical_within_data.loc[:, "Return"][
                self._tickers_without_portfolio
            ]
            benchmark_returns = historical_within_data.loc[:, "Return"][
                self._benchmark_name
            ]

            beta = performance_model.get_beta(returns, benchmark_returns)

        historical_data = self._historical_data[period]

        period_returns = historical_data.loc[:, "Return"][
            self._tickers_without_portfolio
        ]
        risk_free_rate = self._risk_free_rate_data[period]

        treynor_ratio = performance_model.get_treynor_ratio(
            period_returns, risk_free_rate, beta
        )

        treynor_ratio = treynor_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        treynor_ratio = treynor_ratio.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                treynor_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return treynor_ratio

    @handle_portfolio
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

            - Sharpe Ratio = Excess Return / Excess Standard Deviation

        For a given period, for example monthly, this translates into the following:

            - Sharpe Ratio = Average Monthly Excess Return / Standard Deviation of Monthly Excess Returns

        For a rolling period, this translates into the following:

            - Sharpe Ratio = Average Rolling Excess Return / Standard Deviation of Rolling Excess Returns

        Note that this is explicitly already subtracts the Risk Free Rate.

        See definition: https://en.wikipedia.org/wiki/Sharpe_ratio

        Also known as: risk-adjusted return, reward-to-variability ratio.

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_sharpe_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   | -0.8286 | -0.3537 |
        | 2022   | -1.2859 | -0.7606 |
        | 2023   | -2.7296 | -1.0402 |
        | 2024   | -2.8575 | -0.9845 |
        | 2025   | -2.0637 | -1.0411 |
        | 2026   | -2.4952 | -1.6057 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            period_returns = self._historical_data[period].loc[:, "Return"][
                self._tickers_without_portfolio
            ]
            excess_return = performance_model.get_excess_return(
                period_returns, self._risk_free_rate_data[period]
            )
            sharpe_ratio = performance_model.get_rolling_sharpe_ratio(
                excess_return, rolling
            )
        else:
            excess_return = self._within_historical_data[period].loc[
                :, "Excess Return"
            ][self._tickers_without_portfolio]
            sharpe_ratio = performance_model.get_sharpe_ratio(excess_return)

        sharpe_ratio = sharpe_ratio.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        sharpe_ratio = sharpe_ratio.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                sharpe_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return sharpe_ratio

    @handle_portfolio
    @handle_errors
    def get_sortino_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
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

            - Sortino Ratio = Excess Return / Excess Downside Risk

        For a given period, for example monthly, this translates into the following:

            - Sortino Ratio = Average Monthly Excess Return / Average Monthly Excess Downside Risk

        For a rolling period, this translates into the following:

            - Sortino Ratio = Average Rolling Excess Return / Rolling Downside Risk

        Note that this is explicitly already subtracts the Risk Free Rate.

        See definition: https://en.wikipedia.org/wiki/Sortino_ratio

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the Sortino ratio is calculated over a rolling window of this many periods across the
            full return history instead of per `period`. Defaults to None.
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_sortino_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   | -1.0988 | -0.5282 |
        | 2022   | -1.5168 | -0.9959 |
        | 2023   | -2.8934 | -1.3591 |
        | 2024   | -3.097  | -1.3744 |
        | 2025   | -2.4472 | -1.3183 |
        | 2026   | -2.5624 | -1.7784 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            period_returns = self._historical_data[period].loc[:, "Return"][
                self._tickers_without_portfolio
            ]
            excess_return = performance_model.get_excess_return(
                period_returns, self._risk_free_rate_data[period]
            )
            sortino_ratio = performance_model.get_rolling_sortino_ratio(
                excess_return, rolling
            )
        else:
            historical_data = self._within_historical_data[period]
            excess_return = historical_data.loc[:, "Excess Return"][
                self._tickers_without_portfolio
            ]

            sortino_ratio = performance_model.get_sortino_ratio(excess_return)

        sortino_ratio = sortino_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        sortino_ratio = sortino_ratio.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                sortino_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return sortino_ratio

    @handle_portfolio
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
        Similar to the Sharpe Ratio, a higher UPI is better than a lower one (since investors prefer more return
        for less risk).

        Also known as: UPI, Martin ratio.

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_ulcer_performance_index()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  8.5991 |  5.6729 |
        | 2022   | -4.5711 | -5.0182 |
        | 2023   | 13.3465 | 11.6618 |
        | 2024   |  7.4872 |  6.3795 |
        | 2025   |  0.8946 |  0.7159 |
        | 2026   |  2.3126 | -2.6591 |
        """

        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = self._within_historical_data[period]
        returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]

        period_returns = self._historical_data[period].loc[:, "Return"][
            self._tickers_without_portfolio
        ]
        excess_return = (
            period_returns
            if period == "intraday"
            else performance_model.get_excess_return(
                period_returns, self._risk_free_rate_data[period]
            )
        )

        ulcer_index = get_ui(returns, rolling)

        ulcer_performance_index = performance_model.get_ulcer_performance_index(
            excess_return, ulcer_index
        )
        ulcer_performance_index = ulcer_performance_index.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        ulcer_performance_index = ulcer_performance_index.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                ulcer_performance_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return ulcer_performance_index

    @handle_portfolio
    @handle_errors
    def get_calmar_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Calmar Ratio of an investment portfolio or asset's returns.

        The Calmar Ratio is a risk-adjusted return metric that divides the (annualized) return
        of an investment portfolio or asset by its Maximum Drawdown, providing insight into the
        return achieved per unit of the worst historical loss of value.

        The formula is as follows:

        - Calmar Ratio = Return / |Maximum Drawdown|

        See definition: https://en.wikipedia.org/wiki/Calmar_ratio

        Also known as: Drawdown ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            within_period (bool, optional): Whether to calculate the Maximum Drawdown within the
            specified period or for the entire period. Thus whether to look at the Maximum Drawdown
            within a specific year (if period = 'yearly') or look at the entirety of all years.
            Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Calmar Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Calmar Ratio for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_calmar_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  1.8052 |  1.3727 |
        | 2022   | -0.8734 | -0.8942 |
        | 2023   |  3.1929 |  3.1074 |
        | 2024   |  1.9404 |  1.461  |
        | 2025   |  0.2834 |  0.2356 |
        | 2026   |  1.0648 | -0.5212 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        ).loc[:, "Return"][self._tickers_without_portfolio]

        maximum_drawdown = get_max_drawdown(returns)

        period_returns = self._historical_data[period].loc[:, "Return"][
            self._tickers_without_portfolio
        ]

        calmar_ratio = performance_model.get_calmar_ratio(
            period_returns, maximum_drawdown
        )

        calmar_ratio = calmar_ratio.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        calmar_ratio = calmar_ratio.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                calmar_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return calmar_ratio

    @handle_portfolio
    @handle_errors
    def get_sterling_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        adjustment: float = 0.1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Sterling Ratio of an investment portfolio or asset's returns.

        The Sterling Ratio is a risk-adjusted return metric that divides the (annualized) return
        of an investment portfolio or asset by its Average Drawdown plus a fixed adjustment
        (conventionally 10%), providing insight into the return achieved relative to the typical
        depth of its drawdowns rather than only the single worst one (as with the Calmar Ratio).

        The formula is as follows:

        - Sterling Ratio = Return / (|Average Drawdown| + Adjustment)

        Also known as: Sterling-Calmar ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            within_period (bool, optional): Whether to calculate the Average Drawdown within the
            specified period or for the entire period. Thus whether to look at the Average Drawdown
            within a specific year (if period = 'yearly') or look at the entirety of all years.
            Defaults to True.
            adjustment (float, optional): The fixed adjustment added to the Average Drawdown,
            conventionally 0.1 (10%). Defaults to 0.1.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Sterling Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Sterling Ratio for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_sterling_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  2.0094 |  1.7258 |
        | 2022   | -1.074  | -1.4667 |
        | 2023   |  3.2891 |  4.193  |
        | 2024   |  1.9286 |  2.1544 |
        | 2025   |  0.4371 |  0.3785 |
        | 2026   |  0.886  | -0.6097 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        ).loc[:, "Return"][self._tickers_without_portfolio]

        average_drawdown = performance_model.get_average_drawdown(returns)

        period_returns = self._historical_data[period].loc[:, "Return"][
            self._tickers_without_portfolio
        ]

        sterling_ratio = performance_model.get_sterling_ratio(
            period_returns, average_drawdown, adjustment
        )

        sterling_ratio = sterling_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        sterling_ratio = sterling_ratio.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                sterling_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return sterling_ratio

    @handle_portfolio
    @handle_errors
    def get_burke_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Burke Ratio of an investment portfolio or asset's returns.

        The Burke Ratio is a risk-adjusted return metric that divides the excess return (return
        minus the risk-free rate) of an investment portfolio or asset by the square root of the
        sum of its squared drawdowns, penalizing both the frequency and depth of drawdowns more
        heavily than the Calmar or Sterling Ratios.

        The formula is as follows:

        - Burke Ratio = (Return — Risk-Free Rate) / SQRT(SUM(Drawdowns^2))

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            within_period (bool, optional): Whether to calculate the drawdowns within the specified
            period or for the entire period. Thus whether to look at the drawdowns within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Burke Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Burke Ratio for each asset in
        the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as
        a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_burke_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  0.2606 |  0.1513 |
        | 2022   | -0.1172 | -0.1172 |
        | 2023   |  0.502  |  0.4155 |
        | 2024   |  0.2519 |  0.1765 |
        | 2025   |  0.0246 |  0.0195 |
        | 2026   |  0.1417 | -0.1284 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        ).loc[:, "Return"][self._tickers_without_portfolio]

        burke_drawdown_measure = performance_model.get_burke_drawdown_measure(returns)

        period_returns = self._historical_data[period].loc[:, "Return"][
            self._tickers_without_portfolio
        ]
        excess_return = performance_model.get_excess_return(
            period_returns, self._risk_free_rate_data[period]
        )

        burke_ratio = performance_model.get_burke_ratio(
            excess_return, burke_drawdown_measure
        )

        burke_ratio = burke_ratio.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        burke_ratio = burke_ratio.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                burke_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return burke_ratio

    @handle_portfolio
    @handle_errors
    def get_m2_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
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

            - M2 Ratio = (Portfolio's Return — Risk-Free Rate) / Portfolio Standard Deviation

        See definition: https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance

        Also known as: Modigliani-Modigliani measure, M2, risk-adjusted performance.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the M2 ratio is calculated over a rolling window of this many periods across the full
            return history instead of per `period`. Defaults to None.
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_m2_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  1.2868 |  0.8811 |
        | 2022   | -0.8603 | -1.0335 |
        | 2023   |  2.1832 |  1.8102 |
        | 2024   |  1.1242 |  0.9127 |
        | 2025   |  0.1354 |  0.1134 |
        | 2026   |  0.3377 | -0.3943 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_period_data = self._historical_data[period]
        period_returns = historical_period_data.loc[:, "Return"][
            self._tickers_without_portfolio
        ]
        risk_free_rate = self._risk_free_rate_data[period]

        if rolling:
            m2_ratio = performance_model.get_rolling_m2_ratio(
                period_returns, risk_free_rate, rolling
            )
        else:
            daily_returns = self._historical_data["daily"].loc[:, "Return"][
                self._tickers_without_portfolio
            ]
            period_standard_deviation = get_volatility(daily_returns, period)

            m2_ratio = performance_model.get_m2_ratio(
                period_returns, risk_free_rate, period_standard_deviation
            )

        m2_ratio = m2_ratio.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        m2_ratio = m2_ratio.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                m2_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return m2_ratio

    @handle_portfolio
    @handle_errors
    def get_tracking_error(
        self,
        period: str | None = None,
        rolling: int | None = None,
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

            - Tracking Error (TE) = Standard Deviation of (Portfolio Returns — Benchmark Returns)

        See definition: https://en.wikipedia.org/wiki/Tracking_error

        Also known as: active risk, benchmark deviation.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Tracking Error is calculated over a rolling window of this many periods across the
            full return history instead of per `period`. Defaults to None.
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_tracking_error()
        ```

        Which returns:

        | Date   |   AAPL |   TSLA |
        |:-------|-------:|-------:|
        | 2021   | 0.0118 | 0.0317 |
        | 2022   | 0.0115 | 0.0344 |
        | 2023   | 0.009  | 0.0304 |
        | 2024   | 0.0121 | 0.0369 |
        | 2025   | 0.0139 | 0.0328 |
        | 2026   | 0.0154 | 0.0226 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            historical_data = self._historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            tracking_error = performance_model.get_rolling_tracking_error(
                returns, benchmark_returns, rolling
            )
        else:
            historical_data = self._within_historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            tracking_error = performance_model.get_tracking_error(
                returns, benchmark_returns
            )

        tracking_error = tracking_error.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        tracking_error = tracking_error.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                tracking_error,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return tracking_error

    @handle_portfolio
    @handle_errors
    def get_information_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
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

            - Information Ratio (IR) = (Portfolio's Excess Return — Benchmark's Excess Return) / Tracking Error

        See definition: https://en.wikipedia.org/wiki/Information_ratio

        Also known as: active return per risk.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the Information Ratio is calculated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_information_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   |  0.0253 |  0.0381 |
        | 2022   | -0.0212 | -0.0739 |
        | 2023   |  0.0833 |  0.0817 |
        | 2024   |  0.0231 |  0.0499 |
        | 2025   | -0.0106 |  0.0164 |
        | 2026   |  0.0269 | -0.0641 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            historical_data = self._historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            information_ratio = performance_model.get_rolling_information_ratio(
                returns, benchmark_returns, rolling
            )
        else:
            historical_data = self._within_historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            information_ratio = performance_model.get_information_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_upside_capture_ratio(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Upside Capture Ratio of an investment portfolio or asset's returns.

        The Upside Capture Ratio measures how well an investment portfolio or asset performs
        relative to a benchmark during periods in which the benchmark's return is positive. A
        ratio above 1 (or 100%) indicates the asset captured more of the benchmark's gains than
        the benchmark itself.

        The formula is as follows:

        - Upside Capture Ratio = Average Return in Up Periods / Average Benchmark Return in Up Periods

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Upside Capture Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Upside Capture Ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_upside_capture_ratio()
        ```

        Which returns:

        | Date   |   AAPL |   TSLA |
        |:-------|-------:|-------:|
        | 2021   | 1.3754 | 2.1079 |
        | 2022   | 1.3044 | 1.7261 |
        | 2023   | 1.1783 | 2.3099 |
        | 2024   | 1.1158 | 2.5976 |
        | 2025   | 1.0162 | 2.2065 |
        | 2026   | 0.766  | 1.593  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = self._within_historical_data[period]
        returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        upside_capture_ratio = performance_model.get_upside_capture_ratio(
            returns, benchmark_returns
        )

        upside_capture_ratio = upside_capture_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                upside_capture_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return upside_capture_ratio

    @handle_portfolio
    @handle_errors
    def get_downside_capture_ratio(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Downside Capture Ratio of an investment portfolio or asset's returns.

        The Downside Capture Ratio measures how well an investment portfolio or asset performs
        relative to a benchmark during periods in which the benchmark's return is negative. A
        ratio below 1 (or 100%) indicates the asset lost less than the benchmark during those
        periods.

        The formula is as follows:

        - Downside Capture Ratio = Average Return in Down Periods / Average Benchmark Return in Down Periods

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Downside Capture Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Downside Capture Ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_downside_capture_ratio()
        ```

        Which returns:

        | Date   |   AAPL |   TSLA |
        |:-------|-------:|-------:|
        | 2021   | 1.4016 | 2.0639 |
        | 2022   | 1.3043 | 2.0238 |
        | 2023   | 0.9486 | 1.8386 |
        | 2024   | 1.0337 | 2.4414 |
        | 2025   | 1.0842 | 2.2603 |
        | 2026   | 0.5691 | 2.2236 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = self._within_historical_data[period]
        returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        downside_capture_ratio = performance_model.get_downside_capture_ratio(
            returns, benchmark_returns
        )

        downside_capture_ratio = downside_capture_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                downside_capture_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return downside_capture_ratio

    @handle_portfolio
    @handle_errors
    def get_win_rate(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Win Rate of an investment portfolio or asset's returns.

        The Win Rate is the percentage of periods in which the asset's return exceeds the
        benchmark's return.

        Also known as: batting average.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Win Rate values.

        Notes:
        - The method retrieves historical data and calculates the Win Rate for each asset in the
        Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_win_rate()
        ```

        Which returns:

        | Date   |   AAPL |   TSLA |
        |:-------|-------:|-------:|
        | 2021   | 0.4921 | 0.5    |
        | 2022   | 0.4821 | 0.498  |
        | 2023   | 0.576  | 0.532  |
        | 2024   | 0.504  | 0.4683 |
        | 2025   | 0.472  | 0.468  |
        | 2026   | 0.504  | 0.472  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = self._within_historical_data[period]
        returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

        win_rate = performance_model.get_win_rate(returns, benchmark_returns)

        win_rate = win_rate.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        if growth:
            return calculate_growth(
                win_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return win_rate

    @handle_portfolio
    @handle_errors
    def get_kappa_ratio(
        self,
        period: str | None = None,
        order: int = 3,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Kappa Ratio of an investment portfolio or asset's returns.

        The Kappa Ratio is a generalization of the Sortino Ratio that penalizes downside risk
        using a higher-order lower partial moment. The Sortino Ratio is the special case of the
        Kappa Ratio with order=2.

        Note that this already subtracts the Risk Free Rate.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            order (int, optional): The order of the lower partial moment used in the denominator.
            Defaults to 3.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Kappa Ratio values.

        Notes:
        - Daily Kappa Ratio is not an option as the standard deviation for 1 day is close to zero.
        Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates the Kappa Ratio for each asset in
        the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as
        a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_kappa_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021   | -0.5191 | -0.2779 |
        | 2022   | -0.6896 | -0.4956 |
        | 2023   | -0.8901 | -0.6046 |
        | 2024   | -0.9024 | -0.6129 |
        | 2025   | -0.8423 | -0.6148 |
        | 2026   | -0.8718 | -0.7635 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = self._within_historical_data[period]
        excess_return = historical_data.loc[:, "Excess Return"][
            self._tickers_without_portfolio
        ]

        kappa_ratio = performance_model.get_kappa_ratio(excess_return, order)

        kappa_ratio = kappa_ratio.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        kappa_ratio = kappa_ratio.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                kappa_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return kappa_ratio

    @handle_portfolio
    @handle_errors
    def get_omega_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rolling: int | None = None,
        minimum_acceptable_return: float = 0.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Omega Ratio of an investment portfolio or asset's returns.

        The Omega Ratio is a risk-return measure that divides the sum of gains above a minimum
        acceptable return (MAR) by the sum of losses below it, capturing the full shape of the
        return distribution rather than only its first two moments (unlike the Sharpe Ratio).

        The formula is as follows:

        - Omega Ratio = SUM(Gains above MAR) / SUM(Losses below MAR)

        See definition: https://en.wikipedia.org/wiki/Omega_ratio

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            within_period (bool, optional): Whether to calculate the Omega Ratio within the specified
            period or for the entire period. Thus whether to look at the Omega Ratio within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
            Omega Ratio is calculated over a rolling window of this many periods across the full
            return history instead of per `period`. Defaults to None.
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used as
            the threshold between gains and losses. Defaults to 0.0.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Omega Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Omega Ratio for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_omega_ratio()
        ```

        Which returns:

        |      |   AAPL |   TSLA |
        |:-----|-------:|-------:|
        | 2021 | 1.2354 | 1.1945 |
        | 2022 | 0.892  | 0.8129 |
        | 2023 | 1.4034 | 1.3043 |
        | 2024 | 1.2462 | 1.2098 |
        | 2025 | 1.0873 | 1.0871 |
        | 2026 | 1.2062 | 0.9358 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            returns = self._historical_data[period].loc[:, "Return"][
                self._tickers_without_portfolio
            ]
            omega_ratio = performance_model.get_rolling_omega_ratio(
                returns, rolling, minimum_acceptable_return
            )
        else:
            returns = (
                self._within_historical_data[period]
                if within_period
                else self._historical_data[period]
            ).loc[:, "Return"][self._tickers_without_portfolio]

            omega_ratio = performance_model.get_omega_ratio(
                returns, minimum_acceptable_return
            )

        omega_ratio = omega_ratio.round(rounding if rounding else self._rounding).loc[
            self._start_date : self._end_date
        ]

        if growth:
            return calculate_growth(
                omega_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return omega_ratio

    @handle_portfolio
    @handle_errors
    def get_gain_to_pain_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Gain-to-Pain Ratio of an investment portfolio or asset's returns.

        The Gain-to-Pain Ratio, popularized by Jack Schwager, divides the sum of all returns by
        the sum of the absolute value of all losses, summarizing the entire return history into
        a single measure of return earned per unit of pain endured.

        The formula is as follows:

        - Gain-to-Pain Ratio = SUM(Returns) / SUM(|Losses|)

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            within_period (bool, optional): Whether to calculate the Gain-to-Pain Ratio within the
            specified period or for the entire period. Thus whether to look at the Gain-to-Pain Ratio
            within a specific year (if period = 'yearly') or look at the entirety of all years.
            Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Gain-to-Pain Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Gain-to-Pain Ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_gain_to_pain_ratio()
        ```

        Which returns:

        |      |    AAPL |    TSLA |
        |:-----|--------:|--------:|
        | 2021 |  0.2354 |  0.1945 |
        | 2022 | -0.108  | -0.1871 |
        | 2023 |  0.4034 |  0.3043 |
        | 2024 |  0.2462 |  0.2098 |
        | 2025 |  0.0873 |  0.0871 |
        | 2026 |  0.2062 | -0.0642 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        ).loc[:, "Return"][self._tickers_without_portfolio]

        gain_to_pain_ratio = performance_model.get_gain_to_pain_ratio(returns)

        gain_to_pain_ratio = gain_to_pain_ratio.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if growth:
            return calculate_growth(
                gain_to_pain_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return gain_to_pain_ratio

    @handle_portfolio
    @handle_errors
    def get_compound_growth_rate(
        self,
        rounding: int | None = None,
    ):
        """
        This function calculates the Compound Growth Rate (CGR) for different periods: yearly, quarterly, monthly,
        weekly, and daily.

        The CGR is a measure that provides the mean growth rate of an investment over a specified period of time.
        It is a useful measure for comparing the performance of investments over different time periods or across
        different asset classes. The CGR is calculated by taking the ratio of the final value to the initial value,
        raising it to the inverse of the number of periods, and then subtracting one.

        The formula is as follows:

            - CGR = (Final Value / Initial Value) ^ (1 / Number of Periods) — 1

        Also known as: CAGR, compound annual growth rate, annualized return.

        Args:
            rounding (int, optional): The number of decimals to round the results to. If not provided,
            the function will use the default rounding value set in the class instance.

        Returns:
            pd.DataFrame: A DataFrame containing the CGR for each period. The DataFrame has the periods
            as the index and the CGR values as the column.

        Notes:
        - When verifying the calculation, note that rounding applies and it could be slightly off because of that
        This is mostly noticeable when looking at the Compound Daily Growth Rate. Adjust the rounding with the
        rounding parameter accordingly to get a more precise figure.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_compound_growth_rate()
        ```

        Which returns:

        |                                       |   AAPL |   TSLA |   Benchmark |
        |:--------------------------------------|-------:|-------:|------------:|
        | Compound Annual Growth Rate (CAGR)    | 0.0965 | 0.0186 |      0.0779 |
        | Compound Quarterly Growth Rate (CQGR) | 0.0124 | 0.0089 |      0.0087 |
        | Compound Monthly Growth Rate (CMGR)   | 0.0124 | 0.0089 |      0.0087 |
        | Compound Weekly Growth Rate (CWGR)    | 0.0029 | 0.0022 |      0.0021 |
        | Compound Daily Growth Rate (CDGR)     | 0.0006 | 0.0005 |      0.0004 |
        """
        prices = (
            self._historical_data["yearly"]
            .loc[:, "Adj Close"]
            .loc[self._start_date : self._end_date]
        )

        cagr = performance_model.get_compound_growth_rate(prices, len(prices))

        prices = (
            self._historical_data["quarterly"]
            .loc[:, "Adj Close"]
            .loc[self._start_date : self._end_date]
        )

        cqgr = performance_model.get_compound_growth_rate(prices, len(prices))

        prices = (
            self._historical_data["monthly"]
            .loc[:, "Adj Close"]
            .loc[self._start_date : self._end_date]
        )

        cqgr = performance_model.get_compound_growth_rate(prices, len(prices))

        prices = (
            self._historical_data["weekly"]
            .loc[:, "Adj Close"]
            .loc[self._start_date : self._end_date]
        )

        cwgr = performance_model.get_compound_growth_rate(prices, len(prices))

        prices = (
            self._historical_data["daily"]
            .loc[:, "Adj Close"]
            .loc[self._start_date : self._end_date]
        )

        cdgr = performance_model.get_compound_growth_rate(prices, len(prices))

        compound_growth_rate = pd.DataFrame(
            [cagr, cqgr, cqgr, cwgr, cdgr],
            index=[
                "Compound Annual Growth Rate (CAGR)",
                "Compound Quarterly Growth Rate (CQGR)",
                "Compound Monthly Growth Rate (CMGR)",
                "Compound Weekly Growth Rate (CWGR)",
                "Compound Daily Growth Rate (CDGR)",
            ],
        )

        compound_growth_rate = compound_growth_rate.round(
            rounding if rounding else self._rounding
        )

        return compound_growth_rate

    @handle_portfolio
    @handle_errors
    def get_returns(
        self,
        period: str | None = None,
        cumulative: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Return of an investment portfolio or asset for a given period
        based on the daily historical returns.

        The period Return is obtained by compounding the daily returns within each
        period, following the formula:

            - Period Return = ((1 + Return 1) * (1 + Return 2) * ... * (1 + Return N)) - 1

        If cumulative is set to True, the period returns are compounded further into
        a cumulative return over time instead. The cumulative return is always rebased
        to start at 1 at the beginning of the selected date range.

        Also known as: periodic return.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            cumulative (bool, optional): Whether to return the cumulative return over time
            instead of the discrete return per period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Return values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.Series: Return values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Return for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Return values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_returns(period="yearly")
        ```

        Which returns:

        | Date   |    AMZN |    TSLA |   Benchmark |
        |:-------|--------:|--------:|------------:|
        | 2021   |  0.0236 |  0.4983 |      0.2701 |
        | 2022   | -0.496  | -0.6503 |     -0.1949 |
        | 2023   |  0.8089 |  1.0174 |      0.2429 |
        | 2024   |  0.4449 |  0.6255 |      0.2339 |
        | 2025   |  0.0516 |  0.1129 |      0.1638 |
        | 2026   |  0.0508 | -0.1254 |      0.0918 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        returns = self._historical_data["daily"]["Return"]

        period_returns = performance_model.get_returns(
            returns, period, cumulative=cumulative
        ).loc[self._start_date : self._end_date]

        if cumulative:
            period_returns = period_returns / period_returns.iloc[0]

        period_returns = period_returns.round(rounding if rounding else self._rounding)
        period_returns = period_returns.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                period_returns,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return period_returns

    @handle_portfolio
    @handle_errors
    def get_excess_return(
        self,
        period: str | None = None,
        cumulative: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Excess Return of an investment portfolio or asset for a given period
        based on the daily historical returns.

        The Excess Return is defined as the period Return minus the risk free rate.

        If cumulative is set to True, the excess returns are compounded further into
        a cumulative excess return over time instead. The cumulative excess return is
        always rebased to start at 1 at the beginning of the selected date range.

        Also known as: return minus the risk-free rate.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            cumulative (bool, optional): Whether to return the cumulative excess return over time
            instead of the discrete excess return per period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Excess Return values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.Series: Excess Return values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Excess Return for
        the specified `period` for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of Excess Return values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_excess_return(period="yearly")
        ```

        Which returns:

        | Date   |    AMZN |    TSLA |   Benchmark |
        |:-------|--------:|--------:|------------:|
        | 2021   |  0.0085 |  0.4832 |      0.255  |
        | 2022   | -0.5348 | -0.6891 |     -0.2337 |
        | 2023   |  0.7702 |  0.9787 |      0.2042 |
        | 2024   |  0.3992 |  0.5798 |      0.1882 |
        | 2025   |  0.01   |  0.0713 |      0.1222 |
        | 2026   |  0.0059 | -0.1703 |      0.0469 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        returns = self._historical_data["daily"]["Return"]
        period_returns = performance_model.get_returns(returns, period)
        risk_free_rate = self._risk_free_rate_data[period]

        excess_return = performance_model.get_excess_return(
            period_returns, risk_free_rate, cumulative=cumulative
        ).loc[self._start_date : self._end_date]

        if cumulative:
            excess_return = excess_return / excess_return.iloc[0]

        excess_return = excess_return.round(rounding if rounding else self._rounding)
        excess_return = excess_return.dropna(how="all", axis=0)

        if growth:
            return calculate_growth(
                excess_return,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return excess_return
