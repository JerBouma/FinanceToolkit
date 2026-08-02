"""Performance Module"""

__docformat__ = "google"

import warnings

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.performance import performance_model
from financetoolkit.performance.helpers import (
    determine_within_dataset,
    determine_within_historical_data,
    handle_errors,
)
from financetoolkit.risk.risk_model import (
    get_kurtosis,
    get_max_drawdown,
    get_rolling_kurtosis,
    get_rolling_skewness,
    get_skewness,
    get_ui,
    get_volatility,
)
from financetoolkit.utilities.dataframe_model import filter_columns
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import finalize_dataset

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
        self._carhart_four_factor_model: pd.DataFrame = pd.DataFrame()
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
        standardize: bool = False,
    ):
        """
        Calculates and collects all performance metrics.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        toolkit.performance.collect_all_metrics().xs("AAPL", level=1, axis=1)
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
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Beta": self.get_beta(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "CAPM": self.get_capital_asset_pricing_model(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Jensen's Alpha": self.get_jensens_alpha(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Treynor Ratio": self.get_treynor_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Sharpe Ratio": self.get_sharpe_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Sortino Ratio": self.get_sortino_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Ulcer Index": self.get_ulcer_performance_index(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Calmar Ratio": self.get_calmar_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Sterling Ratio": self.get_sterling_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Burke Ratio": self.get_burke_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Omega Ratio": self.get_omega_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Kappa Ratio": self.get_kappa_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Gain to Pain Ratio": self.get_gain_to_pain_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Win Rate": self.get_win_rate(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Upside Capture Ratio": self.get_upside_capture_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Downside Capture Ratio": self.get_downside_capture_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "M2 Ratio": self.get_m2_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Tracking Error": self.get_tracking_error(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Information Ratio": self.get_information_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
        }

        if period != "daily":
            performance_metrics["Returns"] = self.get_returns(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            performance_metrics["Excess Return"] = self.get_excess_return(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
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
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=beta,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_capital_asset_pricing_model(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=capm,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

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
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return filter_columns(
            finalize_dataset(
                dataset=self._fama_and_french_model,
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                growth=growth,
                lag=lag,
                rounding=rounding,
                standardize=standardize,
                axis="rows",
                row_slice=True,
            ),
            show_columns,
        )

    @handle_portfolio
    @handle_errors
    def get_carhart_four_factor_model(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: list[str] | None = None,
    ):
        """
        Calculate Carhart Four Factor model scores for a set of financial assets.

        The Carhart Four Factor model extends the Fama and French Three Factor model with a
        momentum factor, based on the observation that stocks with high prior returns (winners)
        tend to keep outperforming stocks with low prior returns (losers) over the medium term:

            - Market Risk Premium (Mkt-RF): The excess return of the market over the risk-free rate.
            - Size Premium (SMB): The historical excess return of small-cap stocks over large-cap stocks.
            - Value Premium (HML): The historical excess return of value stocks over growth stocks.
            - Momentum (MOM): The historical excess return of prior winner stocks over prior loser stocks.

        The model performs a Multi Linear Regression on all four factors and defines the regression
        parameters for each asset over time based on its exposure to these factors:

            - Excess Return = Intercept + Beta1 * Mkt-RF + Beta2 * SMB + Beta3 * HML + Beta4 * MOM + Residuals

        For more information about the method, see the following paper:

        - Carhart, M.M. (1997). "On Persistence in Mutual Fund Performance." The Journal of
        Finance, 52(1), 57-82.

        Also known as: Carhart model, four-factor model, momentum-augmented Fama-French model.

        Args:
            period (str, optional): The period for the calculation (e.g., "weekly", "monthly", "quarterly", "yearly").
                Defaults to None, using class-defined quarterly or yearly period.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratio values. Defaults to False.
            lag (int or list of int, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Carhart Four Factor model scores for the specified assets.

        Notes:
        - The dataset from Ken French is not always fully up to date. Therefore, some periods could be excluded.
        - Daily Carhart results is not an option as it would attempt to do a linear regression on a single data
        point which will not give any meaningful results.
        - The risk-free rate is the Risk Free Rate reported in the Fama and French dataset (used here, rather
        than the Toolkit's own risk-free rate, to stay consistent with the momentum factor's construction).
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_carhart_four_factor_model(period="quarterly")["AMZN"]
        ```

        Which returns (columns are Intercept, Mkt-RF/SMB/HML/MOM Slope, MSE and R Squared, per ticker):

        |        |   Intercept |   Mkt-RF Slope |   MOM Slope |   R Squared |
        |:-------|-------------:|----------------:|-------------:|-------------:|
        | 2025Q4 |      -0.0194 |          0.0191 |      -0.006 |       0.4786 |
        | 2026Q1 |      -0.0122 |          0.0126 |     -0.0034 |       0.3878 |
        | 2026Q2 |      -0.0119 |          0.0076 |      0.0033 |       0.2513 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data_within = self._within_historical_data[period]
        returns = historical_data_within.loc[:, "Return"][
            self._tickers_without_portfolio
        ]

        self._fama_and_french_dataset = (
            performance_model.obtain_fama_and_french_dataset()
        )
        momentum_dataset = performance_model.obtain_carhart_momentum_dataset()

        carhart_dataset = self._fama_and_french_dataset[
            ["Mkt-RF", "SMB", "HML", "RF"]
        ].merge(momentum_dataset, left_index=True, right_index=True)
        carhart_dataset = carhart_dataset.rename(columns={"Mom": "MOM"})

        carhart_period = determine_within_dataset(
            carhart_dataset, period, correlation=False
        )

        merged_df = carhart_period.merge(returns, left_index=True, right_index=True)

        factors_to_calculate = ["Mkt-RF", "SMB", "HML", "MOM"]
        factor_scores: dict = {}

        logger.info("Calculating Carhart Four Factor Exposures")
        for ticker in self._tickers_without_portfolio:
            factor_scores[ticker] = {}

            for dataset_period in merged_df.index.get_level_values(0):
                factor_data = merged_df.loc[dataset_period][factors_to_calculate]
                excess_returns = (
                    merged_df.loc[dataset_period][ticker]
                    - merged_df.loc[dataset_period]["RF"]
                )

                (
                    factor_scores[ticker][dataset_period],
                    _,
                    error_message,
                ) = performance_model.get_fama_and_french_model_multi(
                    excess_returns=excess_returns, factor_dataset=factor_data
                )

                if error_message:
                    logger.warning(
                        "%s for %s in %s.", error_message, ticker, dataset_period
                    )

            carhart_model = pd.DataFrame.from_dict(
                {
                    (ticker, factor): value
                    for ticker, factor_scores_ticker in factor_scores.items()
                    for factor, value in factor_scores_ticker.items()
                },
                orient="index",
            )

        carhart_model = carhart_model.unstack(level=0, sort=False).swaplevel(
            0, 1, axis=1
        )

        tickers_column_order = carhart_model.columns.get_level_values(0).unique()
        parameters_column_order = carhart_model.columns.get_level_values(1).unique()

        carhart_model = (
            carhart_model.sort_index(axis=1)
            .reindex(tickers_column_order, level=0, axis=1)
            .reindex(parameters_column_order, level=1, axis=1)
        )

        self._carhart_four_factor_model = carhart_model.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        return filter_columns(
            finalize_dataset(
                dataset=self._carhart_four_factor_model,
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                growth=growth,
                lag=lag,
                rounding=rounding,
                standardize=standardize,
                axis="rows",
                row_slice=True,
            ),
            show_columns,
        )

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
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=alpha,
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
    def get_jensens_alpha(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=jensens_alpha,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_treynor_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=treynor_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_sharpe_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=sharpe_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_probabilistic_sharpe_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        benchmark_sharpe_ratio: float = 0.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Probabilistic Sharpe Ratio (PSR), the probability that the true
        (population) Sharpe ratio exceeds a benchmark Sharpe ratio, correcting the
        naive Sharpe ratio significance test for skewed and fat-tailed returns.

        A plain Sharpe ratio significance test (e.g. treating SR̂ as approximately
        normally distributed) implicitly assumes Gaussian, i.i.d. returns. Real asset
        and strategy returns are typically skewed and fat-tailed, which understates the
        true uncertainty around the Sharpe ratio estimate and makes the naive test
        overconfident. The PSR explicitly folds the skewness and (non-excess) kurtosis
        of the underlying returns into the standard error of the Sharpe ratio, giving a
        more honest probability that the strategy truly beats `benchmark_sharpe_ratio`
        rather than 0 or 0.5 simply being a coincidence of a short, lumpy sample.

        The formula is as follows:

            - PSR(SR*) = Φ( (SR̂ − SR*) · sqrt(n − 1) / sqrt(1 − γ₃·SR̂ + ((γ₄ − 1) / 4)·SR̂²) )

        Where SR̂ is the observed Sharpe ratio, SR* is `benchmark_sharpe_ratio`, γ₃ is
        skewness, γ₄ is the non-excess (raw) kurtosis, n is the number of return
        observations and Φ is the standard normal CDF.

        Also known as: PSR, Sharpe ratio significance probability.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling period to use for the calculation. If you select
            period = 'monthly' and set rolling to 12 you obtain the rolling 12-month Probabilistic
            Sharpe Ratio.
            benchmark_sharpe_ratio (float, optional): The hypothesized or benchmark Sharpe ratio
            (SR*) to test the observed Sharpe ratio against. Defaults to 0.0, i.e. testing whether
            the strategy has any skill at all above doing nothing.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Probabilistic Sharpe Ratio values, between 0 and 1.

        Notes:
        - This uses the **non-excess (raw)** kurtosis convention, i.e. a Normal distribution has a
        kurtosis of 3, not 0. Internally this calls `risk_model.get_kurtosis(..., fisher=False)`.
        - The method retrieves historical data and calculates the Probabilistic Sharpe ratio for
        each asset in the Toolkit instance, using the same excess returns as `get_sharpe_ratio`.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_probabilistic_sharpe_ratio()
        ```

        Which returns:

        | Date   |   AAPL |   TSLA |
        |:-------|-------:|-------:|
        | 2021   | 0.0000 | 0.0008 |
        | 2022   | 0.0000 | 0.0000 |
        | 2023   | 0.0000 | 0.0000 |
        | 2024   | 0.0000 | 0.0000 |
        | 2025   | 0.0000 | 0.0000 |
        | 2026   | 0.0000 | 0.0000 |
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
            skewness = get_rolling_skewness(excess_return, rolling)
            kurtosis = get_rolling_kurtosis(excess_return, rolling, fisher=False)
            n_observations = rolling
        else:
            excess_return = self._within_historical_data[period].loc[
                :, "Excess Return"
            ][self._tickers_without_portfolio]

            sharpe_ratio = performance_model.get_sharpe_ratio(excess_return)
            skewness = get_skewness(excess_return)
            kurtosis = get_kurtosis(excess_return, fisher=False)
            n_observations = excess_return.groupby(level=0).count()

        probabilistic_sharpe_ratio = performance_model.get_probabilistic_sharpe_ratio(
            sharpe_ratio=sharpe_ratio,
            benchmark_sharpe_ratio=benchmark_sharpe_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            n_observations=n_observations,
        )

        return finalize_dataset(
            dataset=probabilistic_sharpe_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_deflated_sharpe_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        trials_window: int | None = None,
        n_trials: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Deflated Sharpe Ratio (DSR), the Probabilistic Sharpe Ratio
        corrected for the fact that the reported Sharpe ratio is often the best of many
        strategy variations, parameter combinations, or lookback windows tried during a
        backtest (multiple testing / selection bias / "backtest overfitting").

        The more variations that were tried, the more likely it is that at least one of
        them shows an impressive Sharpe ratio by pure chance, even with zero true skill.
        The DSR accounts for this by first estimating the Sharpe ratio one would expect
        to observe, purely by chance, as the maximum of `n_trials` independent trials
        under the null hypothesis of no skill, and then uses that expected maximum as
        the benchmark (SR*) in the Probabilistic Sharpe Ratio formula, instead of a
        naive benchmark such as 0.

        The formula for the expected maximum Sharpe ratio benchmark is as follows:

            - SR* = sqrt(Var[SR_trials]) · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

        Where N is `n_trials`, Var[SR_trials] is the variance of the Sharpe ratios
        observed across those N trials, and γ ≈ 0.5772 is the Euler-Mascheroni constant.
        DSR = PSR(SR*), i.e. it is always less than or equal to the Probabilistic Sharpe
        Ratio computed against a benchmark of 0.

        This codebase does not track "N literal strategy trials" — there is no record of
        how many parameter combinations were tried before arriving at the current
        Toolkit configuration. As a documented approximation, `Var[SR_trials]` is
        estimated from the variance of an auxiliary *rolling* Sharpe ratio series (see
        `get_rolling_sharpe_ratio`) computed over a `trials_window`-sized window across
        the full return history, and `n_trials` defaults to the number of valid
        (non-NaN) values in that same rolling series. This treats each rolling window as
        if it were one "trial" — a reasonable proxy for how dispersed the Sharpe ratio
        could plausibly have been under different choices, but not a substitute for
        passing the actual number of variations tried (via `n_trials`) when that is
        known, since the quality of the correction depends directly on it.

        Also known as: DSR, backtest overfitting correction, selection-bias-adjusted Sharpe ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling period to use for the primary Sharpe ratio
            being tested. If you select period = 'monthly' and set rolling to 12 you obtain the
            rolling 12-month Deflated Sharpe Ratio.
            trials_window (int, optional): The window size (in units of `period`) used for the
            auxiliary rolling Sharpe ratio series that approximates `Var[SR_trials]` and the
            default `n_trials`, see the Notes above. Defaults to None, which uses half of the
            available return history so that enough overlapping windows exist regardless of
            `period` or date range.
            n_trials (int, optional): The number of independent (or effectively independent)
            strategy variations, parameter combinations, or lookback windows tried before
            arriving at the reported Sharpe ratio. Defaults to None, which falls back to the
            number of valid values in the auxiliary rolling Sharpe ratio series described above.
            Pass this explicitly whenever the actual number of trials is known.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Deflated Sharpe Ratio values, between 0 and 1.

        Notes:
        - This uses the **non-excess (raw)** kurtosis convention, i.e. a Normal distribution has a
        kurtosis of 3, not 0. Internally this calls `risk_model.get_kurtosis(..., fisher=False)`.
        - The method retrieves historical data and calculates the Deflated Sharpe ratio for
        each asset in the Toolkit instance, using the same excess returns as `get_sharpe_ratio`.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_deflated_sharpe_ratio()
        ```

        Which returns:

        | Date   |   AAPL |   TSLA |
        |:-------|-------:|-------:|
        | 2021   | 0.0000 | 0.0000 |
        | 2022   | 0.0000 | 0.0000 |
        | 2023   | 0.0000 | 0.0000 |
        | 2024   | 0.0000 | 0.0000 |
        | 2025   | 0.0000 | 0.0000 |
        | 2026   | 0.0000 | 0.0000 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        period_returns = self._historical_data[period].loc[:, "Return"][
            self._tickers_without_portfolio
        ]
        full_excess_return = performance_model.get_excess_return(
            period_returns, self._risk_free_rate_data[period]
        )

        trials_window = (
            trials_window
            if trials_window is not None
            else max(2, len(period_returns) // 2)
        )
        trials_sharpe_ratio = performance_model.get_rolling_sharpe_ratio(
            full_excess_return, trials_window
        )
        sharpe_ratio_variance = trials_sharpe_ratio.var()
        trials = n_trials if n_trials is not None else trials_sharpe_ratio.count()

        if rolling:
            sharpe_ratio = performance_model.get_rolling_sharpe_ratio(
                full_excess_return, rolling
            )
            skewness = get_rolling_skewness(full_excess_return, rolling)
            kurtosis = get_rolling_kurtosis(full_excess_return, rolling, fisher=False)
            n_observations = rolling
        else:
            excess_return = self._within_historical_data[period].loc[
                :, "Excess Return"
            ][self._tickers_without_portfolio]

            sharpe_ratio = performance_model.get_sharpe_ratio(excess_return)
            skewness = get_skewness(excess_return)
            kurtosis = get_kurtosis(excess_return, fisher=False)
            n_observations = excess_return.groupby(level=0).count()

        deflated_sharpe_ratio = performance_model.get_deflated_sharpe_ratio(
            sharpe_ratio=sharpe_ratio,
            sharpe_ratio_variance=sharpe_ratio_variance,
            n_trials=trials,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        return finalize_dataset(
            dataset=deflated_sharpe_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_sortino_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=sortino_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_ulcer_performance_index(
        self,
        period: str | None = None,
        rolling: int = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=ulcer_performance_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_calmar_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=calmar_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

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
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=sterling_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_burke_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=burke_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_m2_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        The M2 Ratio, also known as the Modigliani-Modigliani Measure, is a financial
        metric used to evaluate the risk-adjusted performance of an investment portfolio
        or strategy. It assesses the excess return generated by the portfolio relative
        to a risk-free investment, taking into account the portfolio's volatility or risk.
        The M2 Ratio helps investors and portfolio managers determine whether the portfolio
        is delivering returns that justify its level of risk.

        The formula is as follows:

            - M2 Ratio = Risk-Free Rate + [(Portfolio's Return — Risk-Free Rate) /
              Portfolio Standard Deviation] × Benchmark Standard Deviation

        This rescales the (dimensionless) Sharpe ratio back into return-space by asking
        what return the portfolio would have earned had it been leveraged or
        de-leveraged, via risk-free borrowing or lending, to match the benchmark's
        volatility exactly -- producing a number directly comparable to the benchmark's
        actual return. Requires a `benchmark_ticker` to be set on the Toolkit instance,
        since the benchmark's standard deviation is part of the formula.

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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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
        benchmark_period_returns = historical_period_data.loc[:, "Return"][
            self._benchmark_name
        ]
        risk_free_rate = self._risk_free_rate_data[period]

        if rolling:
            m2_ratio = performance_model.get_rolling_m2_ratio(
                period_returns, risk_free_rate, benchmark_period_returns, rolling
            )
        else:
            daily_returns = self._historical_data["daily"].loc[:, "Return"][
                self._tickers_without_portfolio
            ]
            benchmark_daily_returns = self._historical_data["daily"].loc[:, "Return"][
                self._benchmark_name
            ]
            period_standard_deviation = get_volatility(daily_returns, period)
            benchmark_standard_deviation = get_volatility(
                benchmark_daily_returns, period
            )

            m2_ratio = performance_model.get_m2_ratio(
                period_returns,
                risk_free_rate,
                period_standard_deviation,
                benchmark_standard_deviation,
            )

        return finalize_dataset(
            dataset=m2_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_tracking_error(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=tracking_error,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_information_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=information_ratio,
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
    def get_upside_capture_ratio(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=upside_capture_ratio,
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
    def get_downside_capture_ratio(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=downside_capture_ratio,
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
    def get_win_rate(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=win_rate,
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
    def get_kappa_ratio(
        self,
        period: str | None = None,
        order: int = 3,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=kappa_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

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
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=omega_ratio,
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
    def get_gain_to_pain_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=gain_to_pain_ratio,
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
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=period_returns,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_excess_return(
        self,
        period: str | None = None,
        cumulative: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        return finalize_dataset(
            dataset=excess_return,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_correlation_matrix(
        self,
        period: str | None = None,
        rounding: int | None = None,
    ):
        """
        Calculate the full pairwise Correlation Matrix across all assets (and the
        benchmark) in the Toolkit instance, based on the daily historical returns.

        Unlike `get_beta`, which relates a single asset to the benchmark, this computes
        the correlation between every pair of assets at once. This is a prerequisite for
        portfolio variance calculations and any mean-variance optimization work.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: The N x N Correlation Matrix, with assets as both the index and
            the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_correlation_matrix()
        ```

        Which returns:

        |           |   AMZN |   TSLA |   Benchmark |
        |:----------|-------:|-------:|------------:|
        | AMZN      | 1      | 0.935  |      0.7751 |
        | TSLA      | 0.935  | 1      |      0.8982 |
        | Benchmark | 0.7751 | 0.8982 |      1      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._historical_data[period]["Return"].loc[
            self._start_date : self._end_date
        ]

        correlation_matrix = performance_model.get_correlation_matrix(returns)

        return correlation_matrix.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_covariance_matrix(
        self,
        period: str | None = None,
        rounding: int | None = None,
    ):
        """
        Calculate the full pairwise Covariance Matrix across all assets (and the
        benchmark) in the Toolkit instance, based on the daily historical returns.

        Unlike `get_covariance`, which relates a single asset to the benchmark, this
        computes the covariance between every pair of assets at once. This is a
        prerequisite for portfolio variance calculations and any mean-variance
        optimization work.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: The N x N Covariance Matrix, with assets as both the index and
            the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_covariance_matrix()
        ```

        Which returns:

        |           |   AMZN |   TSLA |   Benchmark |
        |:----------|-------:|-------:|------------:|
        | AMZN      | 0.1944 | 0.2418 |      0.0592 |
        | TSLA      | 0.2418 | 0.344  |      0.0913 |
        | Benchmark | 0.0592 | 0.0913 |      0.0301 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._historical_data[period]["Return"].loc[
            self._start_date : self._end_date
        ]

        covariance_matrix = performance_model.get_covariance_matrix(returns)

        return covariance_matrix.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_appraisal_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Appraisal Ratio, i.e. Jensen's Alpha divided by the idiosyncratic
        (residual, unsystematic) standard deviation left over from the CAPM regression
        that produced that Alpha.

        Jensen's Alpha (see `get_jensens_alpha`) measures how much return a manager
        generated above what CAPM would predict given the asset's Beta. However, a large
        Alpha achieved with wildly noisy, unpredictable residual returns is far less
        attractive than the same Alpha achieved consistently. The Appraisal Ratio
        normalizes Alpha by that noise (the "specific risk" not explained by market
        exposure), giving a Sharpe-ratio-like measure of stock-picking or timing skill per
        unit of idiosyncratic risk taken.

        The formula is as follows:

        - Appraisal Ratio = Jensen's Alpha / Residual Standard Deviation

        Where the residual standard deviation is the standard deviation of the pointwise
        CAPM regression residuals (Asset Excess Return − Beta * Benchmark Excess Return),
        reusing the exact same CAPM regression formula as `get_jensens_alpha`.

        See definition: https://en.wikipedia.org/wiki/Information_ratio

        Also known as: Treynor-Black Appraisal Ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Appraisal Ratio values.

        Notes:
        - Daily Appraisal Ratio is not an option as the standard deviation for 1 day is close to
        zero. Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates Jensen's Alpha and the CAPM
        regression residuals for each asset in the Toolkit instance, reusing the same Beta and
        CAPM formula as `get_jensens_alpha`.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_appraisal_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    MSFT |
        |:-------|--------:|--------:|
        | 2020   | 37.2641 | 16.0511 |
        | 2021   | -2.5308 | 20.9479 |
        | 2022   | -1.4753 | -3.3995 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            historical_data = self._historical_data[period]
            returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
            benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
            )

            risk_free_rate = self._risk_free_rate_data[period]
            within_excess_return = performance_model.get_excess_return(
                returns, risk_free_rate
            )
            within_benchmark_excess_return = performance_model.get_excess_return(
                benchmark_returns, risk_free_rate
            )

            capm_residuals = performance_model.get_jensens_alpha(
                within_excess_return, 0.0, beta, within_benchmark_excess_return
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

            within_excess_return = historical_within_data.loc[:, "Excess Return"][
                self._tickers_without_portfolio
            ]
            within_benchmark_excess_return = historical_within_data.loc[
                :, "Excess Return"
            ][self._benchmark_name]

            capm_residuals = performance_model.get_capm_residuals(
                within_excess_return, beta, within_benchmark_excess_return
            )

        historical_data = self._historical_data[period]

        period_returns = historical_data.loc[:, "Return"][
            self._tickers_without_portfolio
        ]

        risk_free_rate = self._risk_free_rate_data[period]
        benchmark_period_returns = historical_data.loc[:, "Return"][
            self._benchmark_name
        ]

        jensens_alpha = performance_model.get_jensens_alpha(
            period_returns, risk_free_rate, beta, benchmark_period_returns
        )

        appraisal_ratio = performance_model.get_appraisal_ratio(
            jensens_alpha, capm_residuals
        )

        return finalize_dataset(
            dataset=appraisal_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_fama_decomposition(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Fama (1972) decomposition of total excess return into Selectivity
        and Diversification.

        Jensen's Alpha alone conflates two very different sources of excess return:
        genuine stock/timing selection skill, and simply carrying more total risk than the
        market by holding an under-diversified portfolio (which, in a CAPM world, should
        be compensated with extra return even absent any skill). Fama's decomposition
        separates the two by comparing the portfolio's actual return against two different
        CAPM-implied return benchmarks: one using the portfolio's actual Beta (systematic
        risk only), and one using the portfolio's actual *total* risk ratio
        (Sigma_Portfolio / Sigma_Market) in place of Beta.

        The formulas are as follows:

        - Selectivity = (Asset Return − Risk-Free Rate) − (Sigma_Portfolio / Sigma_Market)
            * (Benchmark Return − Risk-Free Rate)
        - Diversification = [Risk-Free Rate + (Sigma_Portfolio / Sigma_Market)
            * (Benchmark Return − Risk-Free Rate)] − [Risk-Free Rate + Beta * (Benchmark Return − Risk-Free Rate)]

        Selectivity is the return earned above what would be required for a fully
        diversified portfolio carrying the same total risk, i.e. genuine security
        selection or timing skill. Diversification is the extra return the manager left on
        the table (if positive, it is a cost) by taking on unsystematic risk that a fully
        diversified portfolio of the same total risk would not have. Selectivity plus
        Diversification equals Jensen's Alpha (see `get_jensens_alpha`).

        Also known as: Fama's Net Selectivity, Fama performance decomposition.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Selectivity and Diversification values, with a Multi Index of
            (ticker, component) as the columns.

        Notes:
        - Daily Fama Decomposition is not an option as the standard deviation for 1 day is close
        to zero. Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates Beta, the asset's and benchmark's
        standard deviation, and the Selectivity and Diversification components for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_fama_decomposition().xs("AAPL", level=0, axis=1)
        ```

        Which returns:

        | Date   |   Selectivity |   Diversification |
        |:-------|--------------:|-------------------:|
        | 2020   |        0.5708 |              0.0416 |
        | 2021   |       -0.1945 |              0.1653 |
        | 2022   |        0.0220 |             -0.0375 |
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
        benchmark_period_returns = historical_data.loc[:, "Return"][
            self._benchmark_name
        ]

        daily_returns = self._historical_data["daily"].loc[:, "Return"][
            self._tickers_without_portfolio
        ]
        daily_benchmark_returns = self._historical_data["daily"].loc[:, "Return"][
            self._benchmark_name
        ]

        asset_standard_deviation = get_volatility(daily_returns, period)
        benchmark_standard_deviation = get_volatility(daily_benchmark_returns, period)

        selectivity, diversification = performance_model.get_fama_decomposition(
            period_returns,
            risk_free_rate,
            beta,
            benchmark_period_returns,
            asset_standard_deviation,
            benchmark_standard_deviation,
        )

        fama_decomposition = pd.concat(
            {"Selectivity": selectivity, "Diversification": diversification}, axis=1
        ).swaplevel(0, 1, axis=1)

        fama_decomposition = fama_decomposition.reindex(
            self._tickers_without_portfolio, level=0, axis=1
        )

        return finalize_dataset(
            dataset=fama_decomposition,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_adjusted_sharpe_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Adjusted Sharpe Ratio (ASR) of an investment portfolio or asset's
        returns.

        The Sharpe ratio only looks at the mean and standard deviation of returns,
        implicitly assuming a Normal distribution. The Adjusted Sharpe Ratio (Pezier &
        White, 2006) penalizes (or rewards) the Sharpe ratio for negative skewness and
        excess kurtosis using a Cornish-Fisher-style expansion, so that two strategies
        with the same Sharpe ratio but different tail shapes are no longer scored
        identically.

        The formula is as follows:

        - ASR = SR * [1 + (S / 6) * SR − ((K − 3) / 24) * SR^2]

        Where SR is the (ordinary, period) Sharpe ratio, S is the skewness of the same
        returns, and K is the non-excess (raw) kurtosis of the same returns.

        Also known as: Pezier and White Adjusted Sharpe Ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rolling (int, optional): The rolling period to use for the calculation. If you select
            period = 'monthly' and set rolling to 12 you obtain the rolling 12-month Adjusted
            Sharpe Ratio.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Adjusted Sharpe Ratio values.

        Notes:
        - This uses the **non-excess (raw)** kurtosis convention, i.e. a Normal distribution has a
        kurtosis of 3, not 0. Internally this calls `risk_model.get_kurtosis(..., fisher=False)`,
        the same convention documented in `get_probabilistic_sharpe_ratio`.
        - Daily Adjusted Sharpe Ratio is not an option as the standard deviation for 1 day is close
        to zero. Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates the Adjusted Sharpe ratio for each
        asset in the Toolkit instance, using the same excess returns as `get_sharpe_ratio`.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_adjusted_sharpe_ratio()
        ```

        Which returns:

        | Date   |    AAPL |    MSFT |
        |:-------|--------:|--------:|
        | 2020   | -0.2021 | -0.2502 |
        | 2021   | -0.8212 | -0.9321 |
        | 2022   | -1.2058 | -1.2489 |
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
            skewness = get_rolling_skewness(excess_return, rolling)
            kurtosis = get_rolling_kurtosis(excess_return, rolling, fisher=False)
        else:
            excess_return = self._within_historical_data[period].loc[
                :, "Excess Return"
            ][self._tickers_without_portfolio]

            sharpe_ratio = performance_model.get_sharpe_ratio(excess_return)
            skewness = get_skewness(excess_return)
            kurtosis = get_kurtosis(excess_return, fisher=False)

        adjusted_sharpe_ratio = performance_model.get_adjusted_sharpe_ratio(
            sharpe_ratio=sharpe_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        return finalize_dataset(
            dataset=adjusted_sharpe_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_starr_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        alpha: float = 0.05,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the STARR (Stable Tail Adjusted Return Ratio) of an investment
        portfolio or asset's returns.

        The Sharpe ratio penalizes upside and downside volatility equally via the
        standard deviation. The STARR ratio instead scales the mean excess return by the
        Conditional Value at Risk (CVaR / Expected Shortfall), a coherent tail-risk
        measure that only looks at the average magnitude of losses beyond the `alpha`
        quantile. This makes STARR more appropriate than the Sharpe ratio for return
        distributions with fat left tails.

        The formula is as follows:

        - STARR Ratio = Excess Return / |CVaR(alpha)|

        See definition: https://en.wikipedia.org/wiki/Expected_shortfall

        Also known as: Stable Tail Adjusted Return Ratio, Conditional Sharpe Ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            within_period (bool, optional): Whether to calculate the CVaR within the specified
            period or for the entire period. Thus whether to look at the CVaR within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            alpha (float, optional): The confidence level used for the CVaR calculation (e.g. 0.05
            for the worst 5% of outcomes). Defaults to 0.05.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: STARR Ratio values.

        Notes:
        - The method retrieves historical data and calculates the STARR Ratio for each asset in
        the Toolkit instance.
        - Periods with very few return observations (e.g. a partial period at the very start of
        the selected date range) can produce a degenerate (e.g. zero or ±infinite) CVaR, since
        CVaR is not a meaningful statistic with only one or two data points. This mirrors the
        analogous caveat for the Sharpe Ratio needing enough observations for its standard
        deviation to be meaningful.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_starr_ratio()
        ```

        Which returns:

        | Date   |     AAPL |    MSFT |
        |:-------|---------:|--------:|
        | 2020   |  12.0414 |  6.5831 |
        | 2021   |   9.8271 | 18.3701 |
        | 2022   |  -6.7460 | -6.8990 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        ).loc[:, "Return"][self._tickers_without_portfolio]

        historical_data = self._historical_data[period]
        period_returns = historical_data.loc[:, "Return"][
            self._tickers_without_portfolio
        ]
        risk_free_rate = self._risk_free_rate_data[period]

        excess_return = performance_model.get_excess_return(
            period_returns, risk_free_rate
        )

        starr_ratio = performance_model.get_starr_ratio(excess_return, returns, alpha)

        return finalize_dataset(
            dataset=starr_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_rachev_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        alpha: float = 0.05,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Rachev Ratio (R-Ratio) of an investment portfolio or asset's
        returns.

        The Rachev ratio compares the "quality" of the best outcomes to the "quality" of
        the worst outcomes by taking the ratio of the right-tail Expected Shortfall (the
        average of the best `alpha` fraction of returns) to the left-tail Expected
        Shortfall (the average magnitude of the worst `alpha` fraction of returns). A
        ratio above 1 indicates that the average size of extreme gains outweighs the
        average size of extreme losses.

        The formula is as follows:

        - Rachev Ratio = ES_right(alpha) / ES_left(alpha)

        Also known as: R-Ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            within_period (bool, optional): Whether to calculate the Rachev Ratio within the
            specified period or for the entire period. Thus whether to look at the return
            distribution within a specific year (if period = 'yearly') or look at the entirety of
            all years. Defaults to True.
            alpha (float, optional): The confidence level used for both tails (e.g. 0.05 for the
            best/worst 5% of outcomes). Defaults to 0.05.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Rachev Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Rachev Ratio for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_rachev_ratio()
        ```

        Which returns:

        | Date   |   AAPL |   MSFT |
        |:-------|-------:|-------:|
        | 2020   | 1.0649 | 1.0946 |
        | 2021   | 0.9964 | 1.0552 |
        | 2022   | 1.0790 | 1.0200 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        ).loc[:, "Return"][self._tickers_without_portfolio]

        rachev_ratio = performance_model.get_rachev_ratio(returns, alpha)

        return finalize_dataset(
            dataset=rachev_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_treynor_mazuy_model(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Treynor-Mazuy market timing model for each asset in the Toolkit
        instance.

        Jensen's Alpha and Beta from a plain CAPM regression cannot distinguish
        stock-picking skill (selectivity) from market-timing skill (shifting exposure
        ahead of market moves). The Treynor-Mazuy model adds a quadratic term in the
        benchmark excess return to the regression: a manager who successfully increases
        (decreases) market exposure ahead of up (down) markets will show a return profile
        that curves upward as a function of the benchmark return, captured by a positive
        quadratic coefficient (Gamma).

        The formula is as follows:

        - Excess Return = Alpha + Beta * Benchmark Excess Return + Gamma * Benchmark Excess Return^2 + Residuals

        Gamma > 0 indicates positive market-timing ability; Gamma <= 0 indicates no
        timing ability.

        Also known as: Treynor-Mazuy quadratic timing model, TM model.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Alpha, Beta, Gamma and R Squared values, with a Multi Index of
            (ticker, parameter) as the columns.

        Notes:
        - Daily and weekly Treynor-Mazuy results are not an option as there would be too few
        observations within each period to run a meaningful regression.
        - The method retrieves historical data and performs a quadratic regression for each asset
        in the Toolkit instance, within each period.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_treynor_mazuy_model().xs("AAPL", level=0, axis=1)
        ```

        Which returns:

        | Date   |   Alpha |   Beta |   Gamma |   R Squared |
        |:-------|--------:|-------:|--------:|------------:|
        | 2020   |  0.0030 | 1.1648 |  0.3569 |       0.6932 |
        | 2021   |  0.0051 | 1.4745 |  5.9823 |       0.4704 |
        | 2022   |  0.0090 | 1.3691 |  1.5507 |       0.8050 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_within_data = self._within_historical_data[period]
        excess_return = historical_within_data.loc[:, "Excess Return"][
            self._tickers_without_portfolio
        ]
        benchmark_excess_return = historical_within_data.loc[:, "Excess Return"][
            self._benchmark_name
        ]

        logger.info("Calculating Treynor-Mazuy Market Timing Model")

        regression_results: dict[str, pd.DataFrame] = {}

        for ticker in self._tickers_without_portfolio:
            ticker_results: dict = {}

            for sub_period in excess_return.index.get_level_values(0).unique():
                asset_excess = excess_return.loc[sub_period, ticker]
                benchmark_excess = benchmark_excess_return.loc[sub_period]

                result, _ = performance_model.get_treynor_mazuy_model(
                    asset_excess, benchmark_excess
                )
                ticker_results[sub_period] = result

            regression_results[ticker] = pd.DataFrame.from_dict(
                ticker_results, orient="index"
            )

        treynor_mazuy_model = pd.concat(regression_results, axis=1)

        return finalize_dataset(
            dataset=treynor_mazuy_model,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_henriksson_merton_model(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Henriksson-Merton market timing model for each asset in the
        Toolkit instance.

        Like the Treynor-Mazuy model (see `get_treynor_mazuy_model`), this separates
        market-timing skill from selectivity, but models timing as a piecewise (rather
        than quadratic) change in Beta: a "down-market" Beta and an "up-market" Beta.

        The formula is as follows:

        - Excess Return = Alpha + Beta * Benchmark Excess Return
            + Up Market Beta * max(Benchmark Excess Return, 0) + Residuals

        Beta is the "down-market" Beta (the portfolio's market exposure when the
        benchmark excess return is negative), and Beta + Up Market Beta is the
        "up-market" Beta. Up Market Beta > 0 indicates positive market-timing ability;
        Up Market Beta <= 0 indicates no timing ability.

        Also known as: Henriksson-Merton piecewise timing model, HM model.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Alpha, Beta, Up Market Beta and R Squared values, with a Multi Index of
            (ticker, parameter) as the columns.

        Notes:
        - Daily and weekly Henriksson-Merton results are not an option as there would be too few
        observations within each period to run a meaningful regression.
        - The method retrieves historical data and performs a piecewise regression for each asset
        in the Toolkit instance, within each period.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_henriksson_merton_model().xs("AAPL", level=0, axis=1)
        ```

        Which returns:

        | Date   |   Alpha |   Beta |   Up Market Beta |   R Squared |
        |:-------|--------:|-------:|------------------:|------------:|
        | 2020   |  0.0032 | 1.1578 |            -0.0071 |       0.6929 |
        | 2021   |  0.0033 | 1.2403 |             1.3512 |       0.4740 |
        | 2022   |  0.0068 | 1.2399 |             1.5621 |       0.8105 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_within_data = self._within_historical_data[period]
        excess_return = historical_within_data.loc[:, "Excess Return"][
            self._tickers_without_portfolio
        ]
        benchmark_excess_return = historical_within_data.loc[:, "Excess Return"][
            self._benchmark_name
        ]

        logger.info("Calculating Henriksson-Merton Market Timing Model")

        regression_results: dict[str, pd.DataFrame] = {}

        for ticker in self._tickers_without_portfolio:
            ticker_results: dict = {}

            for sub_period in excess_return.index.get_level_values(0).unique():
                asset_excess = excess_return.loc[sub_period, ticker]
                benchmark_excess = benchmark_excess_return.loc[sub_period]

                result, _ = performance_model.get_henriksson_merton_model(
                    asset_excess, benchmark_excess
                )
                ticker_results[sub_period] = result

            regression_results[ticker] = pd.DataFrame.from_dict(
                ticker_results, orient="index"
            )

        henriksson_merton_model = pd.concat(regression_results, axis=1)

        return finalize_dataset(
            dataset=henriksson_merton_model,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )
