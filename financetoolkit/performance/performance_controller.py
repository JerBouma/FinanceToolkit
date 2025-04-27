"""Performance Module"""

__docformat__ = "google"

import warnings

import pandas as pd

from financetoolkit.helpers import calculate_growth, handle_portfolio
from financetoolkit.performance import performance_model
from financetoolkit.performance.helpers import (
    determine_within_dataset,
    determine_within_historical_data,
    handle_errors,
)
from financetoolkit.risk.risk_model import get_ui
from financetoolkit.utilities.logger_model import get_logger

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False

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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to None which
            results in basing it off the quarterly parameter as defined in the class instance.
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
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

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

        ticker_list_iterator = (
            tqdm(
                self._tickers_without_portfolio,
                desc="Calculating Factor Asset Correlations",
            )
            if ENABLE_TQDM & self._progress_bar
            else self._tickers_without_portfolio
        )

        for ticker in ticker_list_iterator:
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

        return self._factor_asset_correlations

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

        ticker_list_iterator = (
            tqdm(
                self._tickers_without_portfolio,
                desc=f"Calculating {'Multi' if method == 'multi' else 'Individual'} Factor Exposures",
            )
            if ENABLE_TQDM & self._progress_bar
            else self._tickers_without_portfolio
        )

        for ticker in ticker_list_iterator:
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

                fama_and_french_model = fama_and_french_model.stack(
                    future_stack=True
                ).unstack(level=[2, 1, 3])

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

            return self._fama_and_french_model, self._fama_and_french_residuals

        if growth:
            return calculate_growth(
                self._fama_and_french_model,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return self._fama_and_french_model

    @handle_portfolio
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

            - Alpha = Asset's Actual Return — Benchmark's Actual Return

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_alpha()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = self._historical_data[period]
        returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
        benchmark_returns = historical_data.loc[:, "Return"][self._benchmark_name]

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_jensens_alpha()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

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

        - - Treynor Ratio = (Portfolio's Return — Risk-Free Rate) / Portfolio Beta

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_treynor_ratio()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

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
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_data = (
            self._within_historical_data[period]
            if not rolling
            else self._historical_data[period]
        )
        excess_return = historical_data.loc[:, "Excess Return"][
            self._tickers_without_portfolio
        ]

        if rolling:
            sharpe_ratio = performance_model.get_rolling_sharpe_ratio(
                excess_return, rolling
            )
        else:
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
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

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
        """

        period = period if period else "quarterly" if self._quarterly else "yearly"
        return_column = "Return" if period == "intraday" else "Excess Return"

        historical_data = self._within_historical_data[period]
        returns = historical_data.loc[:, "Return"][self._tickers_without_portfolio]
        historical_data_within_period = self._historical_data[period]
        excess_return = historical_data_within_period.loc[:, return_column][
            self._tickers_without_portfolio
        ]

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

            - M2 Ratio = (Portfolio's Return — Risk-Free Rate) / Portfolio Standard Deviation

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_m2_ratio()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        historical_period_data = self._historical_data[period]
        period_returns = historical_period_data.loc[:, "Return"][
            self._tickers_without_portfolio
        ]
        period_standard_deviation = historical_period_data.loc[:, "Volatility"][
            self._tickers_without_portfolio
        ]
        risk_free_rate = self._risk_free_rate_data[period]

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_tracking_error()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_information_ratio()
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

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
