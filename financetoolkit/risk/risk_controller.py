"""Risk Module"""

__docformat__ = "google"

import warnings

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import (
    cvar_model,
    evar_model,
    garch_model,
    risk_model,
    var_model,
)
from financetoolkit.risk.helpers import determine_within_historical_data
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.statistics_model import finalize_dataset

# Runtime errors are ignored on purpose given the nature of the calculations
# sometimes leading to division by zero or other mathematical errors. This is however
# for financial analysis purposes not an issue and should not be considered as a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals
# pylint: disable=too-many-boolean-expressions


class Risk:
    """
    The Risk module is meant to calculate important risk metrics such
    as Value at Risk (VaR), Conditional Value at Risk (cVaR), Maximum
    Drawdown, Correlations, GARCH, EWMA and more.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: pd.DataFrame = pd.DataFrame(),
        risk_free_rate_data: pd.DataFrame = pd.DataFrame(),
        intraday_period: str | None = None,
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Risk Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the Toolkit instance.
            historical_data (pd.DataFrame, optional): The historical data containing all periods.
                Defaults to pd.DataFrame().
            risk_free_rate_data (pd.DataFrame, optional): The risk free rate data to use for the
                Excess Volatility calculations. Defaults to pd.DataFrame().
            intraday_period (str | None, optional): The intraday period used for within-period calculations.
                Defaults to None.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculations. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculations. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_value_at_risk(period='yearly')
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2012   |  0      |  0      |
        | 2013   |  0.1754 |  4.96   |
        | 2014   |  1.7515 |  0.9481 |
        | 2015   | -0.1958 |  0.1454 |
        | 2016   |  0.4177 | -0.3437 |
        | 2017   |  2.6368 |  1.2225 |
        | 2018   | -0.2786 |  0.0718 |
        | 2019   |  3.2243 |  0.4707 |
        | 2020   |  1.729  |  8.3319 |
        | 2021   |  1.3179 |  0.8797 |
        | 2022   | -0.8026 | -1.0046 |
        | 2023   |  1.8549 |  1.8238 |
        """
        self._historical_data = historical_data
        self._risk_free_rate_data = risk_free_rate_data
        self._tickers = tickers
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._portfolio_weights: dict | None = None

        # Within Return Calculations
        daily_historical_data = self._historical_data["daily"].copy().fillna(0)
        intraday_historical_data = self._historical_data["intraday"].copy().fillna(0)

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

    @handle_portfolio
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
        Calculates and collects all risk metrics.

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
            pd.Series or pd.DataFrame: Risk metrics calculated based on the specified parameters.

        Notes:
        - The method calculates various risk metrics for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.collect_all_metrics()
        ```

        Which returns:

        |      |   Ulcer Index |   GARCH |   Skewness |   Kurtosis |   Downside Deviation |   Variance |   Volatility |
        |:-----|--------------:|--------:|-----------:|-----------:|---------------------:|-----------:|-------------:|
        | 2021 |        0.0376 |  0.0616 |    -0.0677 |     3.3347 |               0.0102 |     0.063  |       0.2511 |
        | 2022 |        0.0672 |  0.1223 |     0.3199 |     4.012  |               0.0135 |     0.1274 |       0.357  |
        | 2023 |        0.0332 |  0.1667 |    -0.0672 |     4.4211 |               0.0082 |     0.0412 |       0.203  |
        | 2024 |        0.0341 |  0.2181 |     0.4479 |     6.516  |               0.0091 |     0.0515 |       0.2268 |
        | 2025 |        0.0492 |  0.2452 |     1.1284 |    16.8074 |               0.0149 |     0.1056 |       0.3249 |
        | 2026 |        0.0391 |  0.2585 |    -0.2958 |     4.5239 |               0.0119 |     0.0717 |       0.2677 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        rounding = rounding if rounding else self._rounding

        risk_metrics = {
            "Value at Risk": self.get_value_at_risk(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Conditional Value at Risk": self.get_conditional_value_at_risk(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Entropic Value at Risk": self.get_entropic_value_at_risk(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Conditional Drawdown at Risk": self.get_conditional_drawdown_at_risk(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Tail Ratio": self.get_tail_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Maximum Drawdown": self.get_maximum_drawdown(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Maximum Drawdown Duration": self.get_maximum_drawdown_duration(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Maximum Drawdown Recovery Time": self.get_maximum_drawdown_recovery_time(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Ulcer Index": self.get_ulcer_index(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "GARCH": self.get_garch(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Skewness": self.get_skewness(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Kurtosis": self.get_kurtosis(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Downside Deviation": self.get_downside_deviation(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
        }

        if period != "daily":
            risk_metrics["Variance"] = self.get_variance(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            risk_metrics["Volatility"] = self.get_volatility(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            risk_metrics["Excess Volatility"] = self.get_excess_volatility(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            risk_metrics["Mean Absolute Deviation"] = self.get_mean_absolute_deviation(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            risk_metrics["Coefficient of Variation"] = (
                self.get_coefficient_of_variation(
                    period=period,
                    rounding=rounding,
                    growth=growth,
                    lag=lag,
                    standardize=standardize,
                )
            )

        risk_metrics = pd.concat(risk_metrics, axis=1)

        if len(self._tickers) == 1:
            risk_metrics = risk_metrics.xs(self._tickers[0], level=1, axis=1)

        return risk_metrics

    @handle_portfolio
    @handle_errors
    def get_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = True,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        distribution: str = "historic",
        threshold_percentile: float = 0.95,
    ):
        """
        Calculate the Value at Risk (VaR) of an investment portfolio or asset's returns.

        Value at Risk (VaR) is a risk management metric that quantifies the maximum potential loss
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The VaR is calculated as the quantile of the return distribution, representing the loss threshold
        that is not expected to be exceeded with a given confidence level (e.g., 5% for alpha=0.05).

        Also known as: VaR, maximum expected loss, portfolio loss risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for VaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate VaR within the specified period or for the entire
            period. Thus whether to look at the VaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, VaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day VaR). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the VaR calculations (historic, gaussian, cf,
            studentt or evt). Defaults to "historic".
            threshold_percentile (float, optional): Only used when `distribution` is "evt". The percentile
            of losses above which the Generalized Pareto Distribution is fitted. Defaults to 0.95.

        Returns:
            pd.Series: VaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_value_at_risk()
        ```

        Which returns:

        |      |    AMZN |    TSLA |
        |:-----|--------:|--------:|
        | 2012 | -0.0244 | -0.0343 |
        | 2013 | -0.0204 | -0.0537 |
        | 2014 | -0.0312 | -0.0423 |
        | 2015 | -0.0208 | -0.0422 |
        | 2016 | -0.0288 | -0.0394 |
        | 2017 | -0.0154 | -0.0345 |
        | 2018 | -0.0416 | -0.0503 |
        | 2019 | -0.0232 | -0.0492 |
        | 2020 | -0.0369 | -0.0741 |
        | 2021 | -0.0252 | -0.0499 |
        | 2022 | -0.0518 | -0.0713 |
        | 2023 | -0.0271 | -0.054  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        if rolling:
            returns = self._historical_data[period]["Return"]
            value_at_risk = var_model.get_rolling_var_historic(returns, alpha, rolling)
        else:
            returns = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )

            if distribution == "historic":
                value_at_risk = var_model.get_var_historic(returns, alpha)
            elif distribution == "gaussian":
                value_at_risk = var_model.get_var_gaussian(returns, alpha)
            elif distribution == "cf":
                value_at_risk = var_model.get_var_gaussian(returns, alpha, True)
            elif distribution == "studentt":
                value_at_risk = var_model.get_var_studentt(returns, alpha)
            elif distribution == "evt":
                value_at_risk = var_model.get_var_evt(
                    returns, alpha, threshold_percentile
                )
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, cf, studentt or evt."
                )

        if rolling or within_period:
            value_at_risk = value_at_risk.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=value_at_risk,
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
    def get_conditional_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = True,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        distribution: str = "historic",
    ):
        """
        Calculate the Conditional Value at Risk (CVaR) of an investment portfolio or asset's returns.

        Conditional Value at Risk (CVaR) is a risk management metric that quantifies the loss in the worst % of cases of
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The CVaR is calculated as the expected loss given that the loss threshold (VaR) with a given confidence
        level (e.g., 5% for alpha=0.05) is excceeded.

        Also known as: CVaR, expected shortfall, ES, tail risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for CVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, CVaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day CVaR). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the CVaR calculations (historic, gaussian, studentt, laplace
            or logistic). Defaults to "historic".

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates CVaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CVaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_conditional_value_at_risk()
        ```

        Which returns:

        |      |    AMZN |    TSLA |
        |:-----|--------:|--------:|
        | 2012 | -0.0302 | -0.0622 |
        | 2013 | -0.0323 | -0.0807 |
        | 2014 | -0.0552 | -0.0607 |
        | 2015 | -0.0318 | -0.053  |
        | 2016 | -0.0456 | -0.0604 |
        | 2017 | -0.0236 | -0.0483 |
        | 2018 | -0.0540 | -0.0746 |
        | 2019 | -0.0327 | -0.0758 |
        | 2020 | -0.0510 | -0.1262 |
        | 2021 | -0.0327 | -0.0683 |
        | 2022 | -0.0685 | -0.0914 |
        | 2023 | -0.0397 | -0.0747 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        if rolling:
            returns = self._historical_data[period]["Return"]
            conditional_value_at_risk = cvar_model.get_rolling_cvar_historic(
                returns, alpha, rolling
            )
        else:
            returns = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )

            if distribution == "historic":
                conditional_value_at_risk = cvar_model.get_cvar_historic(returns, alpha)
            elif distribution == "gaussian":
                conditional_value_at_risk = cvar_model.get_cvar_gaussian(returns, alpha)
            elif distribution == "studentt":
                conditional_value_at_risk = var_model.get_var_studentt(returns, alpha)
            elif distribution == "laplace":
                conditional_value_at_risk = cvar_model.get_cvar_laplace(returns, alpha)
            elif distribution == "logistic":
                conditional_value_at_risk = cvar_model.get_cvar_logistic(returns, alpha)
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, studentt, laplace or logistic."
                )

        if rolling or within_period:
            conditional_value_at_risk = conditional_value_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=conditional_value_at_risk,
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
    def get_entropic_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Entropic Value at Risk (EVaR) of an investment portfolio or asset's returns.

        Entropic Value at Risk (EVaR) is a risk management metric that quantifies upper bound for the value
        at risk (VaR) and the conditional value at risk (CVaR) over a specified time horizon and confidence
        level. EVaR is obtained from the Chernoff inequality. It provides insights into the downside risk
        associated with an investment and helps investors make informed decisions about risk tolerance.

        The EVaR is calculated as the upper bound of VaR and CVaR with a given confidence level (e.g., 5% for
        alpha=0.05).

        Also known as: EVaR.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for EVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate EVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: EVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates EVaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of EVaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_entropic_value_at_risk()
        ```

        Which returns:

        |      |    AMZN |    TSLA |   SPY |
        |:-----|--------:|--------:|--------:|
        | 2012 | -0.0392 | -0.0604 | -0.0177 |
        | 2013 | -0.0377 | -0.0928 | -0.0152 |
        | 2014 | -0.0481 | -0.0689 | -0.0162 |
        | 2015 | -0.046  | -0.0564 | -0.0227 |
        | 2016 | -0.043  | -0.0571 | -0.0188 |
        | 2017 | -0.0289 | -0.0501 | -0.0091 |
        | 2018 | -0.0518 | -0.085  | -0.0252 |
        | 2019 | -0.0327 | -0.071  | -0.0173 |
        | 2020 | -0.054  | -0.1211 | -0.0497 |
        | 2021 | -0.0352 | -0.0782 | -0.0183 |
        | 2022 | -0.0758 | -0.1012 | -0.0362 |
        | 2023 | -0.0471 | -0.0793 | -0.0188 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = (
            self._within_historical_data[period]["Return"]
            if within_period
            else self._historical_data[period]["Return"]
        )

        entropic_value_at_risk = evar_model.get_evar_gaussian(returns, alpha)

        if within_period:
            entropic_value_at_risk = entropic_value_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=entropic_value_at_risk,
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
    def get_conditional_drawdown_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = True,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Conditional Drawdown at Risk (CDaR) of an investment portfolio or asset's returns.

        Conditional Drawdown at Risk (CDaR) extends the concept of Value at Risk and Conditional Value at
        Risk to the drawdown series instead of the return series. It is calculated as the average of the
        worst drawdowns that exceed the Drawdown at Risk (DaR), i.e. the alpha-quantile of the drawdown
        distribution, giving insight into the depth of the most severe drawdowns an investment portfolio or
        asset could experience.

        Also known as: CDaR.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for CDaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CDaR within the specified period or for the entire
            period. Thus whether to look at the CDaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, CDaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CDaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: CDaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates CDaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CDaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_conditional_drawdown_at_risk()
        ```

        Which returns:

        |      |    AMZN |    TSLA |   Benchmark |
        |:-----|--------:|--------:|------------:|
        | 2021 | -0.1325 | -0.3407 |     -0.0437 |
        | 2022 | -0.499  | -0.6603 |     -0.2424 |
        | 2023 | -0.1756 | -0.2867 |     -0.0832 |
        | 2024 | -0.1612 | -0.3662 |     -0.0576 |
        | 2025 | -0.2721 | -0.4558 |     -0.1459 |
        | 2026 | -0.1869 | -0.2267 |     -0.072  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        if rolling:
            returns = self._historical_data[period]["Return"]
            conditional_drawdown_at_risk = (
                risk_model.get_rolling_conditional_drawdown_at_risk(
                    returns, alpha, rolling
                )
            )
        else:
            returns = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )

            conditional_drawdown_at_risk = risk_model.get_conditional_drawdown_at_risk(
                returns, alpha
            )

        if rolling or within_period:
            conditional_drawdown_at_risk = conditional_drawdown_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=conditional_drawdown_at_risk,
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
    def get_tail_ratio(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = True,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Tail Ratio of an investment portfolio or asset's returns.

        The Tail Ratio compares the size of the right (gain) tail to the left (loss) tail of the
        return distribution, calculated as the absolute value of the (1 - alpha)-th percentile of
        returns divided by the absolute value of the alpha-th percentile of returns. A Tail Ratio
        above 1 indicates that best-case gains outsize worst-case losses.

        Also known as: gain-to-pain tail ratio.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The percentile used to define each tail (e.g., 0.05 uses the 5th and
            95th percentile). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate the Tail Ratio within the specified period or
            for the entire period. Thus whether to look at the Tail Ratio within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Tail
            Ratio is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Tail Ratio values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Tail Ratio values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Tail
        Ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Tail Ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_tail_ratio()
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 | 0.8591 | 1.0173 |      1.0688 |
        | 2022 | 0.8406 | 1.0049 |      0.9524 |
        | 2023 | 1.2609 | 1.1958 |      1.0432 |
        | 2024 | 1.1246 | 1.1942 |      0.9049 |
        | 2025 | 0.9359 | 1.0702 |      0.93   |
        | 2026 | 1.0012 | 0.9592 |      0.8828 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        if rolling:
            returns = self._historical_data[period]["Return"]
            tail_ratio = risk_model.get_rolling_tail_ratio(returns, alpha, rolling)
        else:
            returns = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )

            tail_ratio = risk_model.get_tail_ratio(returns, alpha)

        if rolling or within_period:
            tail_ratio = tail_ratio.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=tail_ratio,
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
    def get_maximum_drawdown(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Maximum Drawdown (MDD) of an investment portfolio or asset's returns.

        Maximum Drawdown (MDD) is a risk management metric that quantifies the largest historical loss of
        n investment portfolio or asset experienced over a specified time horizon. It provides insights into
        the downside risk associated with an investment and helps investors make informed decisions about
        risk tolerance.

        Also known as: max drawdown, peak-to-trough decline.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for CVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Maximum Drawdown values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates MMD for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of MMD values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_maximum_drawdown()
        ```

        Which returns:

        |      |    AMZN |    TSLA |
        |:-----|--------:|--------:|
        | 2012 | -0.1570 | -0.1601 |
        | 2013 | -0.1259 | -0.3768 |
        | 2014 | -0.2948 | -0.3085 |
        | 2015 | -0.1371 | -0.2669 |
        | 2016 | -0.2432 | -0.357  |
        | 2017 | -0.1085 | -0.2227 |
        | 2018 | -0.3410 | -0.3399 |
        | 2019 | -0.1561 | -0.4847 |
        | 2020 | -0.2274 | -0.6063 |
        | 2021 | -0.1457 | -0.3625 |
        | 2022 | -0.5198 | -0.7272 |
        | 2023 | -0.1964 | -0.2823 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = (
            self._within_historical_data[period]["Return"]
            if within_period
            else self._historical_data[period]["Return"]
        )

        maximum_drawdown = risk_model.get_max_drawdown(returns)

        if within_period:
            maximum_drawdown = maximum_drawdown.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=maximum_drawdown,
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
    def get_maximum_drawdown_duration(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Maximum Drawdown Duration of an investment portfolio or asset's returns.

        The Maximum Drawdown Duration is the number of periods between the peak and the lowest point
        of the largest drawdown, giving insight into how long the worst loss of value took to unfold.

        Also known as: drawdown length.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            within_period (bool, optional): Whether to calculate the duration within the specified period or
            for the entire period. Thus whether to look at the duration within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the duration values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Maximum Drawdown Duration values, in number of periods, with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Maximum
        Drawdown Duration for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the duration values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_maximum_drawdown_duration()
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 |     30 |     28 |          21 |
        | 2022 |    248 |    247 |         195 |
        | 2023 |     25 |     73 |          63 |
        | 2024 |     23 |     76 |          14 |
        | 2025 |     52 |     57 |          34 |
        | 2026 |     24 |     64 |          43 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = (
            self._within_historical_data[period]["Return"]
            if within_period
            else self._historical_data[period]["Return"]
        )

        maximum_drawdown_duration = risk_model.get_max_drawdown_duration(returns)

        if within_period:
            maximum_drawdown_duration = maximum_drawdown_duration.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=maximum_drawdown_duration,
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
    def get_maximum_drawdown_recovery_time(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Maximum Drawdown Recovery Time of an investment portfolio or asset's returns.

        The Maximum Drawdown Recovery Time is the number of periods it takes for the cumulative return
        to reach a new high after the lowest point of the largest drawdown. If the drawdown has not yet been
        recovered from within the selected period, this returns NaN.

        Also known as: time to recovery, drawdown recovery.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            within_period (bool, optional): Whether to calculate the recovery time within the specified period
            or for the entire period. Thus whether to look at the recovery time within a specific year (if
            period = 'yearly') or look at the entirety of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the recovery time values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Maximum Drawdown Recovery Time values, in number of periods, with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Maximum
        Drawdown Recovery Time for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the recovery time values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_maximum_drawdown_recovery_time()
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 |    nan |    159 |          13 |
        | 2022 |    nan |    nan |         nan |
        | 2023 |     46 |    nan |          24 |
        | 2024 |     66 |     51 |          32 |
        | 2025 |    135 |    114 |          55 |
        | 2026 |     40 |    nan |          11 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = (
            self._within_historical_data[period]["Return"]
            if within_period
            else self._historical_data[period]["Return"]
        )

        maximum_drawdown_recovery_time = risk_model.get_max_drawdown_recovery_time(
            returns
        )

        if within_period:
            maximum_drawdown_recovery_time = maximum_drawdown_recovery_time.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=maximum_drawdown_recovery_time,
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
    def get_ulcer_index(
        self,
        period: str | None = None,
        rolling: int = 14,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        The Ulcer Index is a financial metric used to assess the risk and volatility of an
        investment portfolio or asset. Developed by Peter Martin in the 1980s, the Ulcer Index
        is particularly useful for evaluating the downside risk and drawdowns associated with investments.

        The Ulcer Index differs from traditional volatility measures like standard deviation or variance
        because it focuses on the depth and duration of drawdowns rather than the dispersion of
        returns.

        The formula is a follows:

        Ulcer Index = SQRT(SUM[(Pn / Highest High)^2] / n)

        Also known as: UI, drawdown risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            rolling (int, optional): The rolling period to use for the calculation. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the UI values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: UI values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates UI for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_ulcer_index()
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2012 | 0.0497 | 0.0454 |      0.0234 |
        | 2013 | 0.035  | 0.0829 |      0.0142 |
        | 2014 | 0.0659 | 0.0746 |      0.0174 |
        | 2015 | 0.0273 | 0.0624 |      0.0238 |
        | 2016 | 0.0519 | 0.0799 |      0.0151 |
        | 2017 | 0.0241 | 0.0616 |      0.0067 |
        | 2018 | 0.0619 | 0.0892 |      0.0356 |
        | 2019 | 0.0373 | 0.0839 |      0.016  |
        | 2020 | 0.0536 | 0.1205 |      0.0594 |
        | 2021 | 0.0427 | 0.085  |      0.0136 |
        | 2022 | 0.1081 | 0.1373 |      0.0492 |
        | 2023 | 0.0475 | 0.0815 |      0.0186 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._within_historical_data[period]["Return"]

        ulcer_index = risk_model.get_ui(returns, rolling)

        ulcer_index = ulcer_index.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=ulcer_index,
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
    def get_garch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates volatility forecasts based on the GARCH model.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is
        for instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH
        models.

        Also known as: GARCH, volatility clustering, conditional heteroscedasticity.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "weekly".
            t (int, optional): Time steps to calculate GARCH for.
            optimization_t (int, optional): Time steps to optimize GRACH for. It is only used if no weights are given.
            within_period (bool, optional): Whether to calculate GARCH within the specified period or for the entire
            period. Thus whether to look at the GARCH within a specific year (if period = 'yearly') or look at the
            entirety of all years. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the GARCH values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: GARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates GARCH for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of GARCH values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_garch()
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2012Q4 | 0      |  0     |      0      |
        | 2013Q1 | 0.0147 |  0.214 |      0.0008 |
        | 2013Q2 | 0.0223 |  0.214 |      0.0024 |
        | 2013Q3 | 0.0262 |  0.214 |      0.0029 |
        | 2013Q4 | 0.0282 |  0.214 |      0.0034 |
        | 2014Q1 | 0.0293 |  0.214 |      0.0045 |
        | 2014Q2 | 0.0298 |  0.214 |      0.0045 |
        | 2014Q3 | 0.03   |  0.214 |      0.0047 |
        | 2014Q4 | 0.0302 |  0.214 |      0.0047 |
        | 2015Q1 | 0.0303 |  0.214 |      0.0048 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = (
            (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )
            .dropna()
            .replace(0, 1e-100)
        )

        garch_sigma_2 = garch_model.get_garch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
        )

        garch_sigma_2 = garch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=garch_sigma_2,
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
    def get_garch_forecast(
        self,
        period: str | None = None,
        time_steps: int = 10,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates sigma_2 forecasts.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is
        for instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH
        models.

        The forecasting with GARCH is done with the following formula:

        - sigma_l ** 2 + (sigma_t ** 2 - sigma_l ** 2) * (alpha + beta) ** (t - 1)

        For more information about the method, see the following book:

        - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "quarterly".
            t (int, optional): Time steps to calculate GARCH and to forecast sigma_2 values for.
            within_period (bool, optional): Whether to calculate GARCH within each specified period or all
            at once. Thus whether to look at the GARCH within each specific year (if period = 'yearly') or
            look at the entirety of all years. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the GARCH values over time. Defaults to
            False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the sigma_2
        forecast for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the forecasted simga_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_garch_forecast()
        ```

        Which returns:

        |      |   AMZN |     TSLA |   Benchmark |
        |:-----|-------:|---------:|------------:|
        | 2024 | 0      |    0     |      0      |
        | 2025 | 0      |    0     |      0      |
        | 2026 | 0.4156 |  252.921 |      0.0058 |
        | 2027 | 0.7897 |  480.55  |      0.011  |
        | 2028 | 1.1263 |  685.417 |      0.0156 |
        | 2029 | 1.4293 |  869.796 |      0.0198 |
        | 2030 | 1.702  | 1035.74  |      0.0236 |
        | 2031 | 1.9474 | 1185.09  |      0.027  |
        | 2032 | 2.1683 | 1319.5   |      0.0301 |
        | 2033 | 2.3671 | 1440.47  |      0.0329 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = (
            (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )
            .dropna()
            .replace(0, 1e-100)
        )

        sigma_2_forecast = garch_model.get_garch_forecast(
            returns, None, time_steps
        ).dropna()

        period_symbol = (
            "W"
            if period == "weekly"
            else (
                "ME" if period == "monthly" else "QE" if period == "quarterly" else "YE"
            )
        )
        period_index = pd.PeriodIndex(
            pd.date_range(
                start=returns.index[-1].to_timestamp(),
                periods=time_steps + 1,
                freq=period_symbol,
            )
        )

        sigma_2_forecast.index = period_index[1:]

        return finalize_dataset(
            dataset=sigma_2_forecast,
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
    def get_skewness(
        self,
        period: str | None = None,
        within_period: bool = True,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Skewness of an investment portfolio or asset's returns.

        Skewness is a statistical measure used in finance to assess the asymmetry in the distribution of
        returns for an investment portfolio or asset over a defined period. It offers valuable insights
        into the shape of the return distribution, indicating whether returns are skewed towards the
        positive or negative side of the mean. Skewness is a crucial tool for investors and analysts seeking
        to understand the potential risk and return characteristics of an investment, aiding in the assessment
        of the distribution's tails and potential outliers. It provides a means to gauge the level of
        skew in returns, enabling more informed investment decisions and risk management strategies.

        Also known as: return distribution asymmetry, tail skew.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for CVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Skewness is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates Skew for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_skewness()
        ```

        Which returns:

        |      |    MSFT |    AAPL |    TSLA |
        |:-----|--------:|--------:|--------:|
        | 2019 | -0.194  | -0.9216 | -0.0646 |
        | 2020 | -0.0747 | -0.0586 | -0.1824 |
        | 2021 | -0.0194 | -0.0716 |  0.6572 |
        | 2022 |  0.1478 |  0.3164 | -0.0263 |
        | 2023 |  0.5252 |  0.0318 | -0.0972 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        if rolling:
            returns = self._historical_data[period]["Return"]
            skewness = risk_model.get_rolling_skewness(returns, rolling)
        else:
            returns = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )

            skewness = risk_model.get_skewness(returns)

        if rolling or within_period:
            skewness = skewness.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=skewness,
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
    def get_kurtosis(
        self,
        period: str | None = None,
        within_period: bool = True,
        fisher: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Kurtosis of an investment portfolio or asset's returns.

        Kurtosis is a statistical measure used in finance to evaluate the shape of the probability
        distribution of returns for an investment portfolio or asset over a defined time period.
        It assesses the "tailedness" of the return distribution, indicating whether returns have
        fatter or thinner tails compared to a normal distribution. Kurtosis plays a critical role
        in risk assessment by revealing the potential presence of extreme outliers or the likelihood
        of heavy tails in the return data. This information aids investors and analysts in
        understanding the degree of risk associated with an investment and assists in making
        more informed decisions regarding risk tolerance. In essence, kurtosis serves as a valuable
        tool for comprehending the distribution characteristics of returns, offering insights
        into the potential for rare but significant events in the financial markets.

        Also known as: tail heaviness, fat tails, leptokurtosis.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at
            the entirety of all years. Defaults to True.
            fisher (bool, optional): Whether to use Fisher's definition of kurtosis (kurtosis = 0.0
            for a normal distribution).
            rolling (int, optional): The rolling window size to use for the calculation. If set, Kurtosis is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time.
            efaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL", "TSLA"]], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_kurtosis()
        ```

        Which returns:

        |      |   MSFT |    AAPL |   TSLA |
        |:-----|-------:|--------:|-------:|
        | 2019 | 4.0972 | 10.0741 | 9.128  |
        | 2020 | 9.2914 |  6.6307 | 5.2189 |
        | 2021 | 3.3152 |  3.3352 | 7.3197 |
        | 2022 | 3.852  |  4.0085 | 3.3553 |
        | 2023 | 4.2908 |  4.4568 | 4.07   |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        if rolling:
            returns = self._historical_data[period]["Return"]
            kurtosis = risk_model.get_rolling_kurtosis(returns, rolling, fisher=fisher)
        else:
            returns = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )

            kurtosis = risk_model.get_kurtosis(returns, fisher=fisher)

        if rolling or within_period:
            kurtosis = kurtosis.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=kurtosis,
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
    def get_variance(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Variance of an investment portfolio or asset's returns for a
        given period based on the daily historical returns.

        Variance measures the spread or dispersion of returns around the mean. A higher
        Variance indicates more variability in the returns, while a lower Variance suggests
        that the returns are closer to the mean.

        The daily Variance is scaled to the given period by multiplying it with the number
        of trading days within that period (e.g. 252 / 52 for weekly).

        Also known as: dispersion, spread.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Variance is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Variance) instead of one value per `period`.
            Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Variance values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Variance values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Variance for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Variance values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_variance(period="yearly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.058  | 0.2999 |      0.0172 |
        | 2022   | 0.2508 | 0.4446 |      0.0589 |
        | 2023   | 0.109  | 0.2922 |      0.0174 |
        | 2024   | 0.0789 | 0.4032 |      0.0158 |
        | 2025   | 0.1184 | 0.4031 |      0.0379 |
        | 2026   | 0.0999 | 0.1859 |      0.02   |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            variance = risk_model.get_rolling_variance(period_returns, period, rolling)
        else:
            returns = self._historical_data["daily"]["Return"]
            variance = risk_model.get_variance(returns, period)

        variance = variance.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=variance,
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
    def get_volatility(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Volatility of an investment portfolio or asset's returns for a
        given period based on the daily historical returns.

        Volatility measures the amount of dispersion or variability in returns. It is the
        square root of the Variance. A higher Volatility indicates greater variability, while
        a lower Volatility suggests that returns are closer to the mean.

        The daily Volatility is scaled to the given period by multiplying it with the square
        root of the number of trading days within that period (e.g. SQRT(252 / 52) for weekly).

        Also known as: standard deviation of returns.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Volatility is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Volatility) instead of one value per `period`.
            Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Volatility values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Volatility for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Volatility values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_volatility(period="yearly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.2409 | 0.5476 |      0.131  |
        | 2022   | 0.5008 | 0.6668 |      0.2427 |
        | 2023   | 0.3302 | 0.5406 |      0.1318 |
        | 2024   | 0.2809 | 0.635  |      0.1258 |
        | 2025   | 0.3442 | 0.6349 |      0.1948 |
        | 2026   | 0.3161 | 0.4312 |      0.1414 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            volatility = risk_model.get_rolling_volatility(
                period_returns, period, rolling
            )
        else:
            returns = self._historical_data["daily"]["Return"]
            volatility = risk_model.get_volatility(returns, period)

        volatility = volatility.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=volatility,
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
    def get_excess_volatility(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Excess Volatility of an investment portfolio or asset's returns for a
        given period based on the daily historical returns.

        Excess Volatility is the Volatility of the Excess Return, i.e. the daily return minus
        the risk free rate, scaled to the given period in the same way as the Volatility.

        Also known as: standard deviation of excess returns.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Excess Volatility is calculated over a rolling window of this many periods (e.g.
            period='monthly' and rolling=6 gives the rolling 6-month Excess Volatility) instead of
            one value per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Excess Volatility values
            over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Excess Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Excess Volatility for
        the specified `period` for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of Excess Volatility values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_excess_volatility(period="yearly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.2414 | 0.5483 |      0.1333 |
        | 2022   | 0.5207 | 0.686  |      0.2663 |
        | 2023   | 0.3421 | 0.5535 |      0.1527 |
        | 2024   | 0.2841 | 0.6346 |      0.1336 |
        | 2025   | 0.3435 | 0.635  |      0.1946 |
        | 2026   | 0.3196 | 0.4331 |      0.1446 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            risk_free_rate = self._risk_free_rate_data[period]
            excess_volatility = risk_model.get_rolling_excess_volatility(
                period_returns, risk_free_rate, period, rolling
            )
        else:
            returns = self._historical_data["daily"]["Return"]
            risk_free_rate = self._risk_free_rate_data["daily"]

            excess_volatility = risk_model.get_excess_volatility(
                returns, risk_free_rate, period
            )

        excess_volatility = excess_volatility.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=excess_volatility,
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
    def get_downside_deviation(
        self,
        period: str | None = None,
        minimum_acceptable_return: float = 0.0,
        within_period: bool = True,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Downside Deviation of an investment portfolio or asset's returns.

        The Downside Deviation, also known as semi-deviation, is the standard deviation of only the
        returns that fall below a minimum acceptable return (MAR), isolating the volatility of negative
        outcomes from the volatility of the overall return distribution. It underlies risk-adjusted
        return measures such as the Sortino Ratio and the Omega Ratio.

        Also known as: semi-deviation, downside risk, downside volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used as the
            threshold below which returns are considered downside. Defaults to 0.0.
            within_period (bool, optional): Whether to calculate the Downside Deviation within the specified
            period or for the entire period. Thus whether to look at the Downside Deviation within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Downside
            Deviation is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Downside Deviation values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Downside Deviation values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Downside Deviation for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Downside Deviation values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_downside_deviation()
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 | 0.0106 | 0.0215 |      0.0058 |
        | 2022 | 0.0202 | 0.0283 |      0.0095 |
        | 2023 | 0.0129 | 0.0217 |      0.005  |
        | 2024 | 0.0118 | 0.0227 |      0.006  |
        | 2025 | 0.0146 | 0.0257 |      0.0096 |
        | 2026 | 0.0123 | 0.0165 |      0.0061 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        if rolling:
            returns = self._historical_data[period]["Return"]
            downside_deviation = risk_model.get_rolling_downside_deviation(
                returns, rolling, minimum_acceptable_return
            )
        else:
            returns = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            )

            downside_deviation = risk_model.get_downside_deviation(
                returns, minimum_acceptable_return
            )

        if rolling or within_period:
            downside_deviation = downside_deviation.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=downside_deviation,
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
    def get_mean_absolute_deviation(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Mean Absolute Deviation (MAD) of an investment portfolio or asset's
        returns for a given period based on the daily historical returns.

        MAD measures the average absolute distance of each return from the mean return. Unlike
        Variance and Volatility, it does not square the deviations, making it less sensitive to
        outliers.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the MAD values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Mean Absolute Deviation values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the MAD for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of MAD values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_mean_absolute_deviation(period="yearly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.0114 | 0.0246 |      0.0062 |
        | 2022   | 0.0235 | 0.032  |      0.0119 |
        | 2023   | 0.0156 | 0.0255 |      0.0065 |
        | 2024   | 0.0132 | 0.0286 |      0.0058 |
        | 2025   | 0.015  | 0.0292 |      0.0074 |
        | 2026   | 0.0157 | 0.0216 |      0.0067 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        returns = self._historical_data["daily"]["Return"]
        mean_absolute_deviation = risk_model.get_mean_absolute_deviation(
            returns, period
        )

        mean_absolute_deviation = mean_absolute_deviation.loc[
            self._start_date : self._end_date
        ]
        mean_absolute_deviation = mean_absolute_deviation.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=mean_absolute_deviation,
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
    def get_coefficient_of_variation(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Coefficient of Variation (CV) of an investment portfolio or asset's
        returns for a given period based on the daily historical returns.

        The Coefficient of Variation is the ratio of the standard deviation to the mean of
        returns, which normalizes dispersion relative to the average return. This makes it
        useful for comparing the relative volatility of assets with different average returns,
        which a raw standard deviation cannot do.

        Also known as: relative standard deviation.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly,
            quarterly, or yearly). Defaults to "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CV values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Coefficient of Variation values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the CV for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CV values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_coefficient_of_variation(period="yearly")
        ```

        Which returns:

        | Date   |     AMZN |      TSLA |   Benchmark |
        |:-------|---------:|----------:|------------:|
        | 2021   |  73.121  |   15.7477 |      8.3938 |
        | 2022   | -14.1417 |  -12.7544 |    -20.4791 |
        | 2023   |   8.0356 |   10.0506 |      9.1833 |
        | 2024   |  10.9398 |   14.7623 |      9.1557 |
        | 2025   |  49.9543 |   32.8037 |     18.0122 |
        | 2026   |  31.132  | -163.952  |     11.047  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        returns = self._historical_data["daily"]["Return"]
        coefficient_of_variation = risk_model.get_coefficient_of_variation(
            returns, period
        )

        coefficient_of_variation = coefficient_of_variation.loc[
            self._start_date : self._end_date
        ]
        coefficient_of_variation = coefficient_of_variation.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=coefficient_of_variation,
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
    def get_ewma_volatility(
        self,
        lambda_: float = 0.94,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the exponentially weighted moving average (EWMA) Volatility of an
        investment portfolio or asset's daily returns, following the RiskMetrics
        methodology.

        Unlike a fixed-window rolling Volatility, EWMA Volatility weights recent
        observations more heavily than older ones, so it reacts faster to changes in the
        underlying volatility regime. It is a simpler, more interpretable alternative to a
        full GARCH fit.

        The formula is as follows:

        - EWMA Variance(t) = lambda * EWMA Variance(t-1) + (1 - lambda) * Return(t-1) ** 2

        Also known as: RiskMetrics volatility, exponentially weighted volatility.

        Args:
            lambda_ (float, optional): The decay factor. Higher values weight the past
            more heavily (slower to react), lower values weight recent returns more
            heavily (faster to react). RiskMetrics uses 0.94 for daily data. Defaults to 0.94.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the EWMA Volatility values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Daily EWMA Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the EWMA Volatility for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the EWMA Volatility values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_ewma_volatility()
        ```

        Which returns:

        | Date       |   AMZN |   TSLA |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2026-06-22 | 0.0229 | 0.0279 |      0.0099 |
        | 2026-06-23 | 0.0223 | 0.0304 |      0.0103 |
        | 2026-06-24 | 0.0216 | 0.0296 |      0.01   |
        | 2026-06-25 | 0.022  | 0.0287 |      0.0097 |
        | 2026-06-26 | 0.0225 | 0.0281 |      0.0096 |
        | 2026-06-29 | 0.0234 | 0.0345 |      0.0101 |
        | 2026-06-30 | 0.0228 | 0.0338 |      0.01   |
        | 2026-07-01 | 0.0224 | 0.0328 |      0.0097 |
        | 2026-07-02 | 0.0218 | 0.037  |      0.0094 |
        | 2026-07-06 | 0.0211 | 0.0395 |      0.0093 |
        """
        returns = self._historical_data["daily"]["Return"]
        ewma_volatility = risk_model.get_ewma_volatility(returns, lambda_)

        ewma_volatility = ewma_volatility.loc[self._start_date : self._end_date]
        ewma_volatility = ewma_volatility.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=ewma_volatility,
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
    def get_autocorrelation(
        self,
        lags: int = 10,
        rounding: int | None = None,
    ):
        """
        Calculate the Autocorrelation Function (ACF) of each asset's daily returns for a
        range of lags.

        The ACF measures the correlation between a return series and a lagged version of
        itself. A significant ACF at a given lag indicates that returns are not fully
        independent over time, which is relevant for assessing return predictability and
        volatility clustering (as opposed to a trading-signal use case, which is why this
        lives in the Risk module rather than Technicals).

        Args:
            lags (int, optional): The number of lags to calculate the ACF for. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.

        Returns:
            pd.DataFrame: The ACF value for each lag (rows) and each asset (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_autocorrelation()
        ```

        Which returns:

        |    |    AMZN |    TSLA |   Benchmark |
        |---:|--------:|--------:|------------:|
        |  1 | -0.0109 | -0.0306 |     -0.0366 |
        |  2 | -0.0013 |  0.0121 |      0.0066 |
        |  3 | -0.0216 |  0.0006 |     -0.0571 |
        |  4 |  0.01   |  0.0117 |     -0.0344 |
        |  5 | -0.0063 | -0.0302 |      0.0002 |
        |  6 |  0.0018 |  0.0298 |     -0.022  |
        |  7 | -0.0451 |  0.0209 |     -0.0076 |
        |  8 | -0.0281 |  0.0092 |     -0.013  |
        |  9 |  0.0017 |  0.0675 |      0.0529 |
        | 10 | -0.0162 | -0.0293 |     -0.0133 |
        """
        returns = self._historical_data["daily"]["Return"]

        autocorrelation = returns.apply(
            lambda column: risk_model.get_autocorrelation(column, lags=lags)
        )

        return autocorrelation.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_hurst_exponent(
        self,
        max_lag: int = 20,
        rounding: int | None = None,
    ):
        """
        Calculate the Hurst Exponent of each asset's daily returns, a measure of
        long-term memory that indicates whether a series is mean-reverting, trending,
        or a random walk.

        The Hurst Exponent (H) is interpreted as follows:

        - H < 0.5: the series is mean-reverting (anti-persistent).
        - H = 0.5: the series is a random walk (no memory).
        - H > 0.5: the series is trending (persistent).

        Args:
            max_lag (int, optional): The maximum lag to use when estimating the exponent.
                Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.

        Returns:
            pd.Series: The estimated Hurst Exponent for each asset.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_hurst_exponent()
        ```

        Which returns:

        |           |       0 |
        |:----------|--------:|
        | AMZN      | -0.0082 |
        | TSLA      |  0.0099 |
        | Benchmark | -0.0077 |
        """
        returns = self._historical_data["daily"]["Return"]

        hurst_exponent = returns.apply(
            lambda column: risk_model.get_hurst_exponent(column, max_lag=max_lag)
        )

        return hurst_exponent.round(rounding if rounding else self._rounding)
