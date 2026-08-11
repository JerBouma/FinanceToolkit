"""Risk Module"""

__docformat__ = "google"

import itertools
import warnings

import numpy as np
import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import (
    backtesting_model,
    copula_model,
    covar_model,
    cvar_model,
    evar_model,
    garch_model,
    market_liquidity_model,
    realized_volatility_model,
    risk_model,
    var_model,
)
from financetoolkit.risk.helpers import determine_within_historical_data
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.statistics_model import (
    convert_annualized_rate_to_period,
    finalize_dataset,
)

# Division by zero is normal in these calculations, not a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

MINIMUM_TICKERS_FOR_ALL_PAIRS = 2

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals
# pylint: disable=too-many-boolean-expressions


def _as_scalar(value: float | np.ndarray | pd.Series) -> float:
    """
    Reduces a Value at Risk or Conditional Value at Risk estimate to a single float.

    A rolling apply only accepts a scalar return value, but the Student-t and Extreme
    Value Theory variants return a length one array or Series when given a Series, which
    would otherwise raise a TypeError before it reaches the backtest.

    Args:
        value (float | np.ndarray | pd.Series): the estimate to reduce.

    Returns:
        float: the estimate as a single float.
    """
    return float(np.asarray(value).reshape(-1)[0])


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

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_value_at_risk(period='yearly')
        ```

        Which returns:

        | Date   |    AAPL |    MSFT |
        |:-------|--------:|--------:|
        | 2020   | -0.0448 | -0.0417 |
        | 2021   | -0.0256 | -0.0211 |
        | 2022   | -0.0373 | -0.0385 |
        """
        self._historical_data = historical_data
        # The risk free rate is quoted as an annualized yield, so it is converted to the
        # matching frequency. Without this, a daily return would have a full year of
        # risk free rate subtracted from it.
        self._risk_free_rate_data = {
            frequency: convert_annualized_rate_to_period(rate, frequency)
            for frequency, rate in risk_free_rate_data.items()
        }
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

        toolkit.risk.collect_all_metrics().xs("AAPL", level=1, axis=1)
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

        rounding = rounding if rounding is not None else self._rounding

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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            alpha (float, optional): The confidence level for VaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate VaR within the specified period or for the entire
            period. Thus whether to look at the VaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, VaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day VaR). Only available for
            `distribution="historic"`; see `get_var_backtest` for a rolling, out-of-sample VaR path
            under the parametric distributions. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the VaR calculations (historic, gaussian, cf,
            cornish-fisher, studentt or evt). Defaults to "historic". Note that "cf" and "cornish-fisher"
            both adjust the gaussian quantile for skewness and kurtosis, but "cornish-fisher" uses the
            more standard Cornish-Fisher expansion (see `var_model.get_var_cornish_fisher`), while "cf"
            is kept for backwards compatibility.
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

        if rolling and distribution != "historic":
            raise ValueError(
                "The rolling parameter is only available for distribution='historic'. "
                "Use get_var_backtest to obtain a rolling, out-of-sample VaR path "
                "under one of the parametric distributions instead."
            )

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
            elif distribution == "cornish-fisher":
                value_at_risk = var_model.get_var_cornish_fisher(returns, alpha)
            elif distribution == "studentt":
                value_at_risk = var_model.get_var_studentt(returns, alpha)
            elif distribution == "evt":
                value_at_risk = var_model.get_var_evt(
                    returns, alpha, threshold_percentile
                )
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, cf, cornish-fisher, "
                    "studentt or evt."
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
        threshold_percentile: float = 0.95,
    ):
        """
        Calculate the Conditional Value at Risk (CVaR) of an investment portfolio or asset's returns.

        Conditional Value at Risk (CVaR) is a risk management metric that quantifies the loss in the worst % of cases of
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The CVaR is calculated as the expected loss given that the loss threshold (VaR) with a given confidence
        level (e.g., 5% for alpha=0.05) is exceeded.

        Also known as: CVaR, expected shortfall, ES, tail risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            alpha (float, optional): The confidence level for CVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, CVaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day CVaR). Only available for
            `distribution="historic"`; see `get_acerbi_szekely_test` for a rolling, out-of-sample CVaR
            path under the parametric distributions. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the CVaR calculations (historic, gaussian, studentt, laplace,
            logistic, cornish-fisher or evt). Defaults to "historic".
            threshold_percentile (float, optional): Only used when `distribution` is "evt". The percentile
            of losses above which the Generalized Pareto Distribution is fitted. Defaults to 0.95.

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

        if rolling and distribution != "historic":
            raise ValueError(
                "The rolling parameter is only available for distribution='historic'. "
                "Use get_acerbi_szekely_test to obtain a rolling, out-of-sample CVaR "
                "path under one of the parametric distributions instead."
            )

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
                conditional_value_at_risk = cvar_model.get_cvar_studentt(returns, alpha)
            elif distribution == "laplace":
                conditional_value_at_risk = cvar_model.get_cvar_laplace(returns, alpha)
            elif distribution == "logistic":
                conditional_value_at_risk = cvar_model.get_cvar_logistic(returns, alpha)
            elif distribution == "cornish-fisher":
                conditional_value_at_risk = cvar_model.get_cvar_cornish_fisher(
                    returns, alpha
                )
            elif distribution == "evt":
                conditional_value_at_risk = cvar_model.get_cvar_evt(
                    returns, alpha, threshold_percentile
                )
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, studentt, laplace, "
                    "logistic, cornish-fisher or evt."
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
        rounding: int | None = None,
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
        an investment portfolio or asset experienced over a specified time horizon. It provides insights into
        the downside risk associated with an investment and helps investors make informed decisions about
        risk tolerance.

        Also known as: max drawdown, peak-to-trough decline.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Maximum Drawdown within the specified period
            or for the entire period. Thus whether to look at the Maximum Drawdown within a specific year
            (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Maximum Drawdown values over time.
            Defaults to False.
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
        rolling: int | None = 14,
        rounding: int | None = None,
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

        The formula is as follows:

        Ulcer Index = SQRT(SUM[((Pn - Highest High) / Highest High)^2] / n)

        Also known as: UI, drawdown risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int | None, optional): The trailing lookback window used as the high-water
                mark reference for each day's drawdown. Pass None for an expanding
                (since-inception) high-water mark instead -- this is what the "Highest High"
                in the formula above literally refers to; a fixed int window is a common,
                distinct variant (e.g. a 14-day trailing high) rather than a substitute for
                "the entire period". Note that passing `rolling=` the full length of your
                return series does NOT give you the since-inception result -- pandas only
                starts producing a rolling value once the full window is filled, so it would
                silently degenerate to just the final period's drawdown; use `rolling=None`
                instead. Defaults to 14.
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
        rounding: int | None = None,
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            time_steps (int, optional): Time steps to calculate GARCH for.
            optimization_t (int, optional): Time steps to optimize GARCH for. It is only used if no weights are given.
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

        toolkit.risk.get_garch(period="quarterly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2024Q2 | 0.0267 | 0.1602 |      0.008  |
        | 2024Q3 | 0.0266 | 0.151  |      0.0069 |
        | 2024Q4 | 0.0265 | 0.163  |      0.0064 |
        | 2025Q1 | 0.0266 | 0.1912 |      0.0056 |
        | 2025Q2 | 0.0266 | 0.1692 |      0.0052 |
        | 2025Q3 | 0.0266 | 0.1567 |      0.0065 |
        | 2025Q4 | 0.0265 | 0.1714 |      0.0066 |
        | 2026Q1 | 0.0265 | 0.1495 |      0.0058 |
        | 2026Q2 | 0.0265 | 0.1523 |      0.0054 |
        | 2026Q3 | 0.0266 | 0.1507 |      0.0083 |
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            time_steps (int, optional): Time steps to calculate GARCH and to forecast sigma_2 values for.
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
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_garch_forecast(period="quarterly")
        ```

        Which returns:

        |        |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2026Q4 | 0.0267 | 0.1703 |      0.0053 |
        | 2027Q1 | 0.0267 | 0.1703 |      0.0053 |
        | 2027Q2 | 0.0267 | 0.1738 |      0.0056 |
        | 2027Q3 | 0.0267 | 0.1745 |      0.0058 |
        | 2027Q4 | 0.0266 | 0.1747 |      0.006  |
        | 2028Q1 | 0.0266 | 0.1747 |      0.0062 |
        | 2028Q2 | 0.0266 | 0.1747 |      0.0063 |
        | 2028Q3 | 0.0266 | 0.1747 |      0.0064 |
        | 2028Q4 | 0.0266 | 0.1747 |      0.0065 |
        | 2029Q1 | 0.0266 | 0.1747 |      0.0066 |
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
    def get_garch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fits a GARCH(1, 1) model to the historical returns and returns the estimated
        Omega, Alpha and Beta parameters for each asset.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is a stochastic model
        for time series, used to model volatility clustering. A GARCH(1, 1) model expresses the
        conditional variance sigma_t^2 as:

        - sigma_t^2 = Omega + Alpha * u_(t-1)^2 + Beta * sigma_(t-1)^2

        With the constraints Omega, Alpha, Beta > 0 and Alpha + Beta < 1. The parameters are
        estimated via simulated annealing, maximizing the GARCH log-likelihood function.

        Unlike `get_garch` and `get_garch_forecast`, which return a (forecasted) volatility path,
        this method returns the fitted parameters themselves. This is useful when the parameters
        are needed directly, for example to seed a separate volatility simulation.

        For more information about the method, see the following book:

        - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1

        Also known as: GARCH weights, GARCH coefficients, conditional variance parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: Omega, Alpha and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits a
        GARCH(1, 1) model for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_garch_parameters(period="quarterly")
        ```

        Which returns:

        |       |   AMZN |   TSLA |   Benchmark |
        |:------|-------:|-------:|------------:|
        | Omega | 0.0191 | 0.1379 |      0.0011 |
        | Alpha | 0.0038 | 0.143  |      0.1528 |
        | Beta  | 0.278  | 0.0677 |      0.6939 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._historical_data[period]["Return"].dropna().replace(0, 1e-100)

        parameters = pd.DataFrame(
            {
                ticker: garch_model.get_garch_weights(
                    returns[ticker].to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Beta"],
        )

        return parameters.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_gjr_garch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates volatility forecasts based on the GJR-GARCH model.

        GJR-GARCH extends GARCH with a leverage term that lets negative shocks (bad
        news) raise volatility by more than positive shocks of the same size, a well
        documented asymmetry in equity returns that symmetric GARCH cannot capture.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        Also known as: GJR-GARCH, threshold GARCH, TGARCH.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            time_steps (int, optional): Time steps to calculate GJR-GARCH for.
            optimization_t (int, optional): Time steps to optimize GJR-GARCH for. It is only used if no
            weights are given.
            within_period (bool, optional): Whether to calculate GJR-GARCH within the specified period or
            for the entire period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the GJR-GARCH values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: GJR-GARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates
        GJR-GARCH for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of GJR-GARCH values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_gjr_garch(period="quarterly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2024Q2 | 0.027  | 0.1542 |      0.0053 |
        | 2024Q3 | 0.0267 | 0.1513 |      0.0047 |
        | 2024Q4 | 0.0266 | 0.1642 |      0.0048 |
        | 2025Q1 | 0.0264 | 0.1937 |      0.0046 |
        | 2025Q2 | 0.027  | 0.1603 |      0.0065 |
        | 2025Q3 | 0.0267 | 0.157  |      0.0053 |
        | 2025Q4 | 0.0265 | 0.173  |      0.005  |
        | 2026Q1 | 0.0264 | 0.1503 |      0.0046 |
        | 2026Q2 | 0.0267 | 0.1506 |      0.0066 |
        | 2026Q3 | 0.0265 | 0.1513 |      0.0061 |
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

        gjr_garch_sigma_2 = garch_model.get_gjr_garch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
        )

        gjr_garch_sigma_2 = gjr_garch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=gjr_garch_sigma_2,
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
    def get_gjr_garch_forecast(
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
        Calculates sigma_2 forecasts based on the GJR-GARCH model.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            time_steps (int, optional): Time steps to calculate GJR-GARCH and to forecast sigma_2 values for.
            within_period (bool, optional): Whether to calculate GJR-GARCH within each specified period or
            all at once. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            growth (bool, optional): Whether to calculate the growth of the GJR-GARCH values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        sigma_2 forecast for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_gjr_garch_forecast(period="quarterly")
        ```

        Which returns:

        |        |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2026Q4 | 0.0258 | 0.1716 |      0.0046 |
        | 2027Q1 | 0.0258 | 0.1716 |      0.0046 |
        | 2027Q2 | 0.0264 | 0.1687 |      0.0071 |
        | 2027Q3 | 0.0268 | 0.1681 |      0.0086 |
        | 2027Q4 | 0.0272 | 0.168  |      0.0094 |
        | 2028Q1 | 0.0274 | 0.168  |      0.0099 |
        | 2028Q2 | 0.0276 | 0.168  |      0.0102 |
        | 2028Q3 | 0.0277 | 0.168  |      0.0103 |
        | 2028Q4 | 0.0278 | 0.168  |      0.0104 |
        | 2029Q1 | 0.0279 | 0.168  |      0.0105 |
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

        sigma_2_forecast = garch_model.get_gjr_garch_forecast(
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
    def get_gjr_garch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fits a GJR-GARCH(1, 1, 1) model to the historical returns and returns the
        estimated Omega, Alpha, Gamma and Beta parameters for each asset.

        A positive Gamma indicates the presence of a leverage effect (negative shocks
        raise volatility by more than positive ones of the same size), which symmetric
        GARCH cannot represent.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        Also known as: GJR-GARCH weights, GJR-GARCH coefficients, leverage parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: Omega, Alpha, Gamma and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits a
        GJR-GARCH(1, 1, 1) model for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_gjr_garch_parameters(period="quarterly")
        ```

        Which returns:

        |       |   AMZN |    TSLA |   Benchmark |
        |:------|-------:|--------:|------------:|
        | Omega | 0.0074 |  0.138  |      0.0045 |
        | Alpha | 0      |  0.1492 |      0.0699 |
        | Gamma | 0.0428 | -0.0828 |      1      |
        | Beta  | 0.7156 |  0.0711 |      0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._historical_data[period]["Return"].dropna().replace(0, 1e-100)

        parameters = pd.DataFrame(
            {
                ticker: garch_model.get_gjr_garch_weights(
                    returns[ticker].to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Gamma", "Beta"],
        )

        return parameters.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_egarch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates volatility forecasts based on the EGARCH model.

        EGARCH models the log of the conditional variance, which avoids having to
        constrain the parameters to keep the variance positive and, like GJR-GARCH,
        lets negative and positive shocks of the same size have a different impact on
        volatility (the leverage effect).

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        Also known as: exponential GARCH, log-GARCH.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            time_steps (int, optional): Time steps to calculate EGARCH for.
            optimization_t (int, optional): Time steps to optimize EGARCH for. It is only used if no
            weights are given.
            within_period (bool, optional): Whether to calculate EGARCH within the specified period or
            for the entire period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the EGARCH values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: EGARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates
        EGARCH for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of EGARCH values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_egarch(period="quarterly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2024Q2 | 0.0263 | 0.1078 |      0.005  |
        | 2024Q3 | 0.0259 | 0.1658 |      0.0047 |
        | 2024Q4 | 0.0255 | 0.1991 |      0.0048 |
        | 2025Q1 | 0.0262 | 0.2357 |      0.0046 |
        | 2025Q2 | 0.0249 | 0.1134 |      0.0078 |
        | 2025Q3 | 0.0262 | 0.1846 |      0.0051 |
        | 2025Q4 | 0.0257 | 0.2112 |      0.0049 |
        | 2026Q1 | 0.0259 | 0.1543 |      0.0046 |
        | 2026Q2 | 0.0251 | 0.1238 |      0.0079 |
        | 2026Q3 | 0.0261 | 0.1672 |      0.0053 |
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

        egarch_sigma_2 = garch_model.get_egarch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
        )

        egarch_sigma_2 = egarch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=egarch_sigma_2,
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
    def get_egarch_forecast(
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
        Calculates sigma_2 forecasts based on the EGARCH model.

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            time_steps (int, optional): Time steps to calculate EGARCH and to forecast sigma_2 values for.
            within_period (bool, optional): Whether to calculate EGARCH within each specified period or
            all at once. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            growth (bool, optional): Whether to calculate the growth of the EGARCH values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        sigma_2 forecast for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_egarch_forecast(period="quarterly")
        ```

        Which returns:

        |        |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2026Q4 | 0.0261 | 0.1672 |      0.0053 |
        | 2027Q1 | 0.0255 | 0.1465 |      0.0066 |
        | 2027Q2 | 0.0255 | 0.1448 |      0.0067 |
        | 2027Q3 | 0.0255 | 0.1447 |      0.0067 |
        | 2027Q4 | 0.0255 | 0.1447 |      0.0067 |
        | 2028Q1 | 0.0255 | 0.1447 |      0.0067 |
        | 2028Q2 | 0.0255 | 0.1447 |      0.0067 |
        | 2028Q3 | 0.0255 | 0.1447 |      0.0067 |
        | 2028Q4 | 0.0255 | 0.1447 |      0.0067 |
        | 2029Q1 | 0.0255 | 0.1447 |      0.0067 |
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

        sigma_2_forecast = garch_model.get_egarch_forecast(
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
    def get_egarch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fits an EGARCH(1, 1) model to the historical returns and returns the estimated
        Omega, Alpha, Gamma and Beta parameters for each asset.

        A negative Gamma indicates the presence of a leverage effect (negative shocks
        raise volatility by more than positive ones of the same size).

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        Also known as: EGARCH weights, EGARCH coefficients, leverage parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: Omega, Alpha, Gamma and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits an
        EGARCH(1, 1) model for each asset in the Toolkit instance.
        - EGARCH's log-variance parameterization is less constrained than plain GARCH, which makes it
        more prone to unstable, boundary-hugging fits on very short return histories (e.g. yearly data).
        Prefer `period="quarterly"` or a higher frequency where possible.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_egarch_parameters(period="quarterly")
        ```

        Which returns:

        |       |    AMZN |    TSLA |   Benchmark |
        |:------|--------:|--------:|------------:|
        | Omega | -3.5971 | -1.7701 |     -4.6755 |
        | Alpha | -0.0115 | -0.0296 |      0.485  |
        | Gamma |  0.0286 |  0.3887 |     -0.4054 |
        | Beta  |  0.0196 |  0.0844 |      0.0651 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._historical_data[period]["Return"].dropna().replace(0, 1e-100)

        parameters = pd.DataFrame(
            {
                ticker: garch_model.get_egarch_weights(
                    returns[ticker].to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Gamma", "Beta"],
        )

        return parameters.round(rounding if rounding is not None else self._rounding)

    def _get_price_column(self, period: str, column: str) -> pd.DataFrame:
        # Reads the plain period-frequency history rather than the "within period"
        # multi-index, so period="daily" simply means daily observations here and
        # (unlike every within-period method above) needs no intraday data.
        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        return self._historical_data[period][column].dropna()

    def _resolve_ticker_pairs(
        self, ticker_a: str | None, ticker_b: str | None
    ) -> list[tuple[str, str]]:
        if ticker_a is None and ticker_b is None:
            if len(self._tickers) < MINIMUM_TICKERS_FOR_ALL_PAIRS:
                raise ValueError(
                    "At least 2 tickers are required in the Toolkit instance to "
                    "calibrate a copula between every pair."
                )

            return list(itertools.combinations(self._tickers, 2))

        if ticker_a is None or ticker_b is None:
            raise ValueError(
                "Either provide both ticker_a and ticker_b, or neither (to use "
                "every pair of tickers in the Toolkit instance)."
            )

        return [(ticker_a, ticker_b)]

    @handle_errors
    def get_tail_dependence_coefficient(
        self,
        ticker_a: str,
        ticker_b: str,
        period: str | None = None,
        column: str = "Return",
        q: float = 0.95,
        method: str = "empirical",
        dof: float = 4.0,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Upper and Lower Tail Dependence Coefficients between `ticker_a`
        and `ticker_b`.

        Correlation only captures the *average* co-movement between two assets -- it
        says nothing about whether they are more likely to crash together than an
        equivalent gaussian relationship would imply. The Tail Dependence Coefficient
        answers that specific question directly: the probability that one asset is in
        extreme distress, given that the other one already is.

        For more information about the method, see the following papers:

        - Embrechts, P., McNeil, A., & Straumann, D. (1999). "Correlation: Pitfalls
        and Alternatives." RISK Magazine, 12, 69-71.
        - Poon, S.H., Rockinger, M., & Tawn, J. (2004). "Extreme Value Dependence in
        Financial Markets: Diagnostics, Models, and Financial Implications." Review of
        Financial Studies, 17(2), 581-610.

        Also known as: tail dependence, extremal dependence coefficient.

        Args:
            ticker_a (str): The first asset.
            ticker_b (str): The second asset.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return", since
            tail dependence between return series is the standard risk management application.
            q (float, optional): The threshold quantile used for the "empirical" method, in (0.5, 1).
            Defaults to 0.95.
            method (str, optional): The estimation method, one of "empirical", "gaussian" or
            "student-t". Defaults to "empirical".
            dof (float, optional): The degrees of freedom of the Student-T copula, only used when
            `method="student-t"`. Defaults to 4.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The Lower and Upper Tail Dependence Coefficients, the linear (Pearson)
            correlation between the two assets, and the number of observations used.

        Notes:
        - The method retrieves historical data based on the specified `period` for the two given
        assets and estimates the tail dependence coefficient between them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_tail_dependence_coefficient("AAPL", "MSFT", period="weekly")
        ```

        Which returns:

        | Metric                |   Value |
        |:-----------------------|--------:|
        | Lower Tail Dependence  |  0.25   |
        | Upper Tail Dependence  |  0.375  |
        | Correlation            |  0.7602 |
        | Observations           | 157      |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        result = copula_model.get_tail_dependence_coefficient(
            returns[ticker_a], returns[ticker_b], q=q, method=method, dof=dof
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_copula_parameters(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        copula: str = "gaussian",
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calibrate a bivariate copula between `ticker_a` and `ticker_b`, via
        maximum likelihood.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is calibrated instead.

        A copula separates the dependence structure between two assets from
        their individual (marginal) return distributions (Sklar's theorem),
        letting the two be modeled independently. Five families are supported,
        each capturing a different shape of dependence:

        - "gaussian": zero tail dependence -- crashes and rallies are no more
        likely to happen together than the correlation alone implies. Included
        mainly as a baseline, since real asset returns typically show more
        joint tail risk than this.
        - "student-t": symmetric, nonzero tail dependence in both tails.
        - "clayton": nonzero *lower* tail dependence only -- assets crash
        together more than they rally together. The most common choice for
        equity return pairs.
        - "gumbel": nonzero *upper* tail dependence only -- assets rally
        together more than they crash together.
        - "frank": zero tail dependence in both tails, but (unlike gaussian) can
        represent negative dependence and is symmetric around independence.

        See `get_best_fitting_copula` to compare all five by AIC on the same
        pair of assets, and see `get_copula_simulation` to draw joint scenarios
        from the fitted copula.

        For more information about the method, see:

        - Sklar, A. (1959). Publications de l'Institut de Statistique de
        l'Universite de Paris, 8, 229-231.
        - McNeil, A.J., Frey, R., & Embrechts, P. (2015). "Quantitative Risk
        Management: Concepts, Techniques and Tools." Princeton University Press.
        - Demarta, S., & McNeil, A.J. (2005). "The T Copula and Related
        Copulas." International Statistical Review, 73(1), 111-129.

        Also known as: copula calibration, copula fit, dependence modeling.

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is calibrated (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            copula (str, optional): The copula family to fit, one of "gaussian",
            "student-t", "clayton", "gumbel" or "frank". Defaults to "gaussian".
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Raises:
            ValueError: If `copula` is not one of "gaussian", "student-t", "clayton", "gumbel" or "frank",
            if only one of `ticker_a`/`ticker_b` is given, if fewer than 2 tickers are available to pair
            up when neither is given, or if fewer than 10 paired, non-missing observations are available.

        Returns:
            pd.Series | pd.DataFrame: The fitted copula parameter(s), the Lower and Upper Tail
            Dependence implied by them, the Log-Likelihood, the AIC and the number of observations used
            -- a Series for a single given pair, or a DataFrame indexed by every (Ticker A, Ticker B)
            pair when neither ticker is given.

        Notes:
        - The method retrieves historical data based on the specified `period` and calibrates the
        requested copula between the given pair, or every pair, of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_copula_parameters("AAPL", "MSFT", copula="clayton", period="weekly")
        ```

        Which returns:

        |                       |    Value |
        |:----------------------|---------:|
        | Theta                 |   0.7342 |
        | Lower Tail Dependence |   0.389  |
        | Upper Tail Dependence |   0      |
        | Log-Likelihood        |  33.6247 |
        | AIC                   | -65.2495 |
        | Observations          | 314      |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "gaussian": copula_model.get_gaussian_copula_parameters,
            "student-t": copula_model.get_student_t_copula_parameters,
            "clayton": copula_model.get_clayton_copula_parameters,
            "gumbel": copula_model.get_gumbel_copula_parameters,
            "frank": copula_model.get_frank_copula_parameters,
        }

        if copula not in fit_functions:
            raise ValueError(f"copula must be one of {', '.join(fit_functions)}.")

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            result = pd.Series(
                fit_functions[copula](returns[ticker_a], returns[ticker_b])
            )

            return result.round(rounding if rounding is not None else self._rounding)

        results = {
            (pair_a, pair_b): fit_functions[copula](returns[pair_a], returns[pair_b])
            for pair_a, pair_b in ticker_pairs
        }

        result_df = pd.DataFrame(results).T
        result_df.index.names = ["Ticker A", "Ticker B"]

        return result_df.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_copula_simulation(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        copula: str = "gaussian",
        period: str | None = None,
        column: str = "Return",
        n_simulations: int = 10_000,
        random_state: int = 42,
        empirical_margins: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calibrate a bivariate copula (see `get_copula_parameters`) between
        `ticker_a` and `ticker_b`, and simulate joint scenarios from it.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is calibrated and simulated instead, as columns
        grouped under a top-level (Ticker A, Ticker B) column per pair.

        This is what a copula-based portfolio Monte Carlo simulation needs: draws
        that preserve the calibrated dependence structure -- including tail
        dependence, for the "student-t", "clayton" and "gumbel" families -- rather
        than the (typically understated) joint crash risk a plain multivariate
        gaussian simulation would produce.

        Also known as: copula Monte Carlo, dependence simulation, scenario
        generation.

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is used (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            copula (str, optional): The copula family to fit and simulate from, one of "gaussian",
            "student-t", "clayton", "gumbel" or "frank". Defaults to "gaussian".
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            n_simulations (int, optional): The number of joint draws to simulate. Defaults to 10,000.
            random_state (int, optional): The seed for the random number generator. Defaults to 42.
            empirical_margins (bool, optional): Whether to map the simulated pseudo-observations back
            to realistic returns via each asset's own empirical (historical) quantile function.
            Defaults to True. When False, the raw pseudo-observations (each in (0, 1)) are returned
            instead.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Raises:
            ValueError: If `copula` is not one of "gaussian", "student-t", "clayton", "gumbel" or "frank",
            if only one of `ticker_a`/`ticker_b` is given, if fewer than 2 tickers are available to pair
            up when neither is given, or if fewer than 10 paired, non-missing observations are available.

        Returns:
            pd.DataFrame: `n_simulations` simulated joint draws, one column per asset (or "U"/"V" when
            `empirical_margins=False`) for a single given pair, or with columns grouped by
            (Ticker A, Ticker B) pair when neither ticker is given.

        Notes:
        - The method retrieves historical data based on the specified `period`, calibrates the
        requested copula, and simulates from it, for the given pair or every pair of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_copula_simulation(
            "AAPL", "MSFT", copula="clayton", period="weekly", n_simulations=5000
        ).describe()
        ```

        Which returns:

        |       |          AAPL |         MSFT |
        |:------|--------------:|-------------:|
        | count | 5000          | 5000         |
        | mean  |    0.00357512 |    0.0028215 |
        | std   |    0.0371953  |    0.0365134 |
        | min   |   -0.1316     |   -0.0764    |
        | 25%   |   -0.0196     |   -0.0211    |
        | 50%   |    0.0019     |    0.00145   |
        | 75%   |    0.0261     |    0.0213    |
        | max   |    0.1315     |    0.2169    |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "gaussian": copula_model.get_gaussian_copula_parameters,
            "student-t": copula_model.get_student_t_copula_parameters,
            "clayton": copula_model.get_clayton_copula_parameters,
            "gumbel": copula_model.get_gumbel_copula_parameters,
            "frank": copula_model.get_frank_copula_parameters,
        }
        simulate_functions = {
            "gaussian": lambda parameters: copula_model.get_gaussian_copula_simulation(
                parameters["Rho"], n_simulations, random_state
            ),
            "student-t": lambda parameters: copula_model.get_student_t_copula_simulation(
                parameters["Rho"],
                parameters["Degrees of Freedom"],
                n_simulations,
                random_state,
            ),
            "clayton": lambda parameters: copula_model.get_clayton_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
            "gumbel": lambda parameters: copula_model.get_gumbel_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
            "frank": lambda parameters: copula_model.get_frank_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
        }

        if copula not in fit_functions:
            raise ValueError(f"copula must be one of {', '.join(fit_functions)}.")

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            simulation = self._simulate_copula_pair(
                ticker_a,
                ticker_b,
                returns,
                fit_functions[copula],
                simulate_functions[copula],
                empirical_margins,
            )

            return simulation.round(
                rounding if rounding is not None else self._rounding
            )

        simulation = pd.concat(
            {
                (pair_a, pair_b): self._simulate_copula_pair(
                    pair_a,
                    pair_b,
                    returns,
                    fit_functions[copula],
                    simulate_functions[copula],
                    empirical_margins,
                )
                for pair_a, pair_b in ticker_pairs
            },
            axis=1,
        )
        simulation.columns.names = ["Ticker A", "Ticker B", None]

        return simulation.round(rounding if rounding is not None else self._rounding)

    @staticmethod
    def _simulate_copula_pair(
        pair_a: str,
        pair_b: str,
        returns: pd.DataFrame,
        fit_function,
        simulate_function,
        empirical_margins: bool,
    ) -> pd.DataFrame:
        fitted_parameters = fit_function(returns[pair_a], returns[pair_b])
        simulation = simulate_function(fitted_parameters)

        if empirical_margins:
            simulation = pd.DataFrame(
                {
                    pair_a: np.quantile(returns[pair_a].dropna(), simulation["U"]),
                    pair_b: np.quantile(returns[pair_b].dropna(), simulation["V"]),
                }
            )

        return simulation

    @handle_errors
    def get_best_fitting_copula(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        period: str | None = None,
        column: str = "Return",
        show_full_results: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calibrate all five supported copula families (see `get_copula_parameters`)
        between `ticker_a` and `ticker_b`, and compare them by AIC (Akaike
        Information Criterion) -- the lower the AIC, the better the fit relative
        to its number of parameters, so the top row is the best-fitting family.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is compared instead.

        Also known as: copula selection, copula comparison.

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is compared (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            show_full_results (bool, optional): Only relevant when neither ticker is given. When False
            (the default), returns a square ticker-by-ticker grid of just the winning copula family
            per pair. When True, returns one row per pair instead, with the winning family's fitted
            parameter(s), Lower and Upper Tail Dependence, Log-Likelihood, AIC and the number of
            observations used. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Raises:
            ValueError: If only one of `ticker_a`/`ticker_b` is given, or if fewer than 2 tickers are
            available to pair up when neither is given.

        Returns:
            pd.DataFrame: For a single given pair, one row per copula family, sorted by AIC (best fit
            first), with each family's fitted parameter(s), Lower and Upper Tail Dependence,
            Log-Likelihood, AIC and the number of observations used, indexed by "Copula". When neither
            ticker is given, a square ticker-by-ticker grid of the winning copula family per pair (or,
            with `show_full_results=True`, one row per pair instead, indexed by (Ticker A, Ticker B),
            with a "Best Copula" column and that family's fitted stats).

        Notes:
        - The method retrieves historical data based on the specified `period` and calibrates every
        supported copula family for the given pair, or every pair, of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_best_fitting_copula("AAPL", "MSFT", period="weekly")
        ```

        Which returns:

        | Copula    |   Lower Tail Dependence |   Upper Tail Dependence |   Log-Likelihood |      AIC |
        |:----------|------------------------:|------------------------:|------------------:|---------:|
        | Student-T |                  0.1915 |                  0.1915 |            51.9971 | -99.9942 |
        | Frank     |                  0      |                  0      |            49.2452 | -96.4903 |
        | Gumbel    |                  0      |                  0.4256 |            48.9139 | -95.8277 |
        | Gaussian  |                  0      |                  0      |            42.908  | -83.816  |
        | Clayton   |                  0.389  |                  0      |            33.6247 | -65.2495 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "Gaussian": copula_model.get_gaussian_copula_parameters,
            "Student-T": copula_model.get_student_t_copula_parameters,
            "Clayton": copula_model.get_clayton_copula_parameters,
            "Gumbel": copula_model.get_gumbel_copula_parameters,
            "Frank": copula_model.get_frank_copula_parameters,
        }

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            comparison_df = self._compare_copula_pair(
                ticker_a, ticker_b, returns, fit_functions
            )

            return comparison_df.round(
                rounding if rounding is not None else self._rounding
            )

        # One row per pair: only the lowest-AIC family, which sorts first, is kept.
        comparison_df = pd.concat(
            {
                (pair_a, pair_b): self._compare_copula_pair(
                    pair_a, pair_b, returns, fit_functions
                ).iloc[[0]]
                for pair_a, pair_b in ticker_pairs
            }
        )
        comparison_df.index.names = ["Ticker A", "Ticker B", "Best Copula"]
        comparison_df = comparison_df.reset_index(level="Best Copula")

        if show_full_results:
            return comparison_df.round(
                rounding if rounding is not None else self._rounding
            )

        tickers = sorted({ticker for pair in ticker_pairs for ticker in pair})
        grid = pd.DataFrame(index=tickers, columns=tickers, dtype=object)

        for (pair_a, pair_b), best_copula in comparison_df["Best Copula"].items():
            grid.loc[pair_a, pair_b] = best_copula
            grid.loc[pair_b, pair_a] = best_copula

        grid.index.name = "Ticker A"
        grid.columns.name = "Ticker B"

        return grid

    @staticmethod
    def _compare_copula_pair(
        pair_a: str, pair_b: str, returns: pd.DataFrame, fit_functions: dict
    ) -> pd.DataFrame:
        comparison = {
            copula_name: fit_function(returns[pair_a], returns[pair_b])
            for copula_name, fit_function in fit_functions.items()
        }

        pair_df = pd.DataFrame(comparison).T.sort_values("AIC")
        pair_df.index.name = "Copula"

        return pair_df

    @handle_errors
    def get_covar(
        self,
        ticker: str,
        conditioning_ticker: str,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the (Delta-)CoVaR of `ticker` conditional on `conditioning_ticker`
        being in its own distress state.

        Ordinary Value at Risk treats each asset in isolation, which misses systemic
        risk -- the fact that one asset's distress can spill over and worsen
        another's risk. CoVaR directly measures that spillover: it is the VaR of
        `ticker`, conditional on `conditioning_ticker` itself being at its own
        `alpha`-VaR, estimated via a linear Quantile Regression of `ticker`'s returns
        on `conditioning_ticker`'s returns at quantile `alpha`. The Delta-CoVaR
        isolates the marginal, distress-specific contribution by subtracting the same
        construction evaluated in the "normal" (median) state instead.

        For more information about the method, see the following paper:

        - Adrian, T., & Brunnermeier, M.K. (2016). "CoVaR." American Economic Review,
        106(7), 1705-1741.

        Also known as: Conditional Value at Risk (systemic risk sense), Delta-CoVaR.

        Args:
            ticker (str): The asset whose conditional VaR is being measured.
            conditioning_ticker (str): The asset (or e.g. a benchmark/index) whose distress
            `ticker` is conditioned on.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The confidence level for both the tail quantile regression and
            the VaR of `conditioning_ticker` (e.g., 0.05 for 95% confidence). Defaults to 0.05.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The CoVaR, the Delta-CoVaR, the tail (alpha-quantile) Quantile Regression
            slope and intercept, and the number of observations used.

        Notes:
        - The method retrieves historical data based on the specified `period` for the two given
        assets and estimates the CoVaR of `ticker` conditional on `conditioning_ticker`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_covar("AAPL", "MSFT", period="weekly")
        ```

        Which returns:

        | Metric                         |   Value |
        |:--------------------------------|--------:|
        | CoVaR                           | -0.1077 |
        | Delta-CoVaR                     | -0.1134 |
        | Quantile Regression Slope       |  0.9214 |
        | Quantile Regression Intercept   | -0.0508 |
        | Observations                    | 157     |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        result = covar_model.get_covar(
            returns[ticker], returns[conditioning_ticker], alpha=alpha
        )

        return result.round(rounding if rounding is not None else self._rounding)

    def _get_portfolio_weights(
        self, returns: pd.DataFrame, weights: dict[str, float] | pd.Series | None
    ) -> pd.Series:
        if weights is None:
            constituent_tickers = [
                ticker
                for ticker in returns.columns
                if ticker not in ("Portfolio", "Benchmark")
            ]
            if not constituent_tickers:
                raise ValueError(
                    "No tickers available to build equal weights from. Please "
                    "provide the `weights` argument explicitly."
                )
            return pd.Series(1 / len(constituent_tickers), index=constituent_tickers)

        weights_series = (
            pd.Series(weights) if isinstance(weights, dict) else weights.copy()
        )
        missing_tickers = [
            ticker for ticker in weights_series.index if ticker not in returns.columns
        ]
        if missing_tickers:
            raise ValueError(
                f"weights contains tickers not present in the Toolkit: {missing_tickers}"
            )

        return weights_series

    @handle_errors
    def get_marginal_value_at_risk(
        self,
        weights: dict[str, float] | None = None,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        distribution: str = "historic",
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Marginal Value at Risk (Marginal VaR) of each asset in a
        portfolio.

        Ordinary VaR (see `get_value_at_risk`) treats each asset in isolation.
        Marginal VaR instead measures how much the *portfolio's* VaR would change for
        an infinitesimal increase in a given asset's portfolio weight — i.e. the
        sensitivity of portfolio risk to each holding, not the risk of the holding on
        its own:

        - Portfolio Return = SUM(weight_i * Return_i)
        - Beta_i = Cov(Return_i, Portfolio Return) / Var(Portfolio Return)
        - Marginal VaR_i = Beta_i * Portfolio VaR

        An asset with Beta_i > 1 contributes disproportionately to portfolio risk,
        while Beta_i < 1 (and especially Beta_i < 0) indicates a diversifying holding.

        For more information about the method, see the following sources:

        - Garman, M.B. (1997). "Taking VaR to Pieces." Risk, 10(10), 70-71.
        - Litterman, R. (1996). "Hot Spots and Hedges." Goldman Sachs Risk Management
        Series.
        - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial
        Risk." 3rd ed., McGraw-Hill, Chapter 7.

        Also known as: Marginal VaR, MVaR.

        Args:
            weights (dict[str, float] | None, optional): Portfolio weights
            keyed by ticker. Normalized internally to sum to 1. Defaults to None, which
            uses equal weights across every ticker in the Toolkit instance (excluding
            the "Portfolio" and "Benchmark" pseudo-tickers, if present).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The confidence level (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            distribution (str, optional): The distribution to use for the underlying portfolio
            VaR calculation (historic, gaussian, cornish-fisher or studentt). Defaults to
            "historic".
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.

        Returns:
            pd.Series: The Marginal VaR of each asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` for
        every ticker in the Toolkit instance and computes each asset's Marginal VaR against
        the weighted portfolio built from `weights`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_marginal_value_at_risk(weights={"AMZN": 0.5, "TSLA": 0.3, "MSFT": 0.2})
        ```

        Which returns:

        |      |   Marginal VaR |
        |:-----|---------------:|
        | AMZN |        -0.0512 |
        | TSLA |        -0.0698 |
        | MSFT |        -0.0331 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)
        weights_series = self._get_portfolio_weights(returns, weights)

        marginal_var = var_model.get_marginal_var(
            returns[weights_series.index], weights_series, alpha, distribution
        )
        marginal_var.name = "Marginal VaR"

        return marginal_var.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_component_value_at_risk(
        self,
        weights: dict[str, float] | None = None,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        distribution: str = "historic",
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Component Value at Risk (Component VaR) of each asset in a
        portfolio.

        Component VaR allocates total portfolio VaR across its constituent assets,
        such that the allocations sum exactly back to the portfolio VaR — an "Euler"
        (or "fully consistent") risk decomposition:

        - Component VaR_i = weight_i * Marginal VaR_i    (see `get_marginal_value_at_risk`)
        - SUM(Component VaR_i) = Portfolio VaR

        Where Marginal VaR measures each asset's risk *sensitivity*, Component VaR
        measures its actual *contribution* in the same units as portfolio VaR, making
        it directly usable to identify which holdings account for the largest share of
        portfolio risk.

        For more information about the method, see the following sources:

        - Garman, M.B. (1997). "Taking VaR to Pieces." Risk, 10(10), 70-71.
        - Litterman, R. (1996). "Hot Spots and Hedges." Goldman Sachs Risk Management
        Series.
        - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial
        Risk." 3rd ed., McGraw-Hill, Chapter 7.

        Also known as: Component VaR, risk contribution.

        Args:
            weights (dict[str, float] | None, optional): Portfolio weights
            keyed by ticker. Normalized internally to sum to 1. Defaults to None, which
            uses equal weights across every ticker in the Toolkit instance (excluding
            the "Portfolio" and "Benchmark" pseudo-tickers, if present).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The confidence level (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            distribution (str, optional): The distribution to use for the underlying portfolio
            VaR calculation (historic, gaussian, cornish-fisher or studentt). Defaults to
            "historic".
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.

        Returns:
            pd.Series: The Component VaR of each asset, plus a "Portfolio" entry equal to
            their sum (the total portfolio VaR), for validation.

        Notes:
        - The method retrieves historical return data based on the specified `period` for
        every ticker in the Toolkit instance and computes each asset's Component VaR against
        the weighted portfolio built from `weights`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_component_value_at_risk(weights={"AMZN": 0.5, "TSLA": 0.3, "MSFT": 0.2})
        ```

        Which returns:

        |           |   Component VaR |
        |:----------|-----------------:|
        | AMZN      |          -0.0256 |
        | TSLA      |          -0.0209 |
        | MSFT      |          -0.0066 |
        | Portfolio |          -0.0531 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)
        weights_series = self._get_portfolio_weights(returns, weights)

        component_var = var_model.get_component_var(
            returns[weights_series.index], weights_series, alpha, distribution
        )
        component_var.loc["Portfolio"] = component_var.sum()
        component_var.name = "Component VaR"

        return component_var.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_var_backtest(
        self,
        period: str | None = None,
        distribution: str = "historic",
        alpha: float = 0.05,
        window_size: int = 252,
        test: str = "both",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Backtest a Value at Risk model against realized returns.

        A Value at Risk estimate is only useful if it is actually well-calibrated
        against reality. This method builds a rolling, out-of-sample VaR path (each
        VaR estimate uses only the `window_size` returns preceding it, never the
        return it is compared against) and tests it with Kupiec's Proportion of
        Failures test (is the overall breach rate consistent with `alpha`?) and/or
        Christoffersen's independence test (are breaches spread out over time, or do
        they cluster together?).

        For more information about the methods, see the following papers:

        - Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy of Risk
        Measurement Models." The Journal of Derivatives, 3(2), 73-84.
        - Christoffersen, P.F. (1998). "Evaluating Interval Forecasts." International
        Economic Review, 39(4), 841-862.

        Also known as: VaR validation, VaR backtest, POF test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "daily", since `window_size` is expressed in return observations of this frequency
            (252 only means "about one year" when `period` is daily).
            distribution (str, optional): The distribution to use for the rolling VaR estimates, one of
            "historic", "gaussian", "studentt" or "evt". Defaults to "historic".
            alpha (float, optional): The confidence level for the VaR estimates (e.g., 0.05 for 95%
            confidence). Defaults to 0.05.
            window_size (int, optional): The rolling window size (in number of return observations) used to
            estimate each VaR value. Defaults to 252 (approximately one trading year of daily returns).
            test (str, optional): Which test(s) to run, one of "kupiec", "christoffersen" or "both". Defaults
            to "both".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The requested test statistic(s) and their p-value(s) per asset.

        Notes:
        - The rolling VaR path is calculated over the full return history, not the `within_period` slices
        used elsewhere in this module, since a meaningful rolling window generally needs more history than
        a single sub-period provides.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_var_backtest(window_size=252)
        ```

        Which returns:

        |                         |    AMZN |    TSLA |   Benchmark |
        |:------------------------|--------:|--------:|------------:|
        | Kupiec Statistic        |  0.0817 |  1.6459 |      0.0018 |
        | P-Value                 |  0.7749 |  0.1995 |      0.9662 |
        | Christoffersen Statistic |  0.0631 |  0.4302 |      1.0847 |
        | P-Value                 |  0.8017 |  0.512  |      0.2977 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if distribution not in ["historic", "gaussian", "studentt", "evt"]:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, or evt."
            )
        if test not in ["kupiec", "christoffersen", "both"]:
            raise ValueError("Test must be kupiec, christoffersen, or both.")

        returns = self._historical_data[period]["Return"].dropna()

        if distribution == "historic":
            rolling_var = var_model.get_rolling_var_historic(
                returns, alpha, window_size
            )
        elif distribution == "gaussian":
            rolling_var = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(var_model.get_var_gaussian(window, alpha)),
                raw=False,
            )
        elif distribution == "studentt":
            rolling_var = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(var_model.get_var_studentt(window, alpha)),
                raw=False,
            )
        else:
            rolling_var = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(var_model.get_var_evt(window, alpha)),
                raw=False,
            )

        # A trailing rolling window includes the current observation, so without the shift
        # each estimate would be compared against a return it was computed from.
        rolling_var = rolling_var.shift(1)

        results = []
        if test in ("kupiec", "both"):
            results.append(
                backtesting_model.get_kupiec_test(returns, rolling_var, alpha)
            )
        if test in ("christoffersen", "both"):
            results.append(
                backtesting_model.get_christoffersen_test(returns, rolling_var)
            )

        result = pd.concat(results, axis=0) if len(results) > 1 else results[0]

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_acerbi_szekely_test(
        self,
        period: str | None = None,
        distribution: str = "historic",
        alpha: float = 0.05,
        window_size: int = 252,
        n_bootstrap: int = 1000,
        random_state: int = 42,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Backtest a Conditional Value at Risk (Expected Shortfall) model against
        realized returns, via the Acerbi-Szekely (2014) Z2 statistic.

        `get_var_backtest` above only backtests the VaR estimate itself -- it checks
        how often (and how independently) the VaR threshold is breached, but says
        nothing about the *severity* of the losses on those breach days, which is
        exactly the extra information a CVaR (Expected Shortfall) estimate is
        supposed to add over VaR. This method builds a rolling, out-of-sample CVaR
        (and VaR) path and compares the actual loss on each breach day to the CVaR
        that was supposed to describe the average loss on such days.

        For more information about the method, see the following paper:

        - Acerbi, C., & Szekely, B. (2014). "Back-Testing Expected Shortfall." RISK
        Magazine, 27(11), 76-81.

        Also known as: Acerbi-Szekely test, ES backtest, Z2 test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "daily", since `window_size` is expressed in return observations of this frequency.
            distribution (str, optional): The distribution to use for the rolling VaR/CVaR estimates,
            one of "historic", "gaussian", "studentt" or "evt". Defaults to "historic".
            alpha (float, optional): The confidence level for the VaR/CVaR estimates (e.g., 0.05 for 95%
            confidence). Defaults to 0.05.
            window_size (int, optional): The rolling window size (in number of return observations) used
            to estimate each VaR/CVaR value. Defaults to 252 (approximately one trading year of daily
            returns).
            n_bootstrap (int, optional): The number of bootstrap resamples used to estimate the Standard
            Error of Z2. Defaults to 1000.
            random_state (int, optional): The seed for the bootstrap random number generator. Defaults to
            42.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The Z2 statistic, its bootstrap Standard Error, its p-value, and the number of
            breaches observed, per asset.

        Notes:
        - The rolling VaR/CVaR path is calculated over the full return history, not the `within_period`
        slices used elsewhere in this module, since a meaningful rolling window generally needs more
        history than a single sub-period provides.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_acerbi_szekely_test(window_size=100)
        ```

        Which returns:

        |                           |    AAPL |    MSFT |   Benchmark |
        |:--------------------------|--------:|--------:|------------:|
        | Acerbi-Szekely Statistic  |  0.0905 |  0.0095 |     -0.003  |
        | Standard Error            |  0.1817 |  0.1723 |      0.1663 |
        | P-Value                   |  0.6185 |  0.9558 |      0.9858 |
        | Breaches                  | 37      | 33      |     33      |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if distribution not in ["historic", "gaussian", "studentt", "evt"]:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, or evt."
            )

        returns = self._historical_data[period]["Return"].dropna()

        if distribution == "historic":
            rolling_var = var_model.get_rolling_var_historic(
                returns, alpha, window_size
            )
            rolling_cvar = cvar_model.get_rolling_cvar_historic(
                returns, alpha, window_size
            )
        elif distribution == "gaussian":
            rolling_var = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(var_model.get_var_gaussian(window, alpha)),
                raw=False,
            )
            rolling_cvar = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(cvar_model.get_cvar_gaussian(window, alpha)),
                raw=False,
            )
        elif distribution == "studentt":
            rolling_var = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(var_model.get_var_studentt(window, alpha)),
                raw=False,
            )
            rolling_cvar = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(cvar_model.get_cvar_studentt(window, alpha)),
                raw=False,
            )
        else:
            rolling_var = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(var_model.get_var_evt(window, alpha)),
                raw=False,
            )
            rolling_cvar = returns.rolling(window=window_size).apply(
                lambda window: _as_scalar(cvar_model.get_cvar_evt(window, alpha)),
                raw=False,
            )

        # A trailing rolling window includes the current observation, so without the shift
        # each estimate would be compared against a return it was computed from.
        rolling_var = rolling_var.shift(1)
        rolling_cvar = rolling_cvar.shift(1)

        result = backtesting_model.get_acerbi_szekely_test(
            returns, rolling_var, rolling_cvar, alpha, n_bootstrap, random_state
        )

        return result.round(rounding if rounding is not None else self._rounding)

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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Skewness within the specified period or for the
            entire period. Thus whether to look at the Skewness within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Skewness is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Skewness values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Skewness values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates Skew for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Skewness values using the specified `lag`.

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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Kurtosis within the specified period or for the
            entire period. Thus whether to look at the Kurtosis within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to True.
            fisher (bool, optional): Whether to use Fisher's definition of kurtosis (kurtosis = 0.0
            for a normal distribution). Defaults to False.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Kurtosis is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Kurtosis values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Kurtosis values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates Kurtosis for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Kurtosis values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

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
    def get_hill_estimator(
        self,
        period: str | None = None,
        within_period: bool = True,
        k: int | float = 0.1,
        tail: str = "left",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Hill Estimator of the tail index of returns, per asset.

        Unlike the (finite-sample) Skewness and Kurtosis above, the Hill Estimator is
        a semi-parametric estimate of how heavy the tail of the return distribution
        actually is, under the assumption that the tail follows a Pareto-type power
        law. Smaller values of the tail index indicate a heavier tail (more extreme
        outliers are likely) -- as a rule of thumb, a tail index below 4 implies the
        Kurtosis is theoretically infinite, and below 2 implies the Variance itself is
        theoretically infinite.

        For more information about the method, see the following paper:

        - Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of
        a Distribution." The Annals of Statistics, 3(5), 1163-1174.

        Also known as: Hill tail index estimator, Hill's estimator.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the estimator within the specified period
            or for the entire period. Defaults to True.
            k (int | float, optional): The number of upper order statistics to use. If a float in (0, 1)
            it is interpreted as the fraction of the strictly positive observations to use. Defaults to
            0.1 (the top 10%).
            tail (str, optional): Which tail to estimate, one of "left" (the loss tail) or "right" (the
            gain tail). Defaults to "left".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The Hill tail index, the Hill shape parameter, its Standard Error and the
            number of order statistics used, per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Hill Estimator for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_hill_estimator(period="weekly", within_period=False)
        ```

        Which returns:

        |                        |   AAPL |   MSFT |   Benchmark |
        |:-----------------------|-------:|-------:|------------:|
        | Hill Tail Index        | 2.6934 | 5.0478 |      2.1843 |
        | Hill Shape (xi)        | 0.3713 | 0.1981 |      0.4578 |
        | Standard Error         | 1.018  | 1.9079 |      0.8256 |
        | Observations Used (k)  | 7      | 7      |      7      |
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
        ).dropna()

        result = risk_model.get_hill_estimator(returns, k=k, tail=tail)

        return result.round(rounding if rounding is not None else self._rounding)

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
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
        method: str = "close_to_close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Volatility of an investment portfolio or asset's returns for a
        given period based on the daily historical prices.

        Volatility measures the amount of dispersion or variability in returns. It is the
        square root of the Variance. A higher Volatility indicates greater variability, while
        a lower Volatility suggests that returns are closer to the mean.

        By default this is the close-to-close Volatility, i.e. the standard deviation of the
        daily returns. The `method` parameter selects one of four range-based estimators
        instead, each of which uses more of the day's price action than just the close and is
        therefore more statistically efficient (i.e. needs fewer observations to reach the
        same precision), at the cost of additional assumptions about how prices move:

        - `"parkinson"` — uses the daily trading range (High vs Low) rather than the
        close-to-close return, assuming prices follow a continuous geometric Brownian motion
        with no drift and no overnight jumps.
        - `"garman_klass"` — extends Parkinson by also incorporating the Open and Close, which
        allows it to account for the opening jump and makes it more efficient still (assuming,
        as Parkinson does, no drift and no overnight jumps beyond the modeled open).
        - `"rogers_satchell"` — drift-independent, meaning it remains unbiased even when the
        underlying asset has a non-zero expected return over the period, at the cost of still
        assuming no overnight jumps.
        - `"yang_zhang"` — a weighted combination of the overnight (close-to-open) Variance,
        the open-to-close Variance and the Rogers-Satchell Variance. It is both
        drift-independent and accounts for overnight jumps, which makes it the most
        statistically efficient of the range-based estimators implemented here.

        In every case the daily Volatility is scaled to the given period by multiplying the
        underlying Variance with the number of trading days within that period (e.g. 252 / 52
        for weekly).

        Also known as: standard deviation of returns. The range-based estimators are also known
        as Parkinson's range-based or high-low Volatility, Garman-Klass range-based Volatility,
        Rogers-Satchell drift-independent Volatility and Yang-Zhang drift-independent
        overnight-aware Volatility.

        For more information about the range-based estimators, see the following papers:

        - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the
        Rate of Return." Journal of Business, 53(1), 61-65.
        - Garman, M.B., & Klass, M.J. (1980). "On the Estimation of Security Price
        Volatilities from Historical Data." Journal of Business, 53(1), 67-78.
        - Rogers, L.C.G., & Satchell, S.E. (1991). "Estimating Variance from High, Low and
        Close Prices." Annals of Applied Probability, 1(4), 504-512.
        - Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on
        High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-491.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Volatility is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Volatility) instead of one value per `period`.
            Only available for method="close_to_close". Defaults to None.
            method (str, optional): Which Volatility estimator to use, one of "close_to_close",
            "parkinson", "garman_klass", "rogers_satchell" or "yang_zhang", as described above.
            Defaults to "close_to_close".
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
        - The method retrieves the daily historical price data and calculates the Volatility for
        the specified `period` for each asset in the Toolkit instance. The close-to-close estimator
        only needs the daily returns, Parkinson needs the High and Low, and the remaining
        estimators need the Open, High, Low and Close.
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

        And, using the daily trading range instead of only the closes:

        ```python
        toolkit.risk.get_volatility(period="yearly", method="parkinson")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.2099 | 0.426  |      0.103  |
        | 2022   | 0.3717 | 0.5547 |      0.1916 |
        | 2023   | 0.2611 | 0.4365 |      0.1101 |
        | 2024   | 0.219  | 0.4357 |      0.0989 |
        | 2025   | 0.267  | 0.5062 |      0.152  |
        | 2026   | 0.2681 | 0.3781 |      0.1111 |
        """
        if method not in (
            "close_to_close",
            "parkinson",
            "garman_klass",
            "rogers_satchell",
            "yang_zhang",
        ):
            raise ValueError(
                "Method must be close_to_close, parkinson, garman_klass, "
                "rogers_satchell, or yang_zhang."
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling and method != "close_to_close":
            raise ValueError(
                "The rolling parameter is only available for method='close_to_close'."
            )

        if method == "close_to_close":
            if rolling:
                period_returns = self._historical_data[period]["Return"]
                volatility = risk_model.get_rolling_volatility(
                    period_returns, period, rolling
                )
            else:
                returns = self._historical_data["daily"]["Return"]
                volatility = risk_model.get_volatility(returns, period)
        elif method == "parkinson":
            volatility = realized_volatility_model.get_parkinson_volatility(
                self._historical_data["daily"]["High"],
                self._historical_data["daily"]["Low"],
                period,
            )
        else:
            estimators = {
                "garman_klass": realized_volatility_model.get_garman_klass_volatility,
                "rogers_satchell": realized_volatility_model.get_rogers_satchell_volatility,
                "yang_zhang": realized_volatility_model.get_yang_zhang_volatility,
            }

            volatility = estimators[method](
                self._historical_data["daily"]["Open"],
                self._historical_data["daily"]["High"],
                self._historical_data["daily"]["Low"],
                self._historical_data["daily"]["Close"],
                period,
            )

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
    def get_har_rv_forecast(
        self,
        estimator: str = "squared_return",
        weekly_window: int = 5,
        monthly_window: int = 22,
        horizon: int = 1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Corsi (2009) Heterogeneous Autoregressive Realized Volatility
        (HAR-RV) forecast of future daily Realized Variance, per asset.

        Volatility clustering happens across multiple, overlapping time horizons at
        once. HAR-RV captures this cheaply -- without the numerical optimization a
        GARCH-family fit requires (see `get_garch`) -- by regressing future daily
        Realized Variance on trailing daily, weekly and monthly average Realized
        Variance components. The daily Realized Variance itself can be constructed in
        several ways via `estimator`: the simplest is the squared daily return, while
        the OHLC range-based estimators (see `get_volatility`, which exposes the same
        estimators via its own `method` parameter) use the daily
        (pre-period-aggregation) term behind each of those estimators instead,
        which is more statistically efficient since it uses the daily trading range
        rather than only the close-to-close move.

        For more information about the method, see the following paper:

        - Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized
        Volatility." Journal of Financial Econometrics, 7(2), 174-196.

        Also known as: HAR-RV model, Corsi's HAR model, Heterogeneous Autoregressive model.

        Args:
            estimator (str, optional): How to construct the daily Realized Variance input, one of
            "squared_return", "parkinson", "garman_klass" or "rogers_satchell". Defaults to
            "squared_return".
            weekly_window (int, optional): The trailing window (in trading days) for the weekly RV
            component. Defaults to 5.
            monthly_window (int, optional): The trailing window (in trading days) for the monthly RV
            component. Defaults to 22.
            horizon (int, optional): The number of days ahead to forecast. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the HAR-RV forecast values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: The HAR-RV forecast of Realized Variance with time as the index.

        Notes:
        - The method retrieves daily historical price data and calculates the HAR-RV forecast for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the HAR-RV forecast values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_har_rv_forecast(estimator="squared_return").tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2022-12-23 | 0.0004 | 0.0002 |      0.0001 |
        | 2022-12-27 | 0.0004 | 0.0002 |      0.0001 |
        | 2022-12-28 | 0.0006 | 0.0003 |      0.0002 |
        | 2022-12-29 | 0.0006 | 0.0006 |      0.0002 |
        | 2022-12-30 | NaN    | NaN    |    NaN      |

        The last row is NaN since there is no `2022-12-31` return yet to forecast against.
        """
        if estimator not in (
            "squared_return",
            "parkinson",
            "garman_klass",
            "rogers_satchell",
        ):
            raise ValueError(
                "estimator must be 'squared_return', 'parkinson', 'garman_klass', "
                "or 'rogers_satchell'."
            )

        if estimator == "squared_return":
            realized_variance = self._historical_data["daily"]["Return"].dropna() ** 2
        else:
            open_prices = self._historical_data["daily"]["Open"]
            high_prices = self._historical_data["daily"]["High"]
            low_prices = self._historical_data["daily"]["Low"]
            close_prices = self._historical_data["daily"]["Close"]

            if estimator == "parkinson":
                realized_variance = np.log(high_prices / low_prices) ** 2 / (
                    4 * np.log(2)
                )
            elif estimator == "garman_klass":
                realized_variance = (
                    0.5 * np.log(high_prices / low_prices) ** 2
                    - (2 * np.log(2) - 1) * np.log(close_prices / open_prices) ** 2
                )
            else:
                realized_variance = np.log(high_prices / close_prices) * np.log(
                    high_prices / open_prices
                ) + np.log(low_prices / close_prices) * np.log(low_prices / open_prices)

        forecast = realized_volatility_model.get_har_rv_forecast(
            realized_variance, weekly_window, monthly_window, horizon
        )

        forecast = forecast.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=forecast,
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
    def get_amihud_illiquidity(
        self,
        period: str | None = None,
        within_period: bool = True,
        scale: float = 1_000_000,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Amihud (2002) Illiquidity ratio, per asset.

        The Amihud ratio measures the average price impact of trading -- how much the
        price moves per dollar of trading volume. A high value means that even a small
        amount of trading moves the price a lot (the asset is illiquid), while a low
        value means the asset can absorb a large amount of trading with little price
        impact (the asset is liquid).

        For more information about the method, see the following paper:

        - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and
        Time-Series Effects." Journal of Financial Markets, 5(1), 31-56.

        Also known as: Amihud illiquidity ratio, ILLIQ, price impact ratio.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the ratio within the specified period or
            for the entire period. Defaults to True.
            scale (float, optional): A multiplier applied to the resulting ratio purely for readability.
            Defaults to 1,000,000.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Amihud Illiquidity values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Amihud Illiquidity values with time as the index.

        Notes:
        - The method retrieves historical return, Close price and Volume data based on the specified
        `period` and calculates the Amihud Illiquidity ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Amihud Illiquidity values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_amihud_illiquidity(period="quarterly", scale=1e12)
        ```

        Which returns:

        | Date   |   AAPL |   MSFT |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2022Q1 | 0.9388 | 1.4796 |      0.2251 |
        | 2022Q2 | 1.4477 | 2.2593 |      0.3401 |
        | 2022Q3 | 1.1619 | 1.942  |      0.3349 |
        | 2022Q4 | 1.5347 | 2.429  |      0.3445 |

        Note that a large `scale` is used here since these are liquid, large-cap
        stocks with very high dollar trading volume relative to their typical daily
        price move -- the default `scale` of 1,000,000 (as used in Amihud's original
        1980s/1990s-era paper) would round these to 0.0 at the default precision.
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        source_data = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        )

        returns = source_data["Return"].dropna()
        dollar_volume = source_data["Close"] * source_data["Volume"]

        illiquidity = market_liquidity_model.get_amihud_illiquidity(
            returns, dollar_volume, scale
        )

        if within_period:
            illiquidity = illiquidity.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=illiquidity,
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
    def get_roll_spread(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Roll (1984) implied bid-ask spread, per asset.

        Roll's model shows that, under a stylized microstructure model in which the
        true (efficient) price follows a random walk and observed trade prices
        randomly bounce between the bid and the ask, the effective spread can be
        backed out purely from the serial covariance of consecutive price changes,
        without needing any actual quote data. If the estimated covariance is zero or
        positive, no valid estimate can be backed out and NaN is returned instead.

        For more information about the method, see the following paper:

        - Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread
        in an Efficient Market." The Journal of Finance, 39(4), 1127-1139.

        Also known as: Roll's implied spread, Roll measure.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the spread within the specified period or
            for the entire period. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The Roll Spread (in price units), the Roll Spread as a percentage of the mean
            price, the underlying lag-1 autocovariance, and whether that autocovariance was negative (i.e.
            whether a valid estimate could be backed out), per asset.

        Notes:
        - The method retrieves historical Close price data based on the specified `period` and calculates
        the Roll Spread for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # Shown here for a single quarter (2022Q2); with the default within_period=True
        # this is computed separately for every quarter in the Toolkit's date range.
        toolkit.risk.get_roll_spread(period="quarterly").xs("2022Q2", level=0)
        ```

        Which returns:

        |                 |    AAPL |    MSFT |   Benchmark |
        |:----------------|--------:|--------:|------------:|
        | Roll Spread     |  3.3103 |  5.1999 |      4.3436 |
        | Roll Spread (%) |  2.1859 |  1.9159 |      1.0603 |
        | Autocovariance  | -2.7396 | -6.7598 |     -4.7167 |
        | Valid Estimate  |  1      |  1      |      1      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        close_prices = (
            self._within_historical_data[period]["Close"]
            if within_period
            else self._historical_data[period]["Close"]
        )

        result = market_liquidity_model.get_roll_spread(close_prices)

        return result.round(rounding if rounding is not None else self._rounding)

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
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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

        return autocorrelation.round(
            rounding if rounding is not None else self._rounding
        )

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
        # The estimator regresses the dispersion of lagged differences on the lag, which
        # only identifies self-affinity on a level series. Feeding it returns, which are
        # already differenced, collapses the exponent towards zero.
        prices = self._historical_data["daily"]["Adj Close"]

        hurst_exponent = prices.apply(
            lambda column: risk_model.get_hurst_exponent(column, max_lag=max_lag)
        )

        return hurst_exponent.round(
            rounding if rounding is not None else self._rounding
        )
