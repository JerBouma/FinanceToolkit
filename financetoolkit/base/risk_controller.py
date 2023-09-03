"""Risk Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import (
    calculate_growth,
    handle_errors,
    handle_return_data_periods,
)
from financetoolkit.risk import risk

# pylint: disable=too-many-instance-attributes,too-few-public-methods


class Risk:
    """
    Risk Controller Class
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
        Initializes the Risk Controller Class.
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

        # Return Calculations
        self._daily_returns = self._daily_historical["Return"]
        self._weekly_returns = (
            self._daily_historical["Return"]
            .groupby(pd.Grouper(freq="W"))
            .apply(lambda x: x)
        )
        self._monthly_returns = (
            self._daily_historical["Return"]
            .groupby(pd.Grouper(freq="M"))
            .apply(lambda x: x)
        )
        self._quarterly_returns = (
            self._daily_historical["Return"]
            .groupby(pd.Grouper(freq="Q"))
            .apply(lambda x: x)
        )
        self._yearly_returns = (
            self._daily_historical["Return"]
            .groupby(pd.Grouper(freq="Y"))
            .apply(lambda x: x)
        )

    @handle_errors
    def get_value_at_risk(
        self,
        period: str = "yearly",
        alpha: float = 0.05,
        within_period: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
        distribution: str = "historic",
    ):
        """
        Calculate the Value at Risk (VaR) of an investment portfolio or asset's returns.

        Value at Risk (VaR) is a risk management metric that quantifies the maximum potential loss
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The VaR is calculated as the quantile of the return distribution, representing the loss threshold
        that is not expected to be exceeded with a given confidence level (e.g., 5% for alpha=0.05).

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for VaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate VaR within the specified period or for the entire
            period. Thus whether to look at the VaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            distribution (str): The distribution to use for the VaR calculations (historic, gaussian, cf or studentt). Defaults to "historic".

        Returns:
            pd.Series: VaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

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
        returns = handle_return_data_periods(self, period, within_period)

        if distribution == "historic":
            value_at_risk = risk.get_var_historic(returns, alpha)
        elif distribution == "gaussian":
            value_at_risk = risk.get_var_gaussian(returns, alpha)
        elif distribution == "cf":
            value_at_risk = risk.get_var_gaussian(returns, alpha, True)
        elif distribution == "studentt":
            value_at_risk = risk.get_var_studentt(returns, alpha)
        else:
            raise ValueError("Distribution must be historic, gaussian, cf or studentt.")

        if growth:
            return calculate_growth(
                value_at_risk,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return value_at_risk.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_conditional_value_at_risk(
        self,
        period: str = "yearly",
        alpha: float = 0.05,
        within_period: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
        distribution: str = "historic",
    ):
        """
        Calculate the Conditional Value at Risk (CVaR) of an investment portfolio or asset's returns.

        Conditional Value at Risk (CVaR) is a risk management metric that quantifies the loss in the worst % of cases of
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The CVaR is calculated as the expected loss given that the loss threshold (VaR) with a given confidence level (e.g., 5% for alpha=0.05) is excceeded.

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
            distribution (str): The distribution to use for the CVaR calculations (historic, gaussian, studentt, laplace or logistic). Defaults to "historic".

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

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
        returns = handle_return_data_periods(self, period, within_period)

        if distribution == "historic":
            conditional_value_at_risk = risk.get_cvar_historic(returns, alpha)
        elif distribution == "gaussian":
            conditional_value_at_risk = risk.get_cvar_gaussian(returns, alpha)
        elif distribution == "studentt":
            conditional_value_at_risk = risk.get_var_studentt(returns, alpha)
        elif distribution == "laplace":
            conditional_value_at_risk = risk.get_cvar_laplace(returns, alpha)
        elif distribution == "logistic":
            conditional_value_at_risk = risk.get_cvar_logistic(returns, alpha)
        else:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, laplace or logistic."
            )

        if growth:
            return calculate_growth(
                conditional_value_at_risk,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return conditional_value_at_risk.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_maximum_drawdown(
        self,
        period: str = "yearly",
        alpha: float = 0.05,
        within_period: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Maximum Drawdown (MDD) of an investment portfolio or asset's returns.

        Maximum Drawdown (MDD) is a risk management metric that quantifies the largest historical loss of an investment portfolio or asset experienced over a specified time horizon.
        It provides insights into the downside risk associated with an investment and helps investors make informed decisions about risk tolerance.

        The CVaR is calculated as the expected loss given that the loss threshold (VaR) with a given confidence level (e.g., 5% for alpha=0.05) is excceeded.

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

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

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
        returns = handle_return_data_periods(self, period, within_period)

        maximum_drawdown = risk.get_max_drawdown(returns)

        if growth:
            return calculate_growth(
                maximum_drawdown,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return maximum_drawdown.round(rounding if rounding else self._rounding)
