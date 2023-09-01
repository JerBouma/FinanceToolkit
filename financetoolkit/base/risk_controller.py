"""Risk Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import calculate_growth, handle_errors
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
        period: str = "quarterly",
        alpha: float = 0.05,
        within_period: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
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
            Defaults to "daily".
            alpha (float, optional): The confidence level for VaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate VaR within the specified period or for the entire
            period. Thus whether to look at the VaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

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
        if period == "daily":
            returns = (
                self._daily_returns
                if within_period
                else self._daily_historical["Return"]
            )
        elif period == "weekly":
            returns = (
                self._weekly_returns
                if within_period
                else self._weekly_historical["Return"]
            )
        elif period == "monthly":
            returns = (
                self._monthly_returns
                if within_period
                else self._monthly_historical["Return"]
            )
        elif period == "quarterly":
            returns = (
                self._quarterly_returns
                if within_period
                else self._quarterly_historical["Return"]
            )
        elif period == "yearly":
            returns = (
                self._yearly_returns
                if within_period
                else self._yearly_historical["Return"]
            )
        else:
            raise ValueError(
                "Period must be daily, monthly, weekly, quarterly, or yearly."
            )

        if returns.index.nlevels > 1:
            periods = returns.index.get_level_values(0).unique()
            value_at_risk = pd.DataFrame()

            for sub_period in periods:
                period_data = risk.get_var(returns.loc[sub_period], 0.05)
                period_data.name = sub_period

                value_at_risk = pd.concat([value_at_risk, period_data], axis=1)
        else:
            value_at_risk = risk.get_var(returns, alpha)

        # Transpose the DataFrame to have time as the index
        value_at_risk = value_at_risk.T

        if growth:
            return calculate_growth(
                value_at_risk,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return value_at_risk.round(rounding if rounding else self._rounding)
