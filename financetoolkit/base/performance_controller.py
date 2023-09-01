"""Performance Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import calculate_growth, handle_errors
from financetoolkit.performance import risk_return

# pylint: disable=too-many-instance-attributes,too-few-public-methods


class Performance:
    """
    Performance Controller Class
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
        Initializes the Performance Controller Class.
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

    @handle_errors
    def get_sharpe_ratio(
        self,
        period: str = "quarterly",
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

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Sharpe ratio values.

        Notes:
        - Daily Sharpe Ratio is not an option asthe standard deviation for 1 days is close to zero. Therefore, it does not
        give any useful insights.
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
            raise ValueError(
                "Daily Sharpe Ratio is not an option as standard deviation for 1 day "
                "is close to zero. Therefore, it does not give any useful insights."
            )

        if period == "weekly":
            historical_data = self._weekly_historical
        elif period == "quarterly":
            historical_data = self._quarterly_historical
        elif period == "yearly":
            historical_data = self._yearly_historical
        else:
            raise ValueError("Period must be weekly, quarterly, or yearly.")

        excess_return = historical_data["Excess Return"]  # type: ignore
        excess_volatility = historical_data["Excess Volatility"]  # type: ignore

        sharpe_ratio = risk_return.get_sharpe_ratio(excess_return, excess_volatility)

        if growth:
            return calculate_growth(
                sharpe_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return sharpe_ratio.round(rounding if rounding else self._rounding)
