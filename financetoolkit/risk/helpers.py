"""Risk Helpers Module"""
__docformat__ = "google"


# pylint: disable=protected-access


def handle_return_data_periods(self, period: str, within_period: bool):
    """
    This function is a specific function solely related to the Risk controller. It
    therefore also requires a self instance to exists with specific parameters.

    Args:
        period (str): the period to return the data for.
        within_period (bool): whether to return the data within the period or the
        entire period.

    Raises:
        ValueError: if the period is not daily, monthly, weekly, quarterly, or yearly.

    Returns:
        pd.Series: the returns for the period.
    """
    if period == "daily":
        return (
            self._daily_returns if within_period else self._daily_historical["Return"]
        )
    if period == "weekly":
        return (
            self._weekly_returns if within_period else self._weekly_historical["Return"]
        )
    if period == "monthly":
        return (
            self._monthly_returns
            if within_period
            else self._monthly_historical["Return"]
        )
    if period == "quarterly":
        return (
            self._quarterly_returns
            if within_period
            else self._quarterly_historical["Return"]
        )
    if period == "yearly":
        return (
            self._yearly_returns if within_period else self._yearly_historical["Return"]
        )

    raise ValueError("Period must be daily, monthly, weekly, quarterly, or yearly.")
