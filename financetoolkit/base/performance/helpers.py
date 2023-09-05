"""Performance Helpers Module"""
__docformat__ = "google"


# pylint: disable=protected-access


def handle_return_data_periods(self, period: str, within_period: bool):
    """
    This function is a specific function solely related to the Ratios controller. It
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
    if period == "weekly":
        return (
            self._weekly_within_historical_data
            if within_period
            else self._weekly_historical_data
        )
    if period == "monthly":
        return (
            self._monthly_within_historical_data
            if within_period
            else self._monthly_historical_data
        )
    if period == "quarterly":
        return (
            self._quarterly_within_historical_data
            if within_period
            else self._quarterly_historical_data
        )
    if period == "yearly":
        return (
            self._yearly_within_historical_data
            if within_period
            else self._yearly_historical_data
        )

    raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")


def handle_risk_free_data_periods(self, period: str):
    """
    This function is a specific function solely related to the Ratios controller. It
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
        return self._daily_risk_free_rate_data
    if period == "weekly":
        return self._weekly_risk_free_rate_data
    if period == "monthly":
        return self._monthly_risk_free_rate_data
    if period == "quarterly":
        return self._quarterly_risk_free_rate_data
    if period == "yearly":
        return self._yearly_risk_free_rate_data

    raise ValueError("Period must be daily, monthly, weekly, quarterly, or yearly.")
