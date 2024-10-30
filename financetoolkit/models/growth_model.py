"""Growth Model"""

__docformat__ = "google"

import pandas as pd

# pylint: disable=too-many-locals


def get_present_value_of_growth_opportunities(
    weighted_average_cost_of_capital: pd.DataFrame,
    earnings_per_share: pd.DataFrame,
    close_prices: pd.DataFrame,
    calculate_daily: bool = False,
) -> pd.DataFrame:
    """
    The Present Value of Growth Opportunities (PVGO) is the net present value of all future investments a company is
    expected to make. It is calculated as the difference between the Close Price and the Earnings Per Share divided by
    the Weighted Average Cost of Capital.

    It is meant to be used as relative valuation metric and therefore doesn't necessarily have a meaning when used
    for one company.


    The formula is as follows:

        PVGO = Close Price — (Earnings Per Share / Weighted Average Cost of Capital)

    Args:
        weighted_average_cost_of_capital (pd.DataFrame): The weighted average cost of capital.
        earnings_per_share (pd.DataFrame): The earnings per share.
        close_prices (pd.DataFrame): The close prices.
        calculate_daily (bool): Whether to calculate the PVGO on a daily basis. If False, the PVGO is calculated
        based on the provided close_prices DataFrame.

    Returns:
        pd.DataFrame: The PVGO.
    """
    earnings_wacc_ratio = (earnings_per_share / weighted_average_cost_of_capital).T

    if calculate_daily:
        earnings_wacc_ratio.index = pd.PeriodIndex(earnings_wacc_ratio.index, freq="D")
        earnings_wacc_ratio = pd.DataFrame(
            earnings_wacc_ratio, index=close_prices.index
        ).ffill()

    return close_prices - earnings_wacc_ratio
