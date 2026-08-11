"""Entropic Value at Risk Model"""

import numpy as np
import pandas as pd

ALPHA_CONSTRAINT = 0.5

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2


def get_evar_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Entropic Value at Risk (EVaR) of returns based on the gaussian distribution.

    For more information see: https://en.wikipedia.org/wiki/Entropic_value_at_risk

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: EVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_evar_gaussian(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    # The entropic bound is expressed on losses; subtracting it keeps the result a
    # negative return, consistent with the VaR and CVaR functions in this module.
    return returns.mean() - returns.std(ddof=0) * np.sqrt(-2 * np.log(alpha))
