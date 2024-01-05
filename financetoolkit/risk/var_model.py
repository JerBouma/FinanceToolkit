"""Value at Risk Model"""

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.risk import risk_model

ALPHA_CONSTRAINT = 0.5

# This is meant for calculations in which a Multi Index exists. This is the case
# when calculating a "within period" in which the first index represents the period
# (e.g. 2020Q1) and the second index the days within that period (January to March)
MULTI_PERIOD_INDEX_LEVELS = 2


def get_var_historic(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the historical Value at Risk (VaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_var_historic, alpha=alpha
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            value_at_risk = pd.concat(period_data_list, axis=1)

            return value_at_risk.T

        return returns.aggregate(get_var_historic, alpha=alpha)
    if isinstance(returns, pd.Series):
        return np.percentile(
            returns, alpha * 100
        )  # The actual calculation without data wrangling

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_var_gaussian(
    returns, alpha: float, cornish_fisher: bool = False
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Value at Risk (VaR) of returns based on the gaussian distribution.

    Adjust za according to the Cornish-Fischer expansion of the quantiles if
    Formula for quantile from "Finance Compact Plus" by Zimmerman; Part 1, page 130-131
    More material/resources:
     - "Numerical Methods and Optimization in Finance" by Gilli, Maringer & Schumann;
     - https://www.value-at-risk.net/the-cornish-fisher-expansion/;
     - https://www.diva-portal.org/smash/get/diva2:442078/FULLTEXT01.pdf, Section 2.4.2, p.18;
     - "Risk Management and Financial Institutions" by John C. Hull

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        cornish_fisher (bool): Whether to adjust the distriution for the skew and kurtosis of the returns
        based on the Cornish-Fischer quantile expansion. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_gaussian(
                returns.loc[sub_period], alpha, cornish_fisher
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    za = stats.norm.ppf(alpha, 0, 1)

    if cornish_fisher:
        S = risk_model.get_skewness(returns)
        K = risk_model.get_kurtosis(returns)
        za = (
            za
            + (za**2 - 1) * S / 6
            + (za**3 - 3 * za) * (K - 3) / 24
            - (2 * za**3 - 5 * za) * (S**2) / 36
        )

    return returns.mean() + za * returns.std(ddof=0)


def get_var_studentt(returns, alpha: float) -> pd.Series | pd.DataFrame:
    """
    Calculate the Value at Risk (VaR) of returns based on the Student-T distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_studentt(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    # Fitting Student-T parameters to the data
    if isinstance(returns, pd.Series):
        v = np.array([stats.t.fit(returns)[0]])
    else:
        v = np.array([stats.t.fit(returns[col])[0] for col in returns.columns])
    za = stats.t.ppf(alpha, v, 1)

    return np.sqrt((v - 2) / v) * za * returns.std(ddof=0) + returns.mean()
