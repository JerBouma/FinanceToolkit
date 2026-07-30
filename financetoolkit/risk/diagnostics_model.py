"""Risk Diagnostics Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.risk import risk_model

# This is meant for calculations in which a Multi Index exists. This is the case
# when calculating a "within period" in which the first index represents the period
# (e.g. 2020Q1) and the second index the days within that period (January to March)
MULTI_PERIOD_INDEX_LEVELS = 2


def get_arch_lm_test(
    returns: pd.Series | pd.DataFrame, lags: int = 5
) -> pd.Series | pd.DataFrame:
    """
    Calculate Engle's Lagrange Multiplier (LM) test for ARCH effects.

    The test regresses squared, mean-demeaned returns on `lags` of themselves and
    tests whether the resulting R-squared is significantly different from zero. Under
    the null hypothesis of no ARCH effects, `n * R-squared` is asymptotically
    chi-squared distributed with `lags` degrees of freedom.

    A significant result (low p-value) indicates that the return series exhibits
    volatility clustering, and a GARCH-family model is an appropriate choice for it.

    For more information about the method, see the following paper:

    - Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates
    of the Variance of United Kingdom Inflation." Econometrica, 50(4), 987-1008.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        lags (int): The number of lags to test for ARCH effects. Defaults to 5.

    Returns:
        pd.Series | pd.DataFrame: The ARCH-LM statistic and its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_arch_lm_test(returns.loc[sub_period], lags=lags)

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_arch_lm_test(returns[column], lags=lags)
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        residuals = (returns - returns.mean()).dropna().to_numpy()
        squared_residuals = residuals**2
        n = len(squared_residuals)

        if n <= lags + 1:
            return pd.Series({"ARCH-LM Statistic": np.nan, "P-Value": np.nan})

        y = squared_residuals[lags:]
        x = np.column_stack(
            [squared_residuals[lags - lag - 1 : n - lag - 1] for lag in range(lags)]
        )
        x = np.column_stack([np.ones(len(y)), x])

        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        fitted_values = x @ coefficients

        residual_sum_of_squares = np.sum((y - fitted_values) ** 2)
        total_sum_of_squares = np.sum((y - y.mean()) ** 2)

        r_squared = (
            1 - residual_sum_of_squares / total_sum_of_squares
            if total_sum_of_squares > 0
            else 0.0
        )

        lm_statistic = len(y) * r_squared
        p_value = stats.chi2.sf(lm_statistic, lags)

        return pd.Series({"ARCH-LM Statistic": lm_statistic, "P-Value": p_value})

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_jarque_bera_test(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Jarque-Bera test for normality.

    The test combines sample skewness and excess kurtosis into a single statistic
    that is asymptotically chi-squared distributed with 2 degrees of freedom under
    the null hypothesis that the data is normally distributed:

    - JB = (n / 6) * (S^2 + (K^2) / 4)

    Where `S` is the skewness, `K` is the excess (Fisher) kurtosis and `n` is the
    number of observations.

    A significant result (low p-value) indicates that returns are not normally
    distributed, which is relevant when choosing between e.g. gaussian and Student-T
    based Value at Risk models.

    For more information about the method, see the following paper:

    - Jarque, C.M. and Bera, A.K. (1987). "A Test for Normality of Observations and
    Regression Residuals." International Statistical Review, 55(2), 163-172.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: The Jarque-Bera statistic and its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_jarque_bera_test(returns.loc[sub_period])

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_jarque_bera_test(returns[column])
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        n = returns.count()
        skewness = risk_model.get_skewness(returns)
        excess_kurtosis = risk_model.get_kurtosis(returns, fisher=True)

        jarque_bera_statistic = (n / 6) * (skewness**2 + (excess_kurtosis**2) / 4)
        p_value = stats.chi2.sf(jarque_bera_statistic, 2)

        return pd.Series(
            {"Jarque-Bera Statistic": jarque_bera_statistic, "P-Value": p_value}
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")
