"""Shumway Hazard Model Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.models.helpers import values_on_or_before

# Shumway (2001), Table 6: the hazard models with market-driven variables, estimated on every
# firm year of NYSE and AMEX companies from 1962 to 1992. Panel B combines the market
# variables with two accounting ratios of Zmijewski's model, panel A uses market variables
# only.
COEFFICIENTS = {
    "combined": {
        "Constant": -13.303,
        "NI/TA": -1.982,
        "TL/TA": 3.593,
        "RELSIZE": -0.467,
        "EXRET": -1.809,
        "SIGMA": 5.791,
    },
    "market": {
        "Constant": -12.027,
        "NI/TA": 0.0,
        "TL/TA": 0.0,
        "RELSIZE": -0.503,
        "EXRET": -2.072,
        "SIGMA": 9.834,
    },
}

# Shumway truncates every variable at its 1st and 99th percentile over the sample, which a
# calculation per company cannot repeat: the variables are clipped instead to the range they
# span in the paper's sample after truncation (Table 1, minimum and maximum).
TRUNCATION_BOUNDS = {
    "NI/TA": (-0.329, 0.231),
    "TL/TA": (0.090, 1.028),
    "RELSIZE": (-13.616, -6.115),
    "EXRET": (-0.817, 2.181),
    "SIGMA": (0.030, 0.325),
}


def get_truncated_variable(values: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Clips a Shumway (2001) variable to the range it spans in the paper's sample (Table 1,
    after truncation at the 1st and 99th percentiles).

    Args:
        values (pd.DataFrame): The variable per ticker and period.
        variable (str): The variable. Options are: "NI/TA", "TL/TA", "RELSIZE", "EXRET", "SIGMA".

    Returns:
        pd.DataFrame: The clipped variable.
    """
    lower, upper = TRUNCATION_BOUNDS[variable]

    return values.clip(lower=lower, upper=upper)


def get_monthly_returns(
    prices: pd.DataFrame,
    dates: pd.DataFrame,
    months: int = 12,
) -> list[pd.DataFrame]:
    """
    The simple returns of every month before a set of dates, where month k is the month that
    ends k months before the date (k = 0 being the month that ends on the date itself).

    A price is the last one on or before a day, at most ten days earlier.

    Args:
        prices (pd.DataFrame): The daily total return prices (Adj Close or a total return
            index) with a DatetimeIndex and the tickers as columns.
        dates (pd.DataFrame): The dates per ticker (rows) and period (columns).
        months (int, optional): The number of months. Defaults to 12.

    Returns:
        list[pd.DataFrame]: The return of each month per ticker and period, the latest month
        first.
    """
    returns = []
    for month in range(months):
        end = dates.apply(lambda column, k=month: column - pd.DateOffset(months=k))
        begin = dates.apply(
            lambda column, k=month: column - pd.DateOffset(months=k + 1)
        )
        returns.append(
            values_on_or_before(prices, end) / values_on_or_before(prices, begin) - 1
        )

    return returns


def get_excess_return(
    company_returns: list[pd.DataFrame], market_returns: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    The past excess return of Shumway (2001).

    It is the return of the company over the months (cumulated from the monthly returns)
    minus the return of the market over the same months.

        - EXRET = Π (1 + r(month)) - Π (1 + m(month))

    Args:
        company_returns (list[pd.DataFrame]): The monthly returns of the companies.
        market_returns (list[pd.DataFrame]): The monthly returns of the market, per ticker and
            period on the same dates.

    Returns:
        pd.DataFrame: The excess return, NaN where a month is missing.
    """
    company = np.prod([1 + frame for frame in company_returns], axis=0)
    market = np.prod([1 + frame for frame in market_returns], axis=0)
    template = company_returns[0]

    return pd.DataFrame(
        company - market, index=template.index, columns=template.columns
    )


def get_residual_sigma(
    company_returns: list[pd.DataFrame], market_returns: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    The idiosyncratic volatility of Shumway (2001).

    It is the standard deviation of the residual of a regression of the company's monthly
    returns on the market's monthly returns over the months, the standard error of the
    regression:

        - r(month) = a + b * m(month) + e(month)
        - SIGMA = sqrt(Σ e² / (N - 2))

    Args:
        company_returns (list[pd.DataFrame]): The monthly returns of the companies.
        market_returns (list[pd.DataFrame]): The monthly returns of the market, per ticker and
            period on the same dates.

    Returns:
        pd.DataFrame: The monthly residual standard deviation, NaN unless every month has a
        return (Shumway drops values from fewer than 12 months).
    """
    company = np.stack([frame.to_numpy(dtype=float) for frame in company_returns])
    market = np.stack([frame.to_numpy(dtype=float) for frame in market_returns])
    observations = company.shape[0]
    template = company_returns[0]
    sigma = np.full(template.shape, np.nan)

    complete = np.isfinite(company).all(axis=0) & np.isfinite(market).all(axis=0)
    for row, column in zip(*np.nonzero(complete)):
        x, y = market[:, row, column], company[:, row, column]
        x_demeaned = x - x.mean()
        variance = (x_demeaned**2).sum()
        if variance == 0:
            continue
        slope = (x_demeaned * (y - y.mean())).sum() / variance
        residuals = y - y.mean() - slope * x_demeaned
        sigma[row, column] = np.sqrt((residuals**2).sum() / (observations - 2))

    return pd.DataFrame(sigma, index=template.index, columns=template.columns)


def get_shumway_score(
    net_income_to_total_assets: pd.DataFrame | float,
    total_liabilities_to_total_assets: pd.DataFrame | float,
    relative_size: pd.DataFrame,
    excess_return: pd.DataFrame,
    sigma: pd.DataFrame,
    method: str = "combined",
) -> pd.DataFrame:
    """
    The hazard score of Shumway (2001), Table 6: the logit of the probability of bankruptcy
    in the following year.

        - Combined (panel B): H = -13.303 - 1.982 NI/TA + 3.593 TL/TA - 0.467 RELSIZE - 1.809
          EXRET + 5.791 SIGMA
        - Market (panel A): H = -12.027 - 0.503 RELSIZE - 2.072 EXRET + 9.834 SIGMA

    Args:
        net_income_to_total_assets (pd.DataFrame | float): NI/TA (unused by the market model).
        total_liabilities_to_total_assets (pd.DataFrame | float): TL/TA (unused by the market
            model).
        relative_size (pd.DataFrame): RELSIZE, ln(Market Value of Equity / Total Market Value).
        excess_return (pd.DataFrame): EXRET, the past year's return above the market's.
        sigma (pd.DataFrame): SIGMA, the monthly idiosyncratic volatility.
        method (str, optional): The variant of the hazard model. Options are: "combined" or
            "market". Defaults to "combined".

    Returns:
        pd.DataFrame: The hazard score H.
    """
    coefficients = COEFFICIENTS[method]
    score = (
        coefficients["Constant"]
        + coefficients["RELSIZE"] * relative_size
        + coefficients["EXRET"] * excess_return
        + coefficients["SIGMA"] * sigma
    )
    if method == "combined":
        score = (
            score
            + coefficients["NI/TA"] * net_income_to_total_assets
            + coefficients["TL/TA"] * total_liabilities_to_total_assets
        )

    return score


def get_shumway_bankruptcy_probability(score: pd.DataFrame) -> pd.DataFrame:
    """
    The probability of bankruptcy in the following year from the hazard score:
    e^H / (1 + e^H).

    Args:
        score (pd.DataFrame): The hazard score H.

    Returns:
        pd.DataFrame: The probability as a decimal.
    """
    return 1 / (1 + np.exp(-score))
