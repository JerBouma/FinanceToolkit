"""Easton Implied Cost of Capital Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd


def get_implied_cost_of_capital(
    earnings_per_share_one: pd.Series | pd.DataFrame | float,
    earnings_per_share_two: pd.Series | pd.DataFrame | float,
    share_price: pd.Series | pd.DataFrame | float,
    dividends_per_share: pd.Series | pd.DataFrame | float,
    modified: bool = True,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate Easton's (2004) implied cost of equity capital.

    It is the expected rate of return at which the share price equals the capitalized
    next-year earnings plus the capitalized abnormal growth in earnings of the year after,
    solved from two years of earnings forecasts and the share price.

    The formula is as follows:

        - PEG: r = √((EPS₂ − EPS₁) / P₀)
        - MPEG: r = A + √(A² + (EPS₂ − EPS₁) / P₀), with A = DPS₁ / (2 × P₀)

    Where EPS₁ and EPS₂ are the expected earnings per share of the next two years, P₀ the
    share price today and DPS₁ the expected dividends per share of the next year. The MPEG
    rate is the positive root of Easton's valuation equation

        - P₀ = EPS₁ / r + (EPS₂ + r × DPS₁ − (1 + r) × EPS₁) / r²
        - P₀ = (EPS₂ + r × DPS₁ − EPS₁) / r²

    the capitalized next-year earnings plus the capitalized abnormal growth in earnings, which
    assumes the abnormal growth in earnings stays constant after the second year. The
    PEG rate leaves out the dividends (DPS₁ = 0), so it ignores the earnings on reinvested
    dividends; the MPEG rate is the one Easton prefers. The rate is a yearly rate, annually
    compounded, as are the earnings and dividends it is based on.

    See: Easton, P. D. (2004). PE Ratios, PEG Ratios, and Estimating the Implied Expected Rate
    of Return on Equity Capital. The Accounting Review, 79(1), 73-95.

    Args:
        earnings_per_share_one (pd.Series | pd.DataFrame | float): The expected earnings per
            share of the next year (EPS₁).
        earnings_per_share_two (pd.Series | pd.DataFrame | float): The expected earnings per
            share of the year after (EPS₂).
        share_price (pd.Series | pd.DataFrame | float): The share price today (P₀).
        dividends_per_share (pd.Series | pd.DataFrame | float): The expected dividends per share
            of the next year (DPS₁), used by the MPEG rate only.
        modified (bool, optional): Whether to return the MPEG rate (with dividends) instead of
            the PEG rate. Defaults to True.

    Returns:
        pd.Series | pd.DataFrame | float: The implied cost of equity capital as a decimal, NaN
            where the earnings do not grow (EPS₂ ≤ EPS₁), the share price is not positive or
            the root's argument is not positive.
    """
    earnings_growth = earnings_per_share_two - earnings_per_share_one
    if not isinstance(share_price, pd.Series | pd.DataFrame) and not share_price > 0:
        return np.nan

    half_dividend_yield = dividends_per_share / (2 * share_price) if modified else 0.0

    # The positive root of Easton's quadratic in r. Only growing earnings and a positive price give a real, positive rate; anything else has no implied rate in the model and is NaN, never 0.  # noqa: E501
    argument = half_dividend_yield**2 + earnings_growth / share_price
    valid = (earnings_growth > 0) & (share_price > 0) & (argument > 0)

    if isinstance(argument, pd.Series | pd.DataFrame):
        return half_dividend_yield + np.sqrt(argument.where(valid))

    return float(half_dividend_yield + np.sqrt(argument)) if valid else np.nan
