"""Cash Flow Return on Investment Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# The range the internal rate of return is searched in: from a loss of 99% a year to a
# return of 1,000% a year.
LOWEST_RATE = -0.99
HIGHEST_RATE = 10.0


def get_cash_flow_return_on_investment(
    gross_investment: pd.DataFrame,
    gross_cash_flow: pd.DataFrame,
    non_depreciating_assets: pd.DataFrame,
    asset_life: pd.DataFrame,
) -> pd.DataFrame:
    """
    The Cash Flow Return on Investment (CFROI) of Madden.

    It is the internal rate of return of the company's assets as one project, where the
    gross investment buys the gross cash flow for every year of the asset life and the
    non-depreciating assets (working capital, land) are released at the end of it.

    The formula is as follows, solved for the CFROI:

        - Gross Investment = Σ (t = 1 .. N) Gross Cash Flow / (1 + CFROI)^t + Non-Depreciating
          Assets / (1 + CFROI)^N

    with N the asset life in years, a fractional life taking the annuity formula
    Gross Cash Flow * (1 - (1 + CFROI)^(-N)) / CFROI.

    Args:
        gross_investment (pd.DataFrame): The gross investment: the depreciating assets at their
            original cost plus the non-depreciating assets.
        gross_cash_flow (pd.DataFrame): The yearly gross cash flow (after-tax cash flow before
            depreciation and interest).
        non_depreciating_assets (pd.DataFrame): The assets released at the end of the life.
        asset_life (pd.DataFrame): The life of the depreciating assets in years.

    Returns:
        pd.DataFrame: The CFROI as a decimal per year (annually compounded), NaN where an input
        is missing, the life or the gross investment is not positive, or no rate between -99% and
        1,000% solves the equation.

    References:
    - Madden, B. J. (1999). "CFROI Valuation: A Total System Approach to Valuing the Firm."
      Butterworth-Heinemann.
    """
    result = pd.DataFrame(
        np.nan, index=gross_investment.index, columns=gross_investment.columns
    )
    frames = [gross_cash_flow, non_depreciating_assets, asset_life]
    frames = [
        frame.reindex(index=gross_investment.index, columns=gross_investment.columns)
        for frame in frames
    ]

    for row in gross_investment.index:
        for column in gross_investment.columns:
            investment = gross_investment.loc[row, column]
            cash_flow, released, life = (frame.loc[row, column] for frame in frames)
            values = np.array([investment, cash_flow, released, life], dtype=float)
            if not np.isfinite(values).all() or life <= 0 or investment <= 0:
                continue

            def net_present_value(rate: float, flows: np.ndarray) -> float:
                investment, cash_flow, released, life = flows
                # The annuity factor of a fractional life, the life itself at a rate of zero.
                annuity = life if rate == 0 else (1 - (1 + rate) ** -life) / rate
                return cash_flow * annuity + released * (1 + rate) ** -life - investment

            low = net_present_value(LOWEST_RATE, values)
            high = net_present_value(HIGHEST_RATE, values)
            if np.sign(low) == np.sign(high):
                continue
            result.loc[row, column] = brentq(
                net_present_value, LOWEST_RATE, HIGHEST_RATE, args=(values,), xtol=1e-12
            )

    return result
