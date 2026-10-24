"""Credit Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

# pylint: disable=too-many-arguments

# Damodaran treats a company with a market capitalization above five billion US dollars as a
# large firm in his synthetic rating tables.
LARGE_FIRM_MARKET_CAP = 5e9


def get_merton_model(
    equity_value: float,
    equity_volatility: float,
    default_point: float,
    risk_free_rate: float,
    time_horizon: float = 1.0,
) -> pd.Series:
    """
    The Merton (1974) structural credit model. Equity is a call option on the company's
    assets with the debt as strike price, so the market value and volatility of the assets,
    which cannot be observed, follow from those of the equity by solving

        E = V N(d1) - D e^(-rT) N(d2)  and  σ_E E = N(d1) σ_V V,

    with d1 = (ln(V/D) + (r + σ_V²/2) T) / (σ_V √T) and d2 = d1 - σ_V √T. The distance to
    default d2 is the number of standard deviations the assets are above the default point
    after the time horizon, and N(-d2) the probability that they fall below it.

    The default point is usually set at short-term debt plus half of long-term debt (the
    KMV convention of Crosbie and Bohn, 2003), as long-term debt need not be repaid within
    the horizon.

    Args:
        equity_value (float): The market value of the equity (market capitalization).
        equity_volatility (float): The annualized volatility of the equity returns.
        default_point (float): The value of the debt at which the company defaults.
        risk_free_rate (float): The annualized risk free rate.
        time_horizon (float, optional): The horizon in years. Defaults to 1.

    Returns:
        pd.Series: The Asset Value, Asset Volatility, Default Point, Distance to Default and
        Probability of Default.
    """
    result = pd.Series(
        {
            "Asset Value": np.nan,
            "Asset Volatility": np.nan,
            "Default Point": default_point,
            "Distance to Default": np.nan,
            "Probability of Default": np.nan,
        }
    )

    if (
        not np.isfinite(
            [equity_value, equity_volatility, default_point, risk_free_rate]
        ).all()
        or equity_value <= 0
        or equity_volatility <= 0
    ):
        return result

    if default_point <= 0:
        # Without debt there is no default point to fall below.
        result["Asset Value"] = equity_value
        result["Asset Volatility"] = equity_volatility
        result["Probability of Default"] = 0.0
        return result

    root_t = np.sqrt(time_horizon)
    discount = np.exp(-risk_free_rate * time_horizon)

    def equations(variables):
        asset_value, asset_volatility = variables
        asset_value = max(asset_value, 1e-9)
        asset_volatility = max(asset_volatility, 1e-9)
        d1 = (
            np.log(asset_value / default_point)
            + (risk_free_rate + asset_volatility**2 / 2) * time_horizon
        ) / (asset_volatility * root_t)
        d2 = d1 - asset_volatility * root_t
        return [
            (asset_value * norm.cdf(d1) - default_point * discount * norm.cdf(d2))
            / equity_value
            - 1,
            norm.cdf(d1)
            * asset_volatility
            * asset_value
            / (equity_volatility * equity_value)
            - 1,
        ]

    start = [
        equity_value + default_point,
        equity_volatility * equity_value / (equity_value + default_point),
    ]
    solution, _, converged, _ = fsolve(equations, start, full_output=True)

    if converged != 1:
        return result

    asset_value, asset_volatility = solution
    distance = (
        np.log(asset_value / default_point)
        + (risk_free_rate - asset_volatility**2 / 2) * time_horizon
    ) / (asset_volatility * root_t)

    result["Asset Value"] = asset_value
    result["Asset Volatility"] = asset_volatility
    result["Distance to Default"] = distance
    result["Probability of Default"] = norm.cdf(-distance)

    return result


def get_synthetic_rating(
    interest_coverage_ratio: float, rating_table: pd.DataFrame
) -> tuple[str, float]:
    """
    The bond rating and default spread that an interest coverage ratio (EBIT divided by
    interest expense) maps to in a synthetic rating table (Damodaran).

    Args:
        interest_coverage_ratio (float): The interest coverage ratio. Infinity (no interest
            expense) maps to the best rating; a ratio below the table's lowest bound maps to
            the worst.
        rating_table (pd.DataFrame): The table of one firm type, with the Lower and Upper
            bound of the ratio, the Rating and the Default Spread per rating.

    Returns:
        tuple[str, float]: The rating and its default spread; ("", NaN) for a missing ratio.
    """
    if interest_coverage_ratio is None or np.isnan(interest_coverage_ratio):
        return "", np.nan

    table = rating_table.sort_values("Lower")
    # Clipped to the table's range, as a ratio outside it (a large operating loss, or no
    # interest expense at all) belongs to the worst or best rating.
    ratio = float(
        np.clip(
            interest_coverage_ratio,
            float(table["Lower"].min()),
            float(table["Upper"].max()),
        )
    )
    match = table[table["Lower"] <= ratio].iloc[-1]

    return str(match["Rating"]), float(match["Default Spread"])
