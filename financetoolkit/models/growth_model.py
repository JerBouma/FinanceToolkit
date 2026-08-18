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

    Notes:
    - In the textbook derivation of PVGO (e.g. Bodie, Kane & Marcus, "Investments"), the no-growth
    value EPS / r is discounted at r = the cost of equity (the "market capitalization rate" for the
    stock), since both the share Price and EPS are equity-only, per-share quantities. This
    implementation instead discounts at the Weighted Average Cost of Capital (WACC), which blends in
    the (typically lower, tax-shielded) cost of debt. Because WACC is generally below the cost of
    equity, EPS / WACC will generally overstate the no-growth value and therefore understate PVGO
    relative to the textbook formula. This is a practical simplification (WACC is already computed
    elsewhere in this toolkit for every company, whereas a standalone cost of equity series requires
    re-deriving the CAPM inputs); treat the resulting PVGO as an approximation, and prefer comparing
    PVGO *across* companies or over time for the same company rather than reading absolute levels
    literally.
    """
    earnings_wacc_ratio = (earnings_per_share / weighted_average_cost_of_capital).T

    if calculate_daily:
        earnings_wacc_ratio.index = pd.PeriodIndex(earnings_wacc_ratio.index, freq="D")
        earnings_wacc_ratio = pd.DataFrame(
            earnings_wacc_ratio, index=close_prices.index
        ).ffill()

    return close_prices - earnings_wacc_ratio


def get_sustainable_growth_rate(
    return_on_equity: float | pd.Series | pd.DataFrame,
    retention_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Sustainable Growth Rate (SGR) is the maximum rate at which a company can grow its
    revenue, using internally generated funds only, without having to raise additional equity
    or increase its financial leverage.

    The formula is as follows:

        SGR = Return on Equity * Retention Ratio

    Also known as: SGR, self-sustainable growth rate.

    Args:
        return_on_equity (float | pd.Series | pd.DataFrame): The return on equity (ROE) of
        the company.
        retention_ratio (float | pd.Series | pd.DataFrame): The retention ratio of the company,
        i.e. the proportion of net income retained rather than paid out as dividends. This is
        equal to (1 - Dividend Payout Ratio).

    Returns:
        float | pd.Series | pd.DataFrame: The Sustainable Growth Rate.

    Notes:
    - Growing faster than the SGR without external financing typically requires either
    improving profitability, reducing the dividend payout, or increasing leverage.
    """
    return return_on_equity * retention_ratio


def get_internal_growth_rate(
    return_on_assets: float | pd.Series | pd.DataFrame,
    retention_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Internal Growth Rate (IGR) is the maximum rate at which a company can grow its
    revenue using only its retained earnings, without raising any external financing
    (neither debt nor equity).

    The formula is as follows:

        IGR = (Return on Assets * Retention Ratio) / (1 - (Return on Assets * Retention Ratio))

    Also known as: IGR.

    Args:
        return_on_assets (float | pd.Series | pd.DataFrame): The return on assets (ROA) of
        the company.
        retention_ratio (float | pd.Series | pd.DataFrame): The retention ratio of the company,
        i.e. the proportion of net income retained rather than paid out as dividends. This is
        equal to (1 - Dividend Payout Ratio).

    Returns:
        float | pd.Series | pd.DataFrame: The Internal Growth Rate.

    Notes:
    - The IGR is more conservative than the Sustainable Growth Rate (SGR) since it assumes
    no additional debt is raised to fund growth, whereas the SGR assumes the company
    maintains its current level of financial leverage.
    """
    return_on_assets_retained = return_on_assets * retention_ratio

    return return_on_assets_retained / (1 - return_on_assets_retained)
