"""Taffler Module"""

__docformat__ = "google"

import pandas as pd


def get_profit_before_tax_to_current_liabilities_ratio(
    profit_before_tax: float | pd.Series | pd.DataFrame,
    current_liabilities: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Profit Before Tax to Current Liabilities Ratio measures how well a company's pre-tax earnings cover its
    short-term obligations. It is the profitability component (X1) of the Taffler z-score and carries about
    53% of the model's discriminating power.

    The formula is as follows:

        - Profit Before Tax to Current Liabilities Ratio = Profit Before Tax / Current Liabilities

    Args:
        profit_before_tax (float | pd.Series | pd.DataFrame): The profit before tax (income before tax).
        current_liabilities (float | pd.Series | pd.DataFrame): The short-term liabilities of a company due
        within one year.

    Returns:
        float | pd.Series | pd.DataFrame: The Profit Before Tax to Current Liabilities Ratio.

    Notes:
    - A low or negative ratio lowers the Taffler z-score.
    """
    return profit_before_tax / current_liabilities


def get_current_assets_to_total_liabilities_ratio(
    current_assets: float | pd.Series | pd.DataFrame,
    total_liabilities: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Current Assets to Total Liabilities Ratio measures how much of a company's liabilities its current
    assets cover. It is the working capital component (X2) of the Taffler z-score.

    The formula is as follows:

        - Current Assets to Total Liabilities Ratio = Current Assets / Total Liabilities

    Args:
        current_assets (float | pd.Series | pd.DataFrame): The assets a company expects to turn into cash
        within one year.
        total_liabilities (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
        liabilities.

    Returns:
        float | pd.Series | pd.DataFrame: The Current Assets to Total Liabilities Ratio.

    Notes:
    - A higher ratio raises the Taffler z-score.
    """
    return current_assets / total_liabilities


def get_current_liabilities_to_total_assets_ratio(
    current_liabilities: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Current Liabilities to Total Assets Ratio measures the share of a company's assets financed by
    short-term obligations. It is the financial risk component (X3) of the Taffler z-score.

    The formula is as follows:

        - Current Liabilities to Total Assets Ratio = Current Liabilities / Total Assets

    Args:
        current_liabilities (float | pd.Series | pd.DataFrame): The short-term liabilities of a company due
        within one year.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Current Liabilities to Total Assets Ratio.

    Notes:
    - A higher ratio lowers the Taffler z-score, as its coefficient is negative.
    """
    return current_liabilities / total_assets


def get_no_credit_interval(
    quick_assets: float | pd.Series | pd.DataFrame,
    current_liabilities: float | pd.Series | pd.DataFrame,
    sales: float | pd.Series | pd.DataFrame,
    profit_before_tax: float | pd.Series | pd.DataFrame,
    depreciation: float | pd.Series | pd.DataFrame,
    days: float = 365,
) -> float | pd.Series | pd.DataFrame:
    """
    The No-Credit Interval is the number of days a company could finance its operations from its quick assets
    after paying its current liabilities, without new credit or revenue. It is the liquidity component (X4) of
    the Taffler z-score, where the daily operating expenses are approximated by sales less profit before tax
    and depreciation, spread over the days of the period.

    The formula is as follows:

        - Daily Operating Expenses = (Sales - Profit Before Tax - Depreciation) / Days
        - No-Credit Interval = (Quick Assets - Current Liabilities) / Daily Operating Expenses

    Args:
        quick_assets (float | pd.Series | pd.DataFrame): The current assets less inventories.
        current_liabilities (float | pd.Series | pd.DataFrame): The short-term liabilities of a company due
        within one year.
        sales (float | pd.Series | pd.DataFrame): The revenue of the period.
        profit_before_tax (float | pd.Series | pd.DataFrame): The profit before tax of the period.
        depreciation (float | pd.Series | pd.DataFrame): The depreciation (and amortization) of the period.
        days (float, optional): The number of days the flows cover. Defaults to 365.

    Returns:
        float | pd.Series | pd.DataFrame: The No-Credit Interval in days.

    Notes:
    - The interval is negative when the current liabilities exceed the quick assets.
    """
    daily_operating_expenses = (sales - profit_before_tax - depreciation) / days

    return (quick_assets - current_liabilities) / daily_operating_expenses


def get_taffler_z_score(
    profit_before_tax_to_current_liabilities_ratio: float | pd.Series | pd.DataFrame,
    current_assets_to_total_liabilities_ratio: float | pd.Series | pd.DataFrame,
    current_liabilities_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    no_credit_interval: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Taffler z-score is a discriminant model of financial distress for UK listed industrial companies,
    estimated by Taffler (1983) and validated over 1979-2003 by Agarwal and Taffler (2007). A negative score
    places a company in the at-risk region.

    The formula is as follows:

        - z = 3.20 + 12.18 X1 + 2.50 X2 - 10.68 X3 + 0.029 X4

    With X1 the Profit Before Tax to Current Liabilities Ratio, X2 the Current Assets to Total Liabilities
    Ratio, X3 the Current Liabilities to Total Assets Ratio and X4 the No-Credit Interval in days.

    Args:
        profit_before_tax_to_current_liabilities_ratio (float | pd.Series | pd.DataFrame): X1, the Profit
        Before Tax to Current Liabilities Ratio.
        current_assets_to_total_liabilities_ratio (float | pd.Series | pd.DataFrame): X2, the Current Assets
        to Total Liabilities Ratio.
        current_liabilities_to_total_assets_ratio (float | pd.Series | pd.DataFrame): X3, the Current
        Liabilities to Total Assets Ratio.
        no_credit_interval (float | pd.Series | pd.DataFrame): X4, the No-Credit Interval in days.

    Returns:
        float | pd.Series | pd.DataFrame: The Taffler z-score.

    Notes:
    - Agarwal and Taffler (2007, Table 1) attribute 53% of the discriminating power to X1, 13% to X2, 18% to
    X3 and 16% to X4 (Mosteller-Wallace contributions).
    - The cut-off is zero: a company with a negative score has a financial profile like that of past
    failures. The model is not meant for financial companies, and its coefficients apply to UK accounts.

    References:
    - Taffler, R. J. (1983). The Assessment of Company Solvency and Performance Using a Statistical Model.
    Accounting and Business Research, 13(52), 295-308.
    - Agarwal, V. and Taffler, R. J. (2007). Twenty-Five Years of the Taffler z-Score Model: Does It Really
    Have Predictive Ability? Accounting and Business Research, 37(4), 285-300.
    """
    return (
        3.20
        + 12.18 * profit_before_tax_to_current_liabilities_ratio
        + 2.50 * current_assets_to_total_liabilities_ratio
        - 10.68 * current_liabilities_to_total_assets_ratio
        + 0.029 * no_credit_interval
    )
