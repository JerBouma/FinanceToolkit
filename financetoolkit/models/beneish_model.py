"""Beneish Module"""

__docformat__ = "google"

import pandas as pd

# pylint: disable=too-many-arguments,too-many-positional-arguments


def get_days_sales_in_receivables_index(
    net_receivables: pd.Series | pd.DataFrame,
    revenue: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Days Sales in Receivables Index (DSRI) measures the change in the proportion of
    receivables to sales between the current and prior period. A large increase in DSRI
    can be indicative of revenue inflation through channel stuffing or fictitious sales.

    The formula is as follows:

        - Receivables to Sales = Net Receivables / Revenue
        - DSRI = Receivables to Sales (t) / Receivables to Sales (t-1)

    Args:
        net_receivables (pd.Series | pd.DataFrame): The net receivables of the company.
        revenue (pd.Series | pd.DataFrame): The revenue of the company.

    Returns:
        pd.Series | pd.DataFrame: The Days Sales in Receivables Index (DSRI).

    Notes:
    - A DSRI greater than 1 indicates that receivables are growing faster than sales, which
    can be a sign of earnings manipulation.
    """
    receivables_to_sales = net_receivables / revenue

    return receivables_to_sales / receivables_to_sales.shift(1, axis=1)


def get_gross_margin_index(
    revenue: pd.Series | pd.DataFrame,
    cost_of_goods_sold: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Gross Margin Index (GMI) measures the deterioration of a company's gross margin
    between the prior and current period. A GMI greater than 1 indicates that gross margin
    has deteriorated, which creates an incentive for a company to manipulate earnings.

    The formula is as follows:

        - Gross Margin = (Revenue - Cost of Goods Sold) / Revenue
        - GMI = Gross Margin (t-1) / Gross Margin (t)

    Args:
        revenue (pd.Series | pd.DataFrame): The revenue of the company.
        cost_of_goods_sold (pd.Series | pd.DataFrame): The cost of goods sold of the company.

    Returns:
        pd.Series | pd.DataFrame: The Gross Margin Index (GMI).

    Notes:
    - A GMI greater than 1 indicates that a company's gross margin has worsened, which can be
    a motive for earnings manipulation.
    """
    gross_margin = (revenue - cost_of_goods_sold) / revenue

    return gross_margin.shift(1, axis=1) / gross_margin


def get_asset_quality_index(
    total_current_assets: pd.Series | pd.DataFrame,
    property_plant_and_equipment: pd.Series | pd.DataFrame,
    total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Asset Quality Index (AQI) measures the change in the proportion of non-current,
    non-property, plant and equipment assets (soft assets, e.g. capitalized costs and
    intangibles) relative to total assets. A rising AQI can indicate that a company is
    capitalizing costs it should be expensing in order to inflate earnings.

    The formula is as follows:

        - Asset Quality = 1 - ((Total Current Assets + Property, Plant and Equipment) / Total Assets)
        - AQI = Asset Quality (t) / Asset Quality (t-1)

    Args:
        total_current_assets (pd.Series | pd.DataFrame): The total current assets of the company.
        property_plant_and_equipment (pd.Series | pd.DataFrame): The property, plant and
        equipment of the company.
        total_assets (pd.Series | pd.DataFrame): The total assets of the company.

    Returns:
        pd.Series | pd.DataFrame: The Asset Quality Index (AQI).

    Notes:
    - An AQI greater than 1 indicates an increasing proportion of soft assets, which can be a
    sign of cost capitalization used to inflate earnings.
    """
    asset_quality = 1 - (
        (total_current_assets + property_plant_and_equipment) / total_assets
    )

    return asset_quality / asset_quality.shift(1, axis=1)


def get_sales_growth_index(
    revenue: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Sales Growth Index (SGI) measures the year-over-year growth in revenue. While
    growth is not inherently a sign of manipulation, high-growth companies face greater
    pressure to meet expectations and are more likely to engage in earnings manipulation
    to sustain the perception of growth.

    The formula is as follows:

        - SGI = Revenue (t) / Revenue (t-1)

    Args:
        revenue (pd.Series | pd.DataFrame): The revenue of the company.

    Returns:
        pd.Series | pd.DataFrame: The Sales Growth Index (SGI).

    Notes:
    - A high SGI is not proof of manipulation by itself but flags companies under pressure to
    keep growth rates up, a common precondition for earnings manipulation.
    """
    return revenue / revenue.shift(1, axis=1)


def get_depreciation_index(
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    property_plant_and_equipment: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Depreciation Index (DEPI) measures the change in the rate at which a company is
    depreciating its assets. A DEPI greater than 1 indicates that the depreciation rate has
    slowed down, which can be a sign that a company is revising the useful lives of its
    assets upwards to boost earnings.

    The formula is as follows:

        - Depreciation Rate = Depreciation and Amortization /
        (Depreciation and Amortization + Property, Plant and Equipment)
        - DEPI = Depreciation Rate (t-1) / Depreciation Rate (t)

    Args:
        depreciation_and_amortization (pd.Series | pd.DataFrame): The depreciation and
        amortization of the company.
        property_plant_and_equipment (pd.Series | pd.DataFrame): The property, plant and
        equipment of the company.

    Returns:
        pd.Series | pd.DataFrame: The Depreciation Index (DEPI).

    Notes:
    - A DEPI greater than 1 indicates a slowing depreciation rate, which can be a sign of
    earnings manipulation through the understatement of depreciation expense.
    """
    depreciation_rate = depreciation_and_amortization / (
        depreciation_and_amortization + property_plant_and_equipment
    )

    return depreciation_rate.shift(1, axis=1) / depreciation_rate


def get_selling_general_and_administrative_expenses_index(
    selling_general_and_administrative_expenses: pd.Series | pd.DataFrame,
    revenue: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Selling, General and Administrative Expenses Index (SGAI) measures the change in
    SG&A expenses relative to sales. A disproportionate increase in SG&A relative to sales
    can be viewed as a negative signal about a company's future prospects and, per the
    original Beneish research, is associated with a higher likelihood of manipulation.

    The formula is as follows:

        - SGA to Sales = Selling, General and Administrative Expenses / Revenue
        - SGAI = SGA to Sales (t) / SGA to Sales (t-1)

    Args:
        selling_general_and_administrative_expenses (pd.Series | pd.DataFrame): The selling,
        general and administrative expenses of the company.
        revenue (pd.Series | pd.DataFrame): The revenue of the company.

    Returns:
        pd.Series | pd.DataFrame: The Selling, General and Administrative Expenses Index (SGAI).

    Notes:
    - An SGAI greater than 1 indicates that SG&A expenses are growing faster than sales.
    """
    sga_to_sales = selling_general_and_administrative_expenses / revenue

    return sga_to_sales / sga_to_sales.shift(1, axis=1)


def get_leverage_index(
    total_current_liabilities: pd.Series | pd.DataFrame,
    long_term_debt: pd.Series | pd.DataFrame,
    total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Leverage Index (LVGI) measures the change in a company's leverage. An increase in
    leverage can create an incentive to manipulate earnings in order to remain within debt
    covenant thresholds.

    The formula is as follows:

        - Leverage = (Total Current Liabilities + Long Term Debt) / Total Assets
        - LVGI = Leverage (t) / Leverage (t-1)

    Args:
        total_current_liabilities (pd.Series | pd.DataFrame): The total current liabilities
        of the company.
        long_term_debt (pd.Series | pd.DataFrame): The long term debt of the company.
        total_assets (pd.Series | pd.DataFrame): The total assets of the company.

    Returns:
        pd.Series | pd.DataFrame: The Leverage Index (LVGI).

    Notes:
    - An LVGI greater than 1 indicates an increase in leverage relative to the prior period.
    """
    leverage = (total_current_liabilities + long_term_debt) / total_assets

    return leverage / leverage.shift(1, axis=1)


def get_total_accruals_to_total_assets(
    net_income: pd.Series | pd.DataFrame,
    cash_flow_from_operations: pd.Series | pd.DataFrame,
    total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Total Accruals to Total Assets (TATA) ratio measures the extent to which reported
    net income is backed by cash. A large positive gap between net income and operating
    cash flow (a high proportion of accruals) is one of the strongest individual predictors
    of earnings manipulation in the Beneish model.

    The formula is as follows:

        - TATA = (Net Income - Cash Flow from Operations) / Total Assets

    Args:
        net_income (pd.Series | pd.DataFrame): The net income of the company.
        cash_flow_from_operations (pd.Series | pd.DataFrame): The cash flow from operations
        of the company.
        total_assets (pd.Series | pd.DataFrame): The total assets of the company.

    Returns:
        pd.Series | pd.DataFrame: The Total Accruals to Total Assets (TATA) ratio.

    Notes:
    - A high TATA indicates that a large portion of earnings is composed of accruals rather
    than cash, which is a hallmark of earnings manipulation.
    """
    return (net_income - cash_flow_from_operations) / total_assets


def get_beneish_m_score(
    days_sales_in_receivables_index: pd.Series | pd.DataFrame,
    gross_margin_index: pd.Series | pd.DataFrame,
    asset_quality_index: pd.Series | pd.DataFrame,
    sales_growth_index: pd.Series | pd.DataFrame,
    depreciation_index: pd.Series | pd.DataFrame,
    selling_general_and_administrative_expenses_index: pd.Series | pd.DataFrame,
    leverage_index: pd.Series | pd.DataFrame,
    total_accruals_to_total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Beneish M-Score is a probabilistic model, developed by Messod Beneish, that uses eight
    financial ratios derived from a company's financial statements to identify whether a company
    has manipulated its earnings.

    The formula is as follows:

        M-Score = -4.84 + 0.92 * DSRI + 0.528 * GMI + 0.404 * AQI + 0.892 * SGI + 0.115 * DEPI
        - 0.172 * SGAI + 4.679 * TATA - 0.327 * LVGI

    The eight variables are:

        - DSRI: Days Sales in Receivables Index
        - GMI: Gross Margin Index
        - AQI: Asset Quality Index
        - SGI: Sales Growth Index
        - DEPI: Depreciation Index
        - SGAI: Selling, General and Administrative Expenses Index
        - TATA: Total Accruals to Total Assets
        - LVGI: Leverage Index

    Also known as: Beneish M-Score, earnings manipulation score.

    Args:
        days_sales_in_receivables_index (pd.Series | pd.DataFrame): The Days Sales in
        Receivables Index (DSRI).
        gross_margin_index (pd.Series | pd.DataFrame): The Gross Margin Index (GMI).
        asset_quality_index (pd.Series | pd.DataFrame): The Asset Quality Index (AQI).
        sales_growth_index (pd.Series | pd.DataFrame): The Sales Growth Index (SGI).
        depreciation_index (pd.Series | pd.DataFrame): The Depreciation Index (DEPI).
        selling_general_and_administrative_expenses_index (pd.Series | pd.DataFrame): The
        Selling, General and Administrative Expenses Index (SGAI).
        leverage_index (pd.Series | pd.DataFrame): The Leverage Index (LVGI).
        total_accruals_to_total_assets (pd.Series | pd.DataFrame): The Total Accruals to
        Total Assets (TATA) ratio.

    Returns:
        pd.Series | pd.DataFrame: The Beneish M-Score.

    Notes:
    - A M-Score greater than -1.78 suggests that the company is likely to be an earnings
    manipulator. A M-Score lower than -1.78 suggests the company is unlikely to be a manipulator.
    - As with the Altman Z-Score and Piotroski F-Score, this is a probabilistic, not a
    definitive, indicator and should be combined with further fundamental analysis.

    References:
    - Beneish, Messod D. "The Detection of Earnings Manipulation." Financial Analysts Journal,
    Vol. 55, No. 5, 1999, pp. 24-36.
    """
    return (
        -4.84
        + 0.92 * days_sales_in_receivables_index
        + 0.528 * gross_margin_index
        + 0.404 * asset_quality_index
        + 0.892 * sales_growth_index
        + 0.115 * depreciation_index
        - 0.172 * selling_general_and_administrative_expenses_index
        + 4.679 * total_accruals_to_total_assets
        - 0.327 * leverage_index
    )
