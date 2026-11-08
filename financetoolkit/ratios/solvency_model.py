"""Solvency Module"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import (
    calculate_change_over_absolute_base,
)


def get_debt_to_assets_ratio(
    total_debt: pd.Series | pd.DataFrame, total_assets: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the debt to assets ratio, a solvency ratio that measures the proportion of a
    company's assets that are financed by debt.

    This ratio is also known as the debt ratio.

    Args:
        total_debt (float or pd.Series): Total debt of the company.
        total_assets (float or pd.Series): Total assets of the company.

    Returns:
        float | pd.Series: The debt ratio value.
    """
    return total_debt / total_assets


def get_debt_to_equity_ratio(
    total_debt: pd.Series | pd.DataFrame, total_equity: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the debt to equity ratio, a solvency ratio that measures the
    proportion of a company's equity that is financed by debt.

    Args:
        total_debt (float or pd.Series): Total debt of the company.
        total_equity (float or pd.Series): Total equity of the company.

    Returns:
        float | pd.Series: The debt to equity ratio value.
    """
    return total_debt / total_equity


def get_interest_coverage_ratio(
    operating_income: pd.Series | pd.DataFrame,
    interest_expense: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the interest coverage ratio, a solvency ratio that measures a company's
    ability to pay its interest expenses on outstanding debt out of its operating profit.

    The formula is as follows (CFA Institute):

        Interest Coverage Ratio = EBIT / Interest Expense

    A zero interest expense gives no meaningful coverage (an infinite ratio), so those
    periods are returned as NaN.

    Also known as: times interest earned, TIE.

    Args:
        operating_income (float or pd.Series): Operating income (EBIT) of the company.
        interest_expense (float or pd.Series): Total interest expense of the company.

    Returns:
        float | pd.Series: The interest coverage ratio value, NaN where the interest expense
            is zero.
    """
    return operating_income / interest_expense.where(interest_expense != 0)


def get_debt_service_coverage_ratio(
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    interest_expense: pd.Series | pd.DataFrame,
    short_term_debt: pd.Series | pd.DataFrame,
    window_years: float = 1.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the debt service coverage ratio, a solvency ratio that measures a company's
    ability to service its debt (pay interest and repay the principal falling due) out
    of its operating earnings.

    Total debt service is the interest expense of the period plus the principal due
    within a year, which is the short-term debt (including the current portion of
    long-term debt) on the balance sheet. The numerator is EBITDA, operating income
    plus depreciation and amortization, as a proxy for the cash operating earnings
    available to service the debt.

    The short-term debt is the principal due over the next year, while the earnings and the
    interest cover the window of the figures: a quarter, a year or a trailing multiple of
    either. The principal is therefore scaled to the length of that window in years, so
    earnings, interest and principal all cover the same span: a quarter of earnings is
    compared with a quarter of the principal due within a year.

    The formula is as follows:

        Debt Service Coverage Ratio = (Operating Income + Depreciation and Amortization) /
            (Interest Expense + Short Term Debt x Window Years)

    A zero debt service gives no meaningful coverage, so those periods are returned as NaN.

    Args:
        operating_income (float or pd.Series): Operating income of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and amortization of
            the company.
        interest_expense (float or pd.Series): Interest expense of the company.
        short_term_debt (float or pd.Series): Short-term debt of the company, the principal
            due within a year.
        window_years (float, optional): The length in years of the window the operating
            income, depreciation and interest cover, e.g. 0.25 for a quarter and 1 for a
            trailing twelve months. Defaults to 1.0.

    Returns:
        float | pd.Series: The debt service coverage ratio value.
    """
    # The principal due within a year, spread evenly over the year, for the window's length.
    debt_service = interest_expense + short_term_debt * window_years

    return (operating_income + depreciation_and_amortization) / debt_service.where(
        debt_service != 0
    )


def get_equity_multiplier(
    average_total_assets: pd.Series | pd.DataFrame,
    average_total_equity: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the equity multiplier, a solvency ratio that measures the degree to which a company
    uses borrowed money (debt) to finance its operations and growth.

    This is also referred to as the company financial leverage.

    Args:
        average_total_assets (float or pd.Series): Average total assets of the company.
            This is typically calculated as (beginning total assets + ending total assets) / 2.
        average_total_equity (float or pd.Series): Average total equity of the company.
            This is typically calculated as (beginning total equity + ending total equity) / 2.

    Returns:
        float | pd.Series: The equity multiplier.
    """
    return average_total_assets / average_total_equity


def get_free_cash_flow_yield(
    free_cash_flow: pd.Series | pd.DataFrame,
    market_capitalization: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the free cash flow yield ratio, which measures the free cash flow
    relative to the market capitalization of the company.

    Args:
        free_cash_flow (float or pd.Series): Free cash flow of the company.
        market_capitalization (float or pd.Series): Market capitalization of the company.

    Returns:
        float | pd.Series: The free cash flow yield ratio.
    """
    return free_cash_flow / market_capitalization


def get_net_debt_to_ebitda_ratio(
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    net_debt: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the net debt to EBITDA ratio, which measures the net debt of the company
    relative to its EBITDA.

    Args:
        operating_income (float or pd.Series): Operating income of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and amortization of the company.
        net_debt (float or pd.Series): Net debt of the company.

    Returns:
        float | pd.Series: The net debt to EBITDA ratio.
    """
    return net_debt / (operating_income + depreciation_and_amortization)


def get_gross_debt_to_ebitda_ratio(
    total_debt: pd.Series | pd.DataFrame,
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the gross debt to EBITDA ratio, which measures the total (gross) debt of
    the company relative to its EBITDA.

    This differs from `get_net_debt_to_ebitda_ratio` in that it uses total (gross) debt
    rather than net debt (total debt minus cash and cash equivalents). Gross debt to
    EBITDA is a more conservative leverage measure since it does not assume that a
    company's cash balance would actually be used to pay down debt, which matters when
    comparing companies with restricted cash, cash earmarked for other purposes, or when
    assessing gross refinancing risk rather than net economic leverage.

    The formula is as follows:

        Gross Debt to EBITDA Ratio = Total Debt / (Operating Income + Depreciation and Amortization)

    Args:
        total_debt (float or pd.Series): Total debt of the company.
        operating_income (float or pd.Series): Operating income of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and amortization of the company.

    Returns:
        float | pd.Series: The gross debt to EBITDA ratio.
    """
    return total_debt / (operating_income + depreciation_and_amortization)


def get_asset_coverage_ratio(
    total_assets: pd.Series | pd.DataFrame,
    intangible_assets: pd.Series | pd.DataFrame,
    current_liabilities: pd.Series | pd.DataFrame,
    short_term_debt: pd.Series | pd.DataFrame,
    total_debt: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the asset coverage ratio, a solvency ratio that measures how well a
    company's tangible assets, after settling non-debt current liabilities, can cover
    its total debt.

    This ratio is commonly used by lenders and bondholders to assess the extent to
    which a company's hard (tangible) assets would be available to repay debt
    obligations in a liquidation scenario, since intangible assets (e.g. goodwill)
    typically have little to no recovery value and non-debt current liabilities are
    assumed to be settled first out of current assets. Short-term debt is netted out
    of current liabilities before subtracting, because it is already captured in
    total debt — otherwise it would be double-counted, once as part of current
    liabilities and again as part of total debt.

    The formula is as follows:

        Asset Coverage Ratio = [(Total Assets - Intangible Assets)
            - (Current Liabilities - Short-Term Debt)] / Total Debt

    Args:
        total_assets (float or pd.Series): Total assets of the company.
        intangible_assets (float or pd.Series): Intangible assets of the company.
        current_liabilities (float or pd.Series): Total current liabilities of the company.
        short_term_debt (float or pd.Series): Short-term (current portion of) debt of the company.
        total_debt (float or pd.Series): Total debt of the company.

    Returns:
        float | pd.Series: The asset coverage ratio.
    """
    return (
        (total_assets - intangible_assets) - (current_liabilities - short_term_debt)
    ) / total_debt


def get_cash_flow_coverage_ratio(
    operating_cash_flow: pd.Series | pd.DataFrame, total_debt: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the cash flow coverage ratio, a solvency ratio that measures a company's ability to pay off its debt
    with its operating cash flow.

    Args:
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.
        total_debt (float or pd.Series): Total debt of the company.

    Returns:
        float | pd.Series: The cash flow coverage ratio value.
    """
    return operating_cash_flow / total_debt


def get_capex_coverage_ratio(
    cash_flow_from_operations: pd.Series | pd.DataFrame,
    capital_expenditure: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the capital expenditure coverage ratio, a solvency ratio that
    measures a company's ability to cover its capital expenditures with its
    cash flow from operations.

    Args:
        cash_flow_from_operations (float or pd.Series): Cash flow from operations of the company.
        capital_expenditure (float or pd.Series): Capital expenditure of the company,
            as reported in the Cash Flow Statement.

    Returns:
        float | pd.Series: The capital expenditure coverage ratio value.
    """
    # Capital Expenditure is reported as a negative cash outflow, so the magnitude is used to keep the ratio positive and consistent with "higher is better".  # noqa: E501
    return cash_flow_from_operations / abs(capital_expenditure)


def get_dividend_capex_coverage_ratio(
    cash_flow_from_operations: pd.Series | pd.DataFrame,
    capital_expenditure: pd.Series | pd.DataFrame,
    dividends: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the dividend paid and capex coverage ratio, a solvency ratio that
    measures a company's ability to cover both its capital expenditures and
    dividend payments with its cash flow from operations.

    Args:
        cash_flow_from_operations (float or pd.Series): Cash flow from operations of the company.
        capital_expenditure (float or pd.Series): Capital expenditure of the company,
            as reported in the Cash Flow Statement.
        dividends (float or pd.Series): Dividend payments of the company, as reported
            in the Cash Flow Statement.

    Returns:
        float | pd.Series: The dividend paid and capex coverage ratio value.
    """
    # Both are reported as negative cash outflows, so their magnitudes are used to keep the ratio positive and consistent with "higher is better".  # noqa: E501
    return cash_flow_from_operations / (abs(capital_expenditure) + abs(dividends))


def get_debt_to_capital_ratio(
    total_debt: pd.Series | pd.DataFrame, total_equity: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the debt to capital ratio, a solvency ratio that measures the proportion
    of a company's total capital (debt plus equity) that is financed by debt.

    Unlike the debt to equity ratio, which can theoretically exceed one or become
    negative with low or negative equity, the debt to capital ratio is bounded
    between 0 and 1 under normal circumstances, making it easier to compare across
    companies with very different capital structures.

    Args:
        total_debt (float or pd.Series): Total debt of the company.
        total_equity (float or pd.Series): Total equity of the company.

    Returns:
        float | pd.Series: The debt to capital ratio value.
    """
    return total_debt / (total_debt + total_equity)


def get_preferred_dividend_coverage_ratio(
    net_income: pd.Series | pd.DataFrame, preferred_dividends: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the preferred dividend coverage ratio, a solvency ratio that measures a
    company's ability to pay dividends owed to preferred shareholders out of its net
    income.

    Args:
        net_income (float or pd.Series): Net income of the company.
        preferred_dividends (float or pd.Series): Preferred dividends paid by the company,
            as reported in the Cash Flow Statement.

    Returns:
        float | pd.Series: The preferred dividend coverage ratio value, NaN where no
            preferred dividends were paid (there is nothing to cover).
    """
    preferred_dividends = abs(preferred_dividends)

    return net_income / preferred_dividends.where(preferred_dividends != 0)


def get_interest_paid_to_expense_ratio(
    interest_paid: pd.Series | pd.DataFrame, interest_expense: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the interest paid to interest expense ratio, which measures how much of
    the accrual-based interest expense reported on the income statement was actually
    paid out in cash during the period.

    A ratio consistently below one can indicate that interest is being accrued
    (e.g. on payment-in-kind debt) rather than paid, while a ratio well above one can
    indicate the payment of previously accrued interest or a mismatch between the cash
    and accrual reporting periods, both of which are relevant quality-of-earnings signals.

    Args:
        interest_paid (float or pd.Series): Interest paid by the company, as reported
            in the Cash Flow Statement.
        interest_expense (float or pd.Series): Interest expense of the company, as reported
            in the Income Statement.

    Returns:
        float | pd.Series: The interest paid to interest expense ratio value.
    """
    return interest_paid / interest_expense


def get_ebitda_interest_coverage_ratio(
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    interest_expense: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the EBITDA interest coverage ratio, the number of times the earnings
    before interest, taxes, depreciation and amortization cover the interest expense.

    The formula is as follows (S&P Global Ratings, Moody's):

        EBITDA Interest Coverage = (Operating Income + Depreciation and Amortization)
            / Interest Expense

    The interest expense is the gross expense, without netting the interest income, as
    S&P Global Ratings does. A zero interest expense gives no meaningful coverage, so
    those periods are NaN.

    Also known as: EBITDA to interest, EBITDA coverage, EBITDA/interest.

    Args:
        operating_income (float or pd.Series): Operating income (EBIT) of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and
            amortization of the company.
        interest_expense (float or pd.Series): Interest expense of the company.

    Returns:
        float | pd.Series: The EBITDA interest coverage value, NaN where the interest
            expense is zero.
    """
    ebitda = operating_income + depreciation_and_amortization

    return ebitda / interest_expense.where(interest_expense != 0)


def get_ebitda_minus_capex_interest_coverage_ratio(
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    capital_expenditure: pd.Series | pd.DataFrame,
    interest_expense: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the EBITDA minus capital expenditure interest coverage ratio, the number
    of times the EBITDA that is left after the capital expenditure covers the interest
    expense.

    The formula is as follows (Moody's):

        (EBITDA - CapEx) / Interest = (Operating Income + Depreciation and Amortization
            - |Capital Expenditure|) / Interest Expense

    Capital expenditure is reported as a negative cash outflow, so its magnitude is
    deducted. A zero interest expense gives NaN.

    Also known as: (EBITDA - CapEx) / interest expense, EBITDA less capex coverage.

    Args:
        operating_income (float or pd.Series): Operating income (EBIT) of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and
            amortization of the company.
        capital_expenditure (float or pd.Series): Capital expenditure of the company, as
            reported in the Cash Flow Statement.
        interest_expense (float or pd.Series): Interest expense of the company.

    Returns:
        float | pd.Series: The (EBITDA - CapEx) interest coverage value, NaN where the
            interest expense is zero.
    """
    ebitda_minus_capex = (
        operating_income + depreciation_and_amortization - abs(capital_expenditure)
    )

    return ebitda_minus_capex / interest_expense.where(interest_expense != 0)


def get_cash_interest_coverage_ratio(
    cash_flow_from_operations: pd.Series | pd.DataFrame,
    interest_paid: pd.Series | pd.DataFrame,
    income_taxes_paid: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the cash interest coverage ratio, the number of times the operating cash
    flow before interest and taxes covers the interest paid in cash.

    The formula is as follows (CFA Institute):

        Cash Interest Coverage = (Cash Flow from Operations + Interest Paid
            + Income Taxes Paid) / Interest Paid

    The interest and taxes paid are added back because the cash flow from operations
    is after both (US GAAP). A zero interest paid gives NaN.

    Also known as: cash flow interest coverage, CFO interest coverage.

    Args:
        cash_flow_from_operations (float or pd.Series): Cash flow from operations.
        interest_paid (float or pd.Series): Interest paid in cash, as reported in the
            Cash Flow Statement.
        income_taxes_paid (float or pd.Series): Income taxes paid in cash, as reported in
            the Cash Flow Statement.

    Returns:
        float | pd.Series: The cash interest coverage value, NaN where no interest is paid.
    """
    return (cash_flow_from_operations + interest_paid + income_taxes_paid) / (
        interest_paid.where(interest_paid != 0)
    )


def get_funds_from_operations_to_debt_ratio(
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    stock_based_compensation: pd.Series | pd.DataFrame,
    interest_paid: pd.Series | pd.DataFrame,
    income_taxes_paid: pd.Series | pd.DataFrame,
    total_debt: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the funds from operations (FFO) to debt ratio, one of the two core
    ratios of S&P Global Ratings: the cash a company generates from its operations,
    before working capital and investment, as a share of its debt.

    The formula is as follows (S&P Global Ratings):

        FFO = EBITDA - Cash Interest Paid - Cash Tax Paid

        EBITDA = Operating Income + Depreciation and Amortization + Stock-Based
            Compensation

        FFO to Debt = FFO / Total Debt

    S&P excludes share-based compensation expense payable in shares from its EBITDA,
    so it is added back. A company with debt pays interest, so an interest paid of zero
    is taken as not reported and gives NaN (counting it as zero would overstate the
    FFO by the whole interest bill), as does a total debt of zero or less.

    Also known as: FFO/debt, funds from operations to debt.

    Args:
        operating_income (float or pd.Series): Operating income (EBIT) of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and
            amortization of the company.
        stock_based_compensation (float or pd.Series): Stock-based compensation of the
            company.
        interest_paid (float or pd.Series): Interest paid in cash.
        income_taxes_paid (float or pd.Series): Income taxes paid in cash.
        total_debt (float or pd.Series): Total debt at the end of the period.

    Returns:
        float | pd.Series: The FFO to debt value, NaN where there is no debt.
    """
    funds_from_operations = (
        operating_income
        + depreciation_and_amortization
        + stock_based_compensation
        - interest_paid.where(interest_paid != 0)
        - income_taxes_paid
    )

    return funds_from_operations / total_debt.where(total_debt > 0)


def get_discretionary_cash_flow_to_debt_ratio(
    cash_flow_from_operations: pd.Series | pd.DataFrame,
    capital_expenditure: pd.Series | pd.DataFrame,
    dividends_paid: pd.Series | pd.DataFrame,
    common_stock_purchased: pd.Series | pd.DataFrame,
    total_debt: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the discretionary cash flow (DCF) to debt ratio, a supplementary ratio of
    S&P Global Ratings: the cash flow that is left after capital expenditure, dividends
    and share buybacks, and so could repay debt, as a share of the debt.

    The formula is as follows (S&P Global Ratings):

        DCF = Cash Flow from Operations - |Capital Expenditure| - |Dividends Paid|
            - |Share Buybacks|

        DCF to Debt = DCF / Total Debt

    Capital expenditure, dividends and buybacks are reported as negative cash outflows,
    so their magnitudes are deducted. A total debt of zero or less gives NaN.

    Also known as: DCF/debt, discretionary cash flow to debt.

    Args:
        cash_flow_from_operations (float or pd.Series): Cash flow from operations.
        capital_expenditure (float or pd.Series): Capital expenditure of the company.
        dividends_paid (float or pd.Series): Dividends paid on common and preferred stock.
        common_stock_purchased (float or pd.Series): Share buybacks (common stock
            purchased).
        total_debt (float or pd.Series): Total debt at the end of the period.

    Returns:
        float | pd.Series: The DCF to debt value, NaN where there is no debt.
    """
    discretionary_cash_flow = (
        cash_flow_from_operations
        - abs(capital_expenditure)
        - abs(dividends_paid)
        - abs(common_stock_purchased)
    )

    return discretionary_cash_flow / total_debt.where(total_debt > 0)


def get_degree_of_leverage(
    result: pd.Series | pd.DataFrame,
    driver: pd.Series | pd.DataFrame,
    lag: int = 1,
) -> pd.Series | pd.DataFrame:
    """
    Calculate a degree of leverage: the percentage change of a result divided by the
    percentage change of the item that drives it, over the same periods (CFA Institute).

    The formula is as follows:

        Degree of Leverage = %Δ Result / %Δ Driver

    with the operating income as result and the revenue as driver for the degree of
    operating leverage (DOL), the net income and operating income for the degree of
    financial leverage (DFL) and the net income and revenue for the degree of total
    leverage (DTL = DOL x DFL).

    Each percentage change is (Value - Previous Value) / |Previous Value|, so a change
    from a negative base keeps its sign. A driver that does not change, or a change
    from a base of zero, gives NaN.

    Args:
        result (pd.Series | pd.DataFrame): The result, e.g. the operating income. A
            DataFrame has the periods as columns.
        driver (pd.Series | pd.DataFrame): The driver, e.g. the revenue, with the same
            shape as the result.
        lag (int, optional): The number of periods between the compared values, e.g. 4
            to compare a quarter with the same quarter a year earlier. Defaults to 1.

    Returns:
        pd.Series | pd.DataFrame: The degree of leverage.
    """
    change_in_result = calculate_change_over_absolute_base(result, lag, "columns")
    change_in_driver = calculate_change_over_absolute_base(driver, lag, "columns")

    degree_of_leverage = change_in_result / change_in_driver.where(
        change_in_driver != 0
    )

    return degree_of_leverage.replace([np.inf, -np.inf], np.nan)
