"""Profitability Module"""

__docformat__ = "google"

import pandas as pd


def get_gross_margin(revenue: pd.Series, cost_of_goods_sold: pd.Series) -> pd.Series:
    """
    Calculate the gross margin, a profitability ratio that measures the percentage of
    revenue that exceeds the cost of goods sold.

    Args:
        revenue (float or pd.Series): Total revenue of the company.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.

    Returns:
        float | pd.Series: The gross margin percentage value.
    """
    return (revenue - cost_of_goods_sold) / revenue


def get_operating_margin(operating_income: pd.Series, revenue: pd.Series) -> pd.Series:
    """
    Calculate the operating margin, a profitability ratio that measures the percentage of
    revenue that remains after deducting operating expenses.

    Args:
        operating_income (float or pd.Series): Operating income of the company.
        revenue (float or pd.Series): Total revenue of the company.

    Returns:
        float | pd.Series: The operating margin percentage value.
    """
    return operating_income / revenue


def get_net_profit_margin(net_income: pd.Series, revenue: pd.Series) -> pd.Series:
    """
    Calculate the net profit margin, a profitability ratio that measures the percentage
    of profit a company earns per dollar of revenue.

    Args:
        net_income (float or pd.Series): Net income of the company.
        revenue (float or pd.Series): Revenue of the company.

    Returns:
        float | pd.Series: The net profit margin value as a percentage.
    """
    return net_income / revenue


def get_interest_coverage_ratio(
    operating_income: pd.Series, interest_expense: pd.Series
) -> pd.Series:
    """
    Compute the Interest Coverage Ratio, a metric that reveals a company's
    ability to cover its interest expenses with its pre-tax profits.
    This ratio measures how many times the operating income covers the
    interest payments of operain required and is crucial in determining a
    company's financial health.

    Args:
        operating_income (float or pd.Series): Operating income of the company.
        interest_expense (float or pd.Series): Interest expense of the company.

    Returns:
        float | pd.Series: The Interest Coverage Ratio
    """
    return operating_income / interest_expense


def get_interest_burden_ratio(
    income_before_tax: pd.Series, operating_income: pd.Series
) -> pd.Series:
    """
    Compute the Interest Burden Ratio, a metric that reveals a company's
    ability to cover its interest expenses with its pre-tax profits.
    This ratio measures how many times the operating income covers the
    interest payments of operain required and is crucial in determining a
    company's financial health.

    Args:
        income_before_tax (float or pd.Series): Income before tax of the company.
        operating_income (float or pd.Series): Operating income of the company.

    Returns:
        float | pd.Series: The Interest Burden Ratio
    """
    return income_before_tax / operating_income


def get_income_before_tax_profit_margin(
    income_before_tax: pd.Series, revenue: pd.Series
) -> pd.Series:
    """
    Calculate the Pretax Profit Margin, which is the ratio of a company's pre-tax profit to its revenue,
    indicating how much profit a company makes before paying taxes on its earnings.

    Args:
        income_before_tax (float or pd.Series): Income before tax of the company.
        revenue (float or pd.Series): Revenue of the company.

    Returns:
        float | pd.Series: The Pretax Profit Margin value.
    """
    return income_before_tax / revenue


def get_effective_tax_rate(
    income_tax_expense: pd.Series, income_before_tax: pd.Series
) -> pd.Series:
    """
    Calculate the effective tax rate, a financial ratio that measures the percentage of pretax income
    that is paid as taxes.

    Args:
        income_tax_expense (float or pd.Series): The amount of income tax paid by the company.
        income_before_tax (float or pd.Series): The company's income before taxes.

    Returns:
        float | pd.Series: The effective tax rate value.
    """
    return income_tax_expense / income_before_tax


def get_return_on_assets(
    net_income: pd.Series, average_total_assets: pd.Series
) -> pd.Series:
    """
    Calculate the return on assets (ROA), a profitability ratio that measures how
    efficiently a company uses its assets to generate profits.

    Args:
        net_income (float or pd.Series): Net income of the company.
        total_assets_begin (float or pd.Series): Total equity at the beginning of the period.
        total_assets_end (float or pd.Series): Total equity at the end of the period.

    Returns:
        float | pd.Series: The ROA percentage value.
    """
    return net_income / average_total_assets


def get_return_on_equity(
    net_income: pd.Series,
    average_total_equity: pd.Series,
) -> pd.Series:
    """
    Calculate the return on equity (ROE), a profitability ratio that measures how
    efficiently a company generates profits using its shareholders' equity.

    Args:
        net_income (float or pd.Series): Net income of the company.
        total_equity_begin (float or pd.Series): Total equity at the beginning of the period.
        total_equity_end (float or pd.Series): Total equity at the end of the period.

    Returns:
        float | pd.Series: The ROE percentage value.
    """
    return net_income / average_total_equity


def get_return_on_invested_capital(
    net_income: pd.Series,
    dividends: pd.Series,
    average_total_equity: pd.Series,
    average_total_debt: pd.Series,
) -> pd.Series:
    """
    Calculate the return on invested capital, a financial ratio that measures the company's return on
    the capital invested in it, including both equity and debt.

    Args:
        net_income (float or pd.Series): The company's net income.
        dividends (float or pd.Series): The dividends paid by the company.
        effective_tax_rate (float or pd.Series): The effective tax rate of the company.
        total_equity_begin (float or pd.Series): The total equity at the beginning of the period.
        totaL_equity_end (float or pd.Series): The total equity at the end of the period.
        total_debt_begin (float or pd.Series): The total debt at the beginning of the period.
        total_debt_end (float or pd.Series): The total debt at the end of the period.

    Returns:
        float | pd.Series: The return on invested capital value.
    """
    return (net_income - dividends) / (average_total_equity + average_total_debt)


def get_income_quality_ratio(
    cash_flow_from_operating_activities: pd.Series,
    net_income: pd.Series,
) -> pd.Series:
    """
    Calculates the income quality ratio, which measures the cash flow from operating
    activities relative to the net income of the company.

    Args:
        cash_flow_from_operating_activities (float or pd.Series): Cash flow from operating activities
            of the company.
        net_income (float or pd.Series): Net income of the company.

    Returns:
        float | pd.Series: The income quality ratio.
    """
    return cash_flow_from_operating_activities / net_income


def get_return_on_tangible_assets(
    net_income: pd.Series,
    average_total_assets: pd.Series,
    average_intangible_assets: pd.Series,
    average_total_liabilities: pd.Series,
) -> pd.Series:
    """
    Calculate the return on tangible assets, which measures the amount of profit
    generated by a company's tangible assets.

    Args:
        net_income (float or pd.Series): The net income of the company.
        interest_expense (float or pd.Series): The interest expense of the company.
        total_assets (float or pd.Series): The total assets of the company.
        total_liabilities (float or pd.Series): The total liabilities of the company.

    Returns:
        float | pd.Series: The return on tangible assets value.
    """
    average_tangible_assets = (
        average_total_assets + average_intangible_assets + average_total_liabilities
    )

    return net_income / average_tangible_assets


def get_return_on_capital_employed(
    net_income: pd.Series,
    interest_expense: pd.Series,
    tax_expense: pd.Series,
    total_assets: pd.Series,
    total_current_liabilities: pd.Series,
) -> pd.Series:
    """
    Calculate the return on capital employed (ROCE), a profitability ratio that measures
    the amount of return a company generates from the capital it has invested in the business.

    Args:
        net_income (float or pd.Series): Net income of the company.
        interest_expense (float or pd.Series): Interest expense of the company.
        tax_expense (float or pd.Series): Tax expense of the company.
        total_assets (float or pd.Series): Total assets of the company.
        total_current_liabilities (float or pd.Series): Total current liabilities of the company.

    Returns:
        float | pd.Series: The ROCE value.
    """
    return (net_income + interest_expense + tax_expense) / (
        total_assets - total_current_liabilities
    )


def get_net_income_per_ebt(
    net_income: pd.Series, income_tax_expense: pd.Series
) -> pd.Series:
    """
    Calculate the net income per earnings before taxes (EBT), a profitability ratio that
    measures the net income generated for each dollar of EBT.

    Args:
        net_income (float or pd.Series): Net income of the company.
        income_tax_expense (float or pd.Series): Income tax expense of the company.

    Returns:
        float | pd.Series: The net income per EBT value.
    """
    return net_income / (net_income + income_tax_expense)


def get_free_cash_flow_operating_cash_flow_ratio(
    free_cash_flow: pd.Series, operating_cash_flow: pd.Series
) -> pd.Series:
    """
    Calculate the free cash flow to operating cash flow ratio, a profitability
    ratio that measures the amount of free cash flow a company generates
    for every dollar of operating cash flow.

    Args:
        free_cash_flow (float or pd.Series): Free cash flow of the company.
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.

    Returns:
        float | pd.Series: The free cash flow to operating cash flow ratio value.
    """
    return free_cash_flow / operating_cash_flow


def get_tax_burden_ratio(
    net_income: pd.Series, income_before_tax: pd.Series
) -> pd.Series:
    """
    Calculate the tax burden ratio, which is the ratio of a company's
    net income to its income before tax, indicating how much of a
    company's income is retained after taxes.

    Args:
        net_income (float or pd.Series): Net income of the company.
        income_before_tax (float or pd.Series): Income before tax of the company.

    Returns:
        float | pd.Series: The NIperEBT value.
    """
    return net_income / income_before_tax


def get_EBT_to_EBIT(
    earnings_before_tax: pd.Series,
    earnings_before_interest_and_taxes: pd.Series,
) -> pd.Series:
    """
    Calculate the EBT to EBIT, which is the ratio of a company's earnings before tax to its earnings before
    interest and taxes, indicating how much of a company's earnings are generated before paying interest on debt.

    Args:
        earnings_before_tax (float or pd.Series): Earnings before tax of the company.
        earnings_before_interest_and_taxes (float or pd.Series): Earnings before interest and taxes of the company.

    Returns:
        float | pd.Series: The EBTperEBIT value.
    """
    return earnings_before_tax / earnings_before_interest_and_taxes


def get_EBIT_to_revenue(
    earnings_before_interest_and_taxes: pd.Series, revenue: pd.Series
) -> pd.Series:
    """
    Calculate the EBITperRevenue, which is the ratio of a company's earnings
    before interest and taxes to its revenue, indicating how much profit a
    company generates from its operations before paying interest on debt
    and taxes on its earnings.

    Args:
        earnings_before_interest_and_taxes (float or pd.Series): Earnings before interest and taxes of the company.
        revenue (float or pd.Series): Revenue of the company.

    Returns:
        float | pd.Series: The EBITperRevenue value.
    """
    return earnings_before_interest_and_taxes / revenue
