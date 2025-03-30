"""Solvency Module"""

__docformat__ = "google"

import pandas as pd


def get_debt_to_assets_ratio(
    total_debt: float | pd.Series | pd.Series, total_assets: float | pd.Series
) -> pd.Series:
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
    total_debt: float | pd.Series, total_equity: float | pd.Series
) -> pd.Series:
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
    operating_income: float | pd.Series,
    depreciation_and_amortization: float | pd.Series,
    interest_expense: float | pd.Series,
) -> pd.Series:
    """
    Calculate the interest coverage ratio, a solvency ratio that measures a company's
    ability to pay its interest expenses on outstanding debt.

    Args:
        operating_income (float or pd.Series): Operating income of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and amortization of the company.
        interest_expense (float or pd.Series): Total interest expense of the company.

    Returns:
        float | pd.Series: The interest coverage ratio value.
    """
    return (operating_income + depreciation_and_amortization) / interest_expense


def get_debt_service_coverage_ratio(
    operating_income: float | pd.Series, current_liabilities: float | pd.Series
) -> pd.Series:
    """
    Calculate the debt service coverage ratio, a solvency ratio that measures a company's
    ability to service its debt with its net operating income.

    Args:
        net_operating_income (float or pd.Series): Net operating income of the company.
        current_liabilities (float or pd.Series): Total debt service of the company.

    Returns:
        float | pd.Series: The debt service coverage ratio value.
    """
    return operating_income / current_liabilities


def get_equity_multiplier(
    average_total_assets: float | pd.Series,
    average_total_equity: float | pd.Series,
) -> pd.Series:
    """
    Calculate the equity multiplier, a solvency ratio that measures the degree to which a company
    uses borrowed money (debt) to finance its operations and growth.

    This is also referred to as the company financial leverage.

    Args:
        total_assets_begin (float or pd.Series): Total assets at the beginning of the period.
        total_assets_end (float or pd.Series): Total assets at the end of the period.
        total_equity_begin (float or pd.Series): Total equity at the beginning of the period.
        total_equity_end (float or pd.Series): Total equity at the end of the period.

    Returns:
        float | pd.Series: The equity multiplier.
    """
    return average_total_assets / average_total_equity


def get_free_cash_flow_yield(
    free_cash_flow: float | pd.Series, market_capitalization: float | pd.Series
) -> pd.Series:
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
    operating_income: float | pd.Series,
    depreciation_and_amortization: float | pd.Series,
    net_debt: float | pd.Series,
) -> pd.Series:
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


def get_cash_flow_coverage_ratio(
    operating_cash_flow: float | pd.Series,
    total_debt: float | pd.Series,
) -> pd.Series:
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
    cash_flow_from_operations: float | pd.Series, capital_expenditure: float | pd.Series
) -> pd.Series:
    """
    Calculate the capital expenditure coverage ratio, a solvency ratio that
    measures a company's ability to cover its capital expenditures with its
    cash flow from operations.

    Args:
        cash_flow_from_operations (float or pd.Series): Cash flow from operations of the company.
        capital_expenditure (float or pd.Series): Capital expenditure of the company.

    Returns:
        float | pd.Series: The capital expenditure coverage ratio value.
    """
    return cash_flow_from_operations / capital_expenditure


def get_dividend_capex_coverage_ratio(
    cash_flow_from_operations: float | pd.Series,
    capital_expenditure: float | pd.Series,
    dividends: float | pd.Series,
) -> pd.Series:
    """
    Calculate the dividend paid and capex coverage ratio, a solvency ratio that
    measures a company's ability to cover both its capital expenditures and
    dividend payments with its cash flow from operations.

    Args:
        cash_flow_from_operations (float or pd.Series): Cash flow from operations of the company.
        capital_expenditure (float or pd.Series): Capital expenditure of the company.
        dividends (float or pd.Series): Dividend payments of the company.

    Returns:
        float | pd.Series: The dividend paid and capex coverage ratio value.
    """
    return cash_flow_from_operations / (capital_expenditure + dividends)
