"""Liquidity Module"""

__docformat__ = "google"


from typing import TypeVar

import pandas as pd

# The calculations return whichever type they are given: a Series for one company,
# a DataFrame for several. Mixed or scalar inputs keep their own @overload stubs.
SeriesOrFrame = TypeVar("SeriesOrFrame", pd.Series, pd.DataFrame)


def get_current_ratio(
    current_assets: SeriesOrFrame, current_liabilities: SeriesOrFrame
) -> SeriesOrFrame:
    """
    Calculate the current ratio, a liquidity ratio that measures a company's ability
    to pay off its short-term liabilities with its current assets.

    This can also be called the working capital ratio.

    Args:
        current_assets (float or pd.Series): Total current assets of the company.
        current_liabilities (float or pd.Series): Total current liabilities of the company.

    Returns:
        float | pd.Series: The current ratio value.
    """
    return current_assets / current_liabilities


def get_quick_ratio(
    cash_and_equivalents: SeriesOrFrame,
    marketable_securities: SeriesOrFrame,
    accounts_receivable: SeriesOrFrame,
    current_liabilities: SeriesOrFrame,
) -> SeriesOrFrame:
    """
    Calculate the quick ratio (also known as the acid-test ratio), a more stringent
    measure of liquidity that excludes inventory from current assets.

    This uses the narrow ("quick assets") formulation, which builds the numerator up
    from the three assets that can be converted to cash quickly, rather than the
    broader (Current Assets - Inventory) / Current Liabilities formulation. The narrow
    version is the stricter of the two since it also excludes prepaid expenses and
    other current assets that cannot readily be turned into cash.

    The formula is as follows:

        Quick Ratio = (Cash and Cash Equivalents + Marketable Securities +
            Accounts Receivable) / Current Liabilities

    Also known as: acid-test ratio.

    Args:
        cash_and_equivalents (float or pd.Series): Total cash and cash equivalents of the company.
        marketable_securities (float or pd.Series): Total marketable securities of the company.
        accounts_receivable (float or pd.Series): Total accounts receivable of the company.
        current_liabilities (float or pd.Series): Total current liabilities of the company.

    Returns:
        float | pd.Series: The quick ratio value.
    """
    return (
        cash_and_equivalents + marketable_securities + accounts_receivable
    ) / current_liabilities


def get_cash_ratio(
    cash_and_equivalents: SeriesOrFrame,
    marketable_securities: SeriesOrFrame,
    current_liabilities: SeriesOrFrame,
) -> SeriesOrFrame:
    """
    Calculate the cash ratio, a liquidity ratio that measures a company's ability
    to pay off its short-term liabilities with its cash and cash equivalents.

    Args:
        cash_and_equivalents (float or pd.Series): Total cash and cash equivalents of the company.
        marketable_securities (float or pd.Series): Total marketable securities of the company.
        current_liabilities (float or pd.Series): Total current liabilities of the company.

    Returns:
        float | pd.Series: The cash ratio value.
    """
    return (cash_and_equivalents + marketable_securities) / current_liabilities


def get_defensive_interval_ratio(
    cash_and_equivalents: SeriesOrFrame,
    marketable_securities: SeriesOrFrame,
    accounts_receivable: SeriesOrFrame,
    daily_operating_expenses: SeriesOrFrame,
) -> SeriesOrFrame:
    """
    Calculate the defensive interval ratio (DIR), a liquidity ratio that measures how
    many days a company could continue to cover its operating expenses using only its
    existing defensive (most liquid) assets, without relying on additional revenue.

    Unlike the current, quick, and cash ratios, which express liquidity relative to
    current liabilities, the defensive interval ratio expresses liquidity relative to
    the company's actual daily cash burn rate. This makes it a more direct measure of
    how long a company could survive a sudden stop in incoming cash flow, which is
    especially relevant for early-stage or cyclical companies.

    The formula is as follows:

        Defensive Interval Ratio = (Cash and Cash Equivalents + Marketable Securities +
            Accounts Receivable) / Daily Operating Expenses

    Where Daily Operating Expenses is typically calculated as
    (Operating Expenses - Non-Cash Charges) / 365, i.e. the average cash operating
    expenses incurred per day.

    Also known as: defensive interval period, basic defense interval.

    Args:
        cash_and_equivalents (float or pd.Series): Total cash and cash equivalents of the company.
        marketable_securities (float or pd.Series): Total marketable securities of the company.
        accounts_receivable (float or pd.Series): Total accounts receivable of the company.
        daily_operating_expenses (float or pd.Series): The average cash operating expenses
            incurred per day, typically (Operating Expenses - Non-Cash Charges) / 365.

    Returns:
        float | pd.Series: The defensive interval ratio value, expressed in days.
    """
    return (
        cash_and_equivalents + marketable_securities + accounts_receivable
    ) / daily_operating_expenses


def get_working_capital(
    current_assets: SeriesOrFrame, current_liabilities: SeriesOrFrame
) -> SeriesOrFrame:
    """
    Calculate the working capital, which is the difference between a company's current assets
    and current liabilities.

    Args:
        current_assets (float or pd.Series): The current assets of the company.
        current_liabilities (float or pd.Series): The current liabilities of the company.

    Returns:
        float | pd.Series: The working capital value.
    """
    return current_assets - current_liabilities


def get_operating_cash_flow_ratio(
    operating_cash_flow: SeriesOrFrame, current_liabilities: SeriesOrFrame
) -> SeriesOrFrame:
    """
    Calculate the operating cash flow ratio, a liquidity ratio that measures a company's
    ability to pay off its current liabilities with its operating cash flow.

    Args:
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.
        current_liabilities (float or pd.Series): Current liabilities of the company.

    Returns:
        float | pd.Series: The operating cash flow ratio value.
    """
    return operating_cash_flow / current_liabilities


def get_operating_cash_flow_sales_ratio(
    operating_cash_flow: SeriesOrFrame, revenue: SeriesOrFrame
) -> SeriesOrFrame:
    """
    Calculate the operating cash flow to sales ratio, a liquidity ratio that measures the ability of a company to generate
    cash from its sales.

    Args:
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.
        revenue (float or pd.Series): Sales of the company.

    Returns:
        float | pd.Series: The operating cash flow to sales ratio value.
    """
    return operating_cash_flow / revenue


def get_short_term_coverage_ratio(
    operating_cash_flow: SeriesOrFrame, short_term_debt: SeriesOrFrame
) -> SeriesOrFrame:
    """
    Calculate the short term coverage ratio, a liquidity ratio that measures a company's
    ability to pay off its short-term (current portion of) debt with its operating cash flow.

    The formula is as follows:

        Short Term Coverage Ratio = Cash Flow from Operations / Short Term Debt

    Also known as: short-term debt coverage.

    For more information about the method, see the following source:
    https://site.financialmodelingprep.com/developer/docs/formula

    Args:
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.
        short_term_debt (float or pd.Series): Short term (current) debt of the company.

    Returns:
        float | pd.Series: The short term coverage ratio value.
    """
    return operating_cash_flow / short_term_debt
