"""Weighted Average Cost of Capital Module"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.performance import performance_model
from financetoolkit.ratios import profitability_model, valuation_model
from financetoolkit.utilities.dataframe_model import concat_frames
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# pylint: disable=too-many-locals


def get_cost_of_equity(
    risk_free_rate: pd.Series,
    beta: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.Series | pd.DataFrame:
    """
    The cost of equity represents the return required by investors (shareholders) for
    holding shares of a company's common stock. It is a key component of the Weighted
    Average Cost of Capital (WACC) and is used to discount future cash flows in financial analysis.
    There are several methods to calculate the cost of equity, but one common approach is to
    use the Capital Asset Pricing Model (CAPM).

    The formula is as follows:

        Cost of Equity (Re) = Risk-Free Rate (Rf) + Beta (β) * (Market Return (Rm) — Risk-Free Rate (Rf))

    Args:
        risk_free_rate (float or pd.Series): The risk-free rate is the rate of return of a
        hypothetical investment with no risk of financial loss.
        beta (pd.Series | pd.DataFrame): Beta is a measure of the volatility, or systematic risk,
        of a security or portfolio compared to the market as a whole.
        benchmark_returns (float or pd.Series): The benchmark return is the return of a
        chosen benchmark, such as the S&P 500 or the Russell 2000.

    Returns:
        pd.Series | pd.DataFrame: The Cost of Equity, shaped like `beta`.
    """
    cost_of_equity = performance_model.get_capital_asset_pricing_model(
        risk_free_rate=risk_free_rate, beta=beta, benchmark_returns=benchmark_returns
    )

    return cost_of_equity


def get_cost_of_debt(
    interest_expense: pd.Series,
    total_debt: pd.Series,
) -> pd.DataFrame:
    """
    The cost of debt represents the interest rate a company pays on its outstanding debt,
    such as bonds, loans, or other forms of borrowed capital. It's a crucial component
    of the Weighted Average Cost of Capital (WACC) and is used in various financial analyses.
    Calculating the cost of debt is relatively straightforward.

    The formula is as follows:

        Cost of Debt (Rd) = Annual Interest Expense / Total Debt

    Args:
        interest_expense (float or pd.Series): Interest expense of the company.
        total_debt (float or pd.Series): Total debt of the company.

    Returns:
        pd.DataFrame: A DataFrame containing Cost of Debt
    """
    return interest_expense / total_debt


def get_corporate_tax_rate(
    income_tax_expense: pd.Series | pd.DataFrame | float,
    income_before_tax: pd.Series | pd.DataFrame | float,
) -> pd.Series | pd.DataFrame | float:
    """
    The Corporate Tax Rate is the effective tax rate (Income Tax Expense / Income Before Tax)
    kept within 0 and 1, and set to 0 for a period without a pre-tax profit. It is the tax rate
    the Weighted Average Cost of Capital applies to the tax shield on debt and the rate the Net
    Operating Profit After Taxes (Economic Value Added, Free Cash Flow to Firm and the
    Intrinsic Valuation built on it) taxes EBIT with, so that these models share one tax rate.

    The formula is as follows:

        Corporate Tax Rate = min(max(Income Tax Expense / Income Before Tax, 0), 1), and 0 when
        Income Before Tax <= 0

    Args:
        income_tax_expense (float, pd.Series or pd.DataFrame): Income tax expense of the company.
        income_before_tax (float, pd.Series or pd.DataFrame): Income before taxes of the company.

    Returns:
        float, pd.Series or pd.DataFrame: The Corporate Tax Rate as a decimal.

    Notes:
    - A tax benefit on a profit gives an effective rate below 0, which would turn the tax shield on
    debt into a tax penalty and raise NOPAT above EBIT; a rate above 1 would shield more than the
    interest itself and turn NOPAT negative on a positive EBIT. The rate is therefore kept within
    0 and 1.
    - Without a pre-tax profit there is no taxable income, so no tax is due on the operating
    profit nor shielded by the interest (Damodaran, Investment Valuation). The rate is 0 rather
    than the undefined or sign-flipped ratio of a tax benefit to a loss.
    """
    # Clip the effective tax rate to [0, 1]: a negative rate (a tax benefit on a profit) or a rate above 1 (tax above the profit) is not a marginal rate a tax shield or NOPAT can use.  # noqa: E501
    corporate_tax_rate = np.clip(
        profitability_model.get_effective_tax_rate(
            income_tax_expense, income_before_tax
        ),
        0,
        1,
    )

    # Without a pre-tax profit there is no taxable income, so the rate is 0 (Damodaran) rather than the ratio of a tax benefit to a loss.  # noqa: E501
    if isinstance(corporate_tax_rate, pd.Series | pd.DataFrame):
        corporate_tax_rate = corporate_tax_rate.mask(income_before_tax <= 0, 0)
    elif income_before_tax <= 0:
        corporate_tax_rate = 0.0

    return corporate_tax_rate


def get_weighted_average_cost_of_capital(
    share_price: pd.Series,
    total_shares_outstanding: pd.Series,
    interest_expense: pd.Series,
    total_debt: pd.Series,
    risk_free_rate: pd.Series,
    beta: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    income_tax_expense: pd.Series,
    income_before_tax: pd.Series,
) -> pd.DataFrame:
    """
    The Weighted Average Cost of Capital (WACC) is a financial metric used to estimate the
    cost of capital for a company. It represents the average rate of return a company must
    pay to its investors for using their capital. WACC takes into account the cost of both
    equity and debt, weighted by their respective proportions in the company's capital
    structure.

    The formula is as follows:

        WACC = (Market Value of Equity / Total Market Value) * Cost of Equity +
        (Market Value of Debt / Total Market Value) * Cost of Debt * (1 — Corporate Tax Rate)

    Args:
        share_price (float or pd.Series): The price of a single share of a company's stock.
        total_shares_outstanding (float or pd.Series): The total number of shares of a company's stock.
        interest_expense (float or pd.Series): Interest expense of the company.
        total_debt (float or pd.Series): Total debt of the company.
        risk_free_rate (float or pd.Series): The risk-free rate is the rate of return of a
        hypothetical investment with no risk of financial loss.
        beta (pd.Series | pd.DataFrame): Beta is a measure of the volatility, or systematic risk,
        of a security or portfolio compared to the market as a whole.
        benchmark_returns (float or pd.Series): The benchmark return is the return of a
        chosen benchmark, such as the S&P 500 or the Russell 2000.
        income_tax_expense (float or pd.Series): Income tax expense of the company.
        income_before_tax (float or pd.Series): Income before taxes of the company.

    Returns:
        pd.DataFrame: A DataFrame containing the Weighted Average Cost of Capital and the components
        it is built from (Market Value Equity, Market Value Debt, Cost of Equity, Cost of Debt and
        the Corporate Tax Rate).

    Notes:
    - The market value of equity is calculated as the share price multiplied by the total shares outstanding.
    - The market value of debt is calculated as the total debt. This is a simplification, as the market value of debt
    should be calculated as the present value of all future cash flows of the debt.
    - The cost of equity is calculated using the Capital Asset Pricing Model (CAPM).
    - The cost of debt is calculated as the interest expense divided by the total debt at the end of
    the period. Debt that is raised or repaid during the period makes this differ from the rate on the
    average debt outstanding.
    - The corporate tax rate is calculated as the income tax expense divided by the income before taxes,
    kept within 0 and 1, and 0 when there is no pre-tax profit (no taxable income for the interest to
    shield).
    """
    # Calculate the Market Value of Equity
    market_value_equity = valuation_model.get_market_cap(
        share_price, total_shares_outstanding
    )

    # Calculate the Market Value of Debt
    market_value_debt = total_debt

    # Calculate the Total Market Value
    market_value_total = market_value_equity + market_value_debt

    # Calculate Equity and Debt weights
    equity_weight = market_value_equity / market_value_total
    debt_weight = market_value_debt / market_value_total

    # Calculate the Cost of Equity
    cost_of_equity = get_cost_of_equity(risk_free_rate, beta, benchmark_returns).T

    # Calculate the Cost of Debt
    cost_of_debt = get_cost_of_debt(interest_expense, total_debt)

    # If the cost of debt is infinite, change it to 0; note that a negative infinity is just as possible as a positive one (a net interest income against zero debt) and must be caught too, which a membership test against np.inf alone would miss  # noqa: E501
    if np.isinf(cost_of_debt.to_numpy()).any():
        logger.info(
            "Please note that the Cost of Debt contains Inf. This is due to Total Debt being 0, "
            "therefore Cost of Debt is adjusted to 0 for those periods."
        )
        cost_of_debt = cost_of_debt.replace([np.inf, -np.inf], 0)

    # Calculate the Corporate Tax Rate, kept within 0 and 1 and 0 without a pre-tax profit (see get_corporate_tax_rate).  # noqa: E501
    corporate_tax_rate = get_corporate_tax_rate(income_tax_expense, income_before_tax)

    # Without debt there is no cost of debt (0 / 0 interest over debt) and the debt
    # carries no weight, so the WACC is the cost of equity rather than missing.
    # A period the balance sheet lacks compares as missing, not as debt-free.
    without_debt = (market_value_debt == 0).astype(bool)
    if isinstance(cost_of_debt, pd.DataFrame | pd.Series):
        without_debt = (
            without_debt.reindex_like(cost_of_debt).fillna(False).astype(bool)
        )
    cost_of_debt = cost_of_debt.mask(without_debt & cost_of_debt.isna(), 0)
    after_tax_cost_of_debt = (
        debt_weight * cost_of_debt * (1 - corporate_tax_rate)
    ).mask(without_debt, 0)

    # Calculate the Weighted Average Cost of Capital
    weighted_average_cost_of_capital = (
        equity_weight * cost_of_equity + after_tax_cost_of_debt
    )

    # Create a dictionary with the WACC components
    components = {
        "Market Value Equity": market_value_equity,
        "Market Value Debt": market_value_debt,
        "Cost of Equity": cost_of_equity,
        "Cost of Debt": cost_of_debt,
        "Corporate Tax Rate": corporate_tax_rate,
        "Weighted Average Cost of Capital": weighted_average_cost_of_capital,
    }

    if isinstance(market_value_equity, pd.DataFrame):
        return (
            concat_frames(components)
            .swaplevel(1, 0)
            .sort_index(level=0, sort_remaining=False)
        )

    return pd.DataFrame.from_dict(components, orient="index")
