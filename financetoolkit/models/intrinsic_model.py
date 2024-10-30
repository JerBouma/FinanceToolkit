"""Intrinsic Value Module"""

__docformat__ = "google"

import pandas as pd

# pylint: disable=too-many-locals


def get_intrinsic_value(
    cash_flow: float,
    growth_rate: float,
    perpetual_growth_rate: float,
    weighted_average_cost_of_capital: float,
    cash_and_cash_equivalents: float,
    total_debt: float,
    shares_outstanding: float,
    periods: int = 5,
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
        beta (float or pd.Series): Beta is a measure of the volatility, or systematic risk,
        of a security or portfolio compared to the market as a whole.
        benchmark_returns (float or pd.Series): The benchmark return is the return of a
        chosen benchmark, such as the S&P 500 or the Russell 2000.
        income_tax_expense (float or pd.Series): Income tax expense of the company.
        income_before_tax (float or pd.Series): Income before taxes of the company.

    Returns:
        pd.DataFrame: A DataFrame containing the Dupont analysis components.

    Notes:
    - The market value of equity is calculated as the share price multiplied by the total shares outstanding.
    - The market value of debt is calculated as the total debt. This is a simplification, as the market value of debt
    should be calculated as the present value of all future cash flows of the debt.
    - The cost of equity is calculated using the Capital Asset Pricing Model (CAPM).
    - The cost of debt is calculated as the interest expense divided by the total debt.
    - The corporate tax rate is calculated as the income tax expense divided by the income before taxes.
    """
    components = {}

    cash_flow_projection = [cash_flow]

    # Cash Flow to Use
    for period in range(1, periods + 1):
        if period == 1:
            cash_flow_projection.append(cash_flow_projection[0] * (1 + growth_rate))
        else:
            cash_flow_projection.append(
                cash_flow_projection[period - 1] * (1 + growth_rate)
            )

    # Calculate the Terminal Value
    terminal_value = (
        cash_flow_projection[-1]
        * (1 + perpetual_growth_rate)
        / (weighted_average_cost_of_capital - perpetual_growth_rate)
    )

    # Add Terminal Value to the end of the cash flow projection
    cash_flow_projection[-1] = cash_flow_projection[-1] + terminal_value

    # Calculate the Present Value based on the Discounted Cash Flow Formula
    cash_flow_present_value = []
    for index, cash_flow_value in enumerate(cash_flow_projection):
        cash_flow_present_value.append(
            cash_flow_value / (1 + weighted_average_cost_of_capital) ** (index + 1)
        )

    # Calculate the Enterprise Value
    enterprise_value = sum(cash_flow_present_value)

    # Calculate the Equity Value
    equity_value = enterprise_value + cash_and_cash_equivalents - total_debt

    # Calculate the Intrinsic Value
    intrinsic_value = equity_value / shares_outstanding

    # Combine the components into a dictionary
    components = {
        "Terminal Value": terminal_value,
        "Cash Flow Projection": cash_flow_projection[-1],
        "Enterprise Value": enterprise_value,
        "Equity Value": equity_value,
        "Intrinsic Value": intrinsic_value,
    }

    return pd.DataFrame.from_dict(
        components, orient="index", columns=[f"Periods = {periods}"]
    )


def get_gorden_growth_model(
    dividends_per_share: float,
    rate_of_return: float,
    growth_rate: float,
):
    """
    Calculates the intrinsic value of a stock using the Gorden Growth Model.

    The Gorden Growth Model is a method for calculating the intrinsic value of a stock,
    based on a future series of dividends that grow at a constant rate. It is a popular
    and straightforward variant of the dividend discount model (DDM). The Gorden Growth
    Model assumes that dividends increase at a constant rate indefinitely. The model
    is named after Myron J. Gorden of the University of Washington Foster School of
    Business, who originally published it in 1959.

    The formula is as follows:

    - Intrinsic Value = (Dividends Per Share * (1 + Growth Rate)) / (Rate of Return - Growth Rate)

    Args:
        dividends_per_share (float): the dividends per share.
        rate_of_return (float): the rate of return.
        growth_rate (float): the growth rate.

    Returns:
        float: the intrinsic value of the stock.
    """
    return (dividends_per_share * (1 + growth_rate)) / (rate_of_return - growth_rate)
