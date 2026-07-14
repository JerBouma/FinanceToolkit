"""Intrinsic Value Module"""

__docformat__ = "google"

import numpy as np
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
    Intrinsic value is a fundamental concept in finance and investing that represents the true
    worth of an asset, independent of its current market price. This function estimates the
    intrinsic value of a company using a Discounted Cash Flow (DCF) model: it projects a base
    cash flow forward at a constant growth rate, discounts a terminal value off the final
    projection, and discounts every projected cash flow back to the present at the weighted
    average cost of capital.

    The formula is as follows:

        - Cash Flow Projection_t = Cash Flow_t-1 * (1 + Growth Rate)
        - Terminal Value = Last Cash Flow Projection * (1 + Perpetual Growth Rate) /
        (Weighted Average Cost of Capital — Perpetual Growth Rate)
        - Enterprise Value = Sum of Present Value of Cash Flow Projections + Terminal Value
        - Equity Value = Enterprise Value — Total Debt + Cash and Cash Equivalents
        - Intrinsic Value = Equity Value / Total Shares Outstanding

    Args:
        cash_flow (float): The base cash flow (e.g. the most recent period's Free Cash Flow)
        used as the starting point for the cash flow projections.
        growth_rate (float): The growth rate used to project the cash flow forward each period.
        perpetual_growth_rate (float): The perpetual (terminal) growth rate used to calculate
        the Terminal Value from the final cash flow projection.
        weighted_average_cost_of_capital (float): The discount rate used to discount the
        projected cash flows and the Terminal Value back to their present value.
        cash_and_cash_equivalents (float): The cash and cash equivalents of the company.
        total_debt (float): The total debt of the company.
        shares_outstanding (float): The total shares outstanding of the company.
        periods (int, optional): The number of periods to project the cash flow for. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing the Terminal Value, Cash Flow Projection, Enterprise
        Value, Equity Value and Intrinsic Value.

    Notes:
    - The results are highly dependent on the input. Therefore, think carefully about each input
    parameter to ensure the results are accurate (given your beliefs).
    - The Weighted Average Cost of Capital must be greater than the Perpetual Growth Rate,
    otherwise the Terminal Value formula divides by a non-positive number.
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


def get_graham_number(
    earnings_per_share: pd.Series, book_value_per_share: pd.Series
) -> pd.Series:
    """
    Calculate the Graham Number, a conservative estimate of a stock's fair value based
    on its earnings and book value, as devised by Benjamin Graham.

    Args:
        earnings_per_share (float or pd.Series): Earnings per share of the company.
        book_value_per_share (float or pd.Series): Book value per share of the company.

    Returns:
        float | pd.Series: The Graham Number value.
    """
    return np.sqrt(22.5 * earnings_per_share * book_value_per_share)
