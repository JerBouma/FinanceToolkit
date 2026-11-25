"""Intrinsic Value Module"""

__docformat__ = "google"

from math import fsum

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def get_intrinsic_value(
    cash_flow: float,
    growth_rate: float,
    perpetual_growth_rate: float,
    weighted_average_cost_of_capital: float,
    cash_and_cash_equivalents: float,
    total_debt: float,
    shares_outstanding: float,
    periods: int = 5,
    short_term_investments: float = 0.0,
    minority_interest: float = 0.0,
    preferred_stock: float = 0.0,
) -> pd.DataFrame:
    """
    Intrinsic value is a fundamental concept in finance and investing that represents the
    true worth of an asset, independent of its current market price.

    This function estimates the intrinsic value of a company using a Discounted Cash Flow
    (DCF) model: it projects a base cash flow forward at a constant growth rate, discounts a
    terminal value off the final projection, and discounts every projected cash flow back to
    the present at the weighted average cost of capital.

    The formula is as follows:

        - Cash Flow Projection_t = Cash Flow_t-1 * (1 + Growth Rate)
        - Terminal Value = Last Cash Flow Projection * (1 + Perpetual Growth Rate) /
        (Weighted Average Cost of Capital — Perpetual Growth Rate)
        - Present Value of Cash Flows = Sum of Cash Flow Projection_t / (1 + Weighted Average
        Cost of Capital)^t for t = 1 to Periods
        - Present Value of Terminal Value = Terminal Value / (1 + Weighted Average Cost of
        Capital)^Periods
        - Enterprise Value = Present Value of Cash Flows + Present Value of Terminal Value
        - Equity Value = Enterprise Value + Cash and Cash Equivalents + Short Term Investments —
        Total Debt — Minority Interest — Preferred Stock
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
        short_term_investments (float, optional): The marketable securities that are not cash
        equivalents, added back with the cash. Defaults to 0.0.
        minority_interest (float, optional): The minority (non-controlling) interest, deducted as a
        claim on the Enterprise Value that does not belong to the shareholders. Defaults to 0.0.
        preferred_stock (float, optional): The preferred stock, deducted as a claim senior to the
        common shareholders. Defaults to 0.0.

    Returns:
        pd.DataFrame: A DataFrame containing the Terminal Value, Cash Flow Projection (the last
        projected cash flow), Present Value of Cash Flows, Present Value of Terminal Value,
        Enterprise Value, Equity Value and Intrinsic Value.

    Notes:
    - The results are highly dependent on the input. Therefore, think carefully about each input
    parameter to ensure the results are accurate (given your beliefs).
    - The Weighted Average Cost of Capital must be greater than the Perpetual Growth Rate,
    otherwise the Terminal Value formula divides by a non-positive number. A ValueError is
    raised when it is not.
    - The cash flows are discounted at the end of each period (no mid-year convention), and the
    Terminal Value at the end of the last projected period. The Present Value of Cash Flows and
    the Present Value of Terminal Value add up to the Enterprise Value, which shows how much of
    it comes from the years beyond the projection.
    - The Weighted Average Cost of Capital is the discount rate of the cash flow to all capital
    providers, so the base cash flow should be the Free Cash Flow to the Firm. Debt is subtracted
    from the Enterprise Value afterwards, which counts interest twice when the base cash flow is
    already after interest (such as the reported Free Cash Flow).
    - The cash and the short term investments are added back together, the mirror image of the
    Enterprise Value that deducts cash and marketable securities together (Damodaran). The
    operating cash flows do not include the income on these assets, so they are valued separately.
    - The Enterprise Value of the whole firm also belongs to the minority shareholders of its
    subsidiaries and to the preferred shareholders, so their claims are deducted with the debt
    to arrive at the value of the common equity (Damodaran, Investment Valuation, chapter 16), the
    mirror image of the Enterprise Value that adds them.
    """
    if weighted_average_cost_of_capital <= perpetual_growth_rate:
        raise ValueError(
            f"The Weighted Average Cost of Capital ({weighted_average_cost_of_capital}) must be greater "
            f"than the Perpetual Growth Rate ({perpetual_growth_rate}): the Terminal Value of a cash flow "
            "growing at or above the discount rate forever is infinite."
        )

    components = {}

    # Project the cash flow forward. The given cash flow is the latest realized one and is therefore not itself discounted, the projection starts one period after it.  # noqa: E501
    cash_flow_projection = []
    projected_cash_flow = cash_flow

    for _ in range(periods):
        projected_cash_flow = projected_cash_flow * (1 + growth_rate)
        cash_flow_projection.append(projected_cash_flow)

    # Calculate the Terminal Value
    terminal_value = (
        cash_flow_projection[-1]
        * (1 + perpetual_growth_rate)
        / (weighted_average_cost_of_capital - perpetual_growth_rate)
    )

    # Calculate the Present Value based on the Discounted Cash Flow Formula
    cash_flow_present_value = [
        cash_flow_value / (1 + weighted_average_cost_of_capital) ** (index + 1)
        for index, cash_flow_value in enumerate(cash_flow_projection)
    ]

    # The Terminal Value sits at the end of the projection horizon and is therefore discounted over the number of periods, not one period further.  # noqa: E501
    terminal_value_present_value = (
        terminal_value / (1 + weighted_average_cost_of_capital) ** periods
    )

    # Calculate the Enterprise Value, using fsum so the total is correctly rounded rather than accumulation-order dependent (builtin sum only compensates on 3.12+)  # noqa: E501
    enterprise_value = fsum([*cash_flow_present_value, terminal_value_present_value])

    # Calculate the Equity Value: the cash and the marketable securities belong to the shareholders on top of the operating Enterprise Value, while the debt, the minority interest and the preferred stock are claims ahead of them.  # noqa: E501
    equity_value = (
        enterprise_value
        + cash_and_cash_equivalents
        + short_term_investments
        - total_debt
        - minority_interest
        - preferred_stock
    )

    # Calculate the Intrinsic Value; without shares (none reported, or 0) there is no value
    # per share rather than an infinite one.
    intrinsic_value = (
        equity_value / shares_outstanding if shares_outstanding > 0 else np.nan
    )

    # Combine the components into a dictionary
    components = {
        "Terminal Value": terminal_value,
        "Cash Flow Projection": cash_flow_projection[-1],
        "Present Value of Cash Flows": fsum(cash_flow_present_value),
        "Present Value of Terminal Value": terminal_value_present_value,
        "Enterprise Value": enterprise_value,
        "Equity Value": equity_value,
        "Intrinsic Value": intrinsic_value,
    }

    return pd.DataFrame.from_dict(
        components, orient="index", columns=[f"Periods = {periods}"]
    )


def get_gordon_growth_model(
    dividends_per_share: float,
    rate_of_return: float,
    growth_rate: float,
):
    """
    Calculates the intrinsic value of a stock using the Gordon Growth Model.

    The Gordon Growth Model is a method for calculating the intrinsic
    value of a stock, based on a future series of dividends that grow at a constant rate. It
    is a popular and straightforward variant of the dividend discount model (DDM). The Gordon
    Growth Model assumes that dividends increase at a constant rate indefinitely. The model is
    named after Myron J. Gordon, Professor Emeritus of Finance at the Rotman School of
    Management, University of Toronto, who originally published it (with Eli Shapiro) in 1956
    and developed it further in a 1959 paper.

    The formula is as follows:

    - Intrinsic Value = (Dividends Per Share * (1 + Growth Rate)) / (Rate of Return - Growth Rate)

    Also known as: Gordon Growth Model, GGM, dividend discount model (single-stage/constant-growth
    variant).

    Args:
        dividends_per_share (float): the dividends per share.
        rate_of_return (float): the rate of return.
        growth_rate (float): the growth rate.

    Returns:
        float: the intrinsic value of the stock.

    Notes:
    - The Rate of Return must be greater than the Growth Rate, otherwise the formula divides by a
    non-positive number. A ValueError is raised when it is not.

    References:
    - Gordon, Myron J. "Dividends, Earnings, and Stock Prices." The Review of Economics and
    Statistics, Vol. 41, No. 2, 1959, pp. 99-105.
    """
    if rate_of_return <= growth_rate:
        raise ValueError(
            f"The Rate of Return ({rate_of_return}) must be greater than the Growth Rate ({growth_rate}): "
            "dividends growing at or above the discount rate forever have an infinite value."
        )

    return (dividends_per_share * (1 + growth_rate)) / (rate_of_return - growth_rate)


def get_graham_number(
    earnings_per_share: pd.Series | pd.DataFrame,
    book_value_per_share: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Graham Number, a conservative estimate of a stock's fair value based
    on its earnings and book value, as devised by Benjamin Graham.

    Args:
        earnings_per_share (float or pd.Series): Earnings per share of the company.
        book_value_per_share (float or pd.Series): Book value per share of the company.

    Returns:
        float | pd.Series: The Graham Number value, missing where earnings or book value
        are not positive.
    """
    # The formula assumes a profitable company with positive equity: a loss and negative
    # equity would multiply into a positive fair value, and a single negative one into
    # the square root of a negative number.
    product = 22.5 * earnings_per_share * book_value_per_share
    profitable_with_equity = (earnings_per_share > 0) & (book_value_per_share > 0)

    if isinstance(product, pd.Series | pd.DataFrame):
        return np.sqrt(product.where(profitable_with_equity))

    return float(np.sqrt(product)) if profitable_with_equity else np.nan


def get_graham_intrinsic_value(
    earnings_per_share: float | pd.Series | pd.DataFrame,
    growth_rate: float | pd.Series | pd.DataFrame,
    bond_yield: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculate the intrinsic value of a share with Benjamin Graham's growth formula, in the
    form he gave it in 1974 with the yield of high grade corporate bonds.

    The formula is as follows:

        Intrinsic Value = Earnings per Share x (8.5 + 2 x g) x 4.4 / Y

    Graham wrote g, the expected annual growth rate of the earnings over the next 7 to 10
    years, and Y, the current yield of AAA corporate bonds, in percent (5 for 5%), and 4.4 is
    the average AAA yield of 1962. Here both are decimals (0.05 for 5%), which the formula
    multiplies by 100 before it applies Graham's constants, so the value is the same.

    Args:
        earnings_per_share (float | pd.Series | pd.DataFrame): The annual earnings per share.
        growth_rate (float | pd.Series | pd.DataFrame): The expected annual growth rate of
            the earnings, as a decimal.
        bond_yield (float | pd.Series | pd.DataFrame): The yield of AAA corporate bonds, as a
            decimal.

    Returns:
        float | pd.Series | pd.DataFrame: The intrinsic value per share, NaN where the
        earnings or the bond yield are not positive.

    References:
    - Graham, Benjamin. "The Intelligent Investor." Fourth revised edition, Harper & Row,
    1973, chapter 11 (Intrinsic Value = Earnings per Share x (8.5 + 2 x g)).
    - Graham, Benjamin. "The Decade 1965-1974: Its Significance for Financial Analysts." In
    "The Renaissance of Value," Financial Analysts Research Foundation, 1974 (the 4.4 / Y
    adjustment).
    """
    # Graham's constants are in percent: 8.5 is the multiple of a company without growth,
    # 2 the multiple per percentage point of growth and 4.4 the AAA yield of 1962 in percent.
    value = (
        earnings_per_share * (8.5 + 2 * growth_rate * 100) * 4.4 / (bond_yield * 100)
    )

    # A loss has no growth value, and a yield of zero or less divides by nothing meaningful.
    valid = (earnings_per_share > 0) & (bond_yield > 0)

    if isinstance(valid, pd.Series | pd.DataFrame):
        return value.where(valid)
    if isinstance(value, pd.Series | pd.DataFrame):
        return value if valid else value * np.nan

    return float(value) if valid else np.nan


def get_free_cash_flow_to_firm(
    net_operating_profit_after_taxes: float | pd.Series | pd.DataFrame,
    depreciation_and_amortization: float | pd.Series | pd.DataFrame,
    capital_expenditure: float | pd.Series | pd.DataFrame,
    change_in_net_working_capital: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Free Cash Flow to the Firm (FCFF) is the cash flow available to all providers of
    capital, both debt and equity holders.

    It is what is left after the company has paid all of its operating expenses and invested
    in the assets (fixed assets and working capital) needed to sustain its operations.
    Because it is measured before any financing cash flows (interest, debt repayments or
    issuances, dividends), FCFF is capital-structure neutral, which makes it the appropriate
    cash flow to discount at the Weighted Average Cost of Capital (WACC) when valuing the
    Enterprise Value of a company directly (as opposed to FCFE, which values equity directly
    and should be discounted at the Cost of Equity).

    The formula is as follows:

        - FCFF = NOPAT + Depreciation and Amortization - Capital Expenditure -
        Change in Net Working Capital

    Also known as: FCFF, unlevered free cash flow.

    Args:
        net_operating_profit_after_taxes (float | pd.Series | pd.DataFrame): The Net
        Operating Profit After Taxes (NOPAT), i.e. EBIT * (1 - Effective Tax Rate). See
        `get_net_operating_profit_after_taxes` in the EVA module.
        depreciation_and_amortization (float | pd.Series | pd.DataFrame): The depreciation
        and amortization of the company, added back because it is a non-cash expense.
        capital_expenditure (float | pd.Series | pd.DataFrame): The capital expenditure of
        the company, expressed as a positive number representing the amount of cash spent on
        fixed assets.
        change_in_net_working_capital (float | pd.Series | pd.DataFrame): The change in net
        working capital of the company, expressed as a positive number when working capital
        has increased (a use of cash) and a negative number when it has decreased (a source
        of cash).

    Returns:
        float | pd.Series | pd.DataFrame: The Free Cash Flow to the Firm (FCFF).

    Notes:
    - This function expects capital_expenditure and change_in_net_working_capital as
    positive-magnitude figures (spend / increase), matching the standard academic formula.
    Financial statement data sources, including the cash flow statement used elsewhere in
    this toolkit, often store these figures using a cash-flow-impact sign convention instead
    (i.e. a use of cash is already negative); when sourcing these inputs from such a
    statement, negate them first so they match the convention expected here.
    - FCFF is the cash flow base typically used together with the Weighted Average Cost of
    Capital (WACC) in a DCF valuation, see `get_intrinsic_value`.
    """
    if not isinstance(
        net_operating_profit_after_taxes, int | float | pd.Series | pd.DataFrame
    ):
        raise TypeError(
            "net_operating_profit_after_taxes must be a float, pd.Series or pd.DataFrame, "
            f"not {type(net_operating_profit_after_taxes)}."
        )

    return (
        net_operating_profit_after_taxes
        + depreciation_and_amortization
        - capital_expenditure
        - change_in_net_working_capital
    )


def get_free_cash_flow_to_equity(
    net_income: float | pd.Series | pd.DataFrame,
    depreciation_and_amortization: float | pd.Series | pd.DataFrame,
    capital_expenditure: float | pd.Series | pd.DataFrame,
    change_in_net_working_capital: float | pd.Series | pd.DataFrame,
    net_borrowing: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Free Cash Flow to Equity (FCFE) is the cash flow available to a company's common equity
    holders.

    It is what is left after all operating expenses, reinvestment needs (capital expenditure
    and working capital), and net payments to (or from) debt holders have been accounted
    for. Unlike FCFF, FCFE is a levered cash flow measure, so it should be discounted at the
    Cost of Equity, not the Weighted Average Cost of Capital, when used in a valuation.

    The formula is as follows:

        - FCFE = Net Income + Depreciation and Amortization - Capital Expenditure -
        Change in Net Working Capital + Net Borrowing

    Also known as: FCFE, levered free cash flow.

    Args:
        net_income (float | pd.Series | pd.DataFrame): The net income (or loss) of a company
        for the period.
        depreciation_and_amortization (float | pd.Series | pd.DataFrame): The depreciation
        and amortization of the company, added back because it is a non-cash expense.
        capital_expenditure (float | pd.Series | pd.DataFrame): The capital expenditure of
        the company, expressed as a positive number representing the amount of cash spent on
        fixed assets.
        change_in_net_working_capital (float | pd.Series | pd.DataFrame): The change in net
        working capital of the company, expressed as a positive number when working capital
        has increased (a use of cash) and a negative number when it has decreased (a source
        of cash).
        net_borrowing (float | pd.Series | pd.DataFrame): The net amount of new debt raised
        during the period (new debt issued minus debt repaid), expressed as a positive number
        when the company is a net borrower and a negative number when it is a net repayer of
        debt.

    Returns:
        float | pd.Series | pd.DataFrame: The Free Cash Flow to Equity (FCFE).

    Notes:
    - This function expects capital_expenditure and change_in_net_working_capital as
    positive-magnitude figures (spend / increase), matching the standard academic formula and
    mirroring `get_free_cash_flow_to_firm`. Financial statement data sources often store these
    figures using a cash-flow-impact sign convention instead (i.e. a use of cash is already
    negative); when sourcing these inputs from such a statement, negate them first.
    net_borrowing, by contrast, is typically already reported using the cash-flow-impact
    convention directly (positive = cash inflow from net new debt) and does not need negating.
    - FCFE can also be derived from FCFF as: FCFE = FCFF - Interest Expense *
    (1 - Effective Tax Rate) + Net Borrowing. The direct formula used here avoids the need to
    separately track interest expense.
    """
    if not isinstance(net_income, int | float | pd.Series | pd.DataFrame):
        raise TypeError(
            f"net_income must be a float, pd.Series or pd.DataFrame, not {type(net_income)}."
        )

    return (
        net_income
        + depreciation_and_amortization
        - capital_expenditure
        - change_in_net_working_capital
        + net_borrowing
    )


def get_two_stage_dividend_discount_model(
    dividends_per_share: float,
    rate_of_return: float,
    high_growth_rate: float,
    stable_growth_rate: float,
    high_growth_periods: int = 5,
    transition_periods: int = 0,
) -> pd.DataFrame:
    """
    The Two-Stage Dividend Discount Model extends the (single-stage) Gordon Growth Model to
    companies that are not expected to grow at a constant rate forever.

    It explicitly projects and discounts dividends over an initial "high-growth" phase (e.g.
    a young, fast-growing company), and then, from the end of that phase onward, assumes
    dividends settle into perpetual growth at a lower, more sustainable "stable" rate —
    valued with the Gordon Growth Model as a terminal value.

    The formula is as follows:

        - Dividend Projection_t = Dividends Per Share * (1 + High Growth Rate)^t,
        for t = 1, ..., High Growth Periods
        - High-Growth Phase Present Value = Sum of Dividend Projection_t / (1 + Rate of
        Return)^t
        - Terminal Value = Dividend Projection_(High Growth Periods) * (1 + Stable Growth
        Rate) / (Rate of Return - Stable Growth Rate)
        - Terminal Value Present Value = Terminal Value / (1 + Rate of Return)^(High Growth
        Periods)
        - Intrinsic Value = High-Growth Phase Present Value + Terminal Value Present Value

    Also known as: two-stage DDM, two-stage dividend discount model.

    Args:
        dividends_per_share (float): The most recent (base) dividends per share, used as the
        starting point for the high-growth phase projections.
        rate_of_return (float): The discount rate (required rate of return) used to discount
        both the high-growth phase dividends and the Terminal Value back to their present
        value.
        high_growth_rate (float): The constant growth rate applied to dividends during the
        explicit high-growth phase.
        stable_growth_rate (float): The perpetual (terminal) growth rate applied to dividends
        from the end of the high-growth phase onward.
        high_growth_periods (int, optional): The number of periods in the explicit
        high-growth phase. Defaults to 5.
        transition_periods (int, optional): The number of periods between the high-growth
        phase and the stable phase in which the growth rate declines linearly from the high to
        the stable growth rate, which makes it the Three-Stage Dividend Discount Model. In
        transition period k (of n) dividends grow at High Growth Rate - (High Growth Rate -
        Stable Growth Rate) * k / (n + 1). Defaults to 0, no transition phase.

    Returns:
        pd.DataFrame: A DataFrame containing the final high-growth dividend, the present
        value of the high-growth phase, the present value of the transition phase (only with
        transition periods), the Terminal Value (and its present value), and the resulting
        Intrinsic Value.

    Notes:
    - The Rate of Return must be greater than the Stable Growth Rate, otherwise the Terminal
    Value formula divides by a non-positive number. A ValueError is raised when it is not.
    - The results are highly dependent on the input assumptions, in particular the length of
    the high-growth phase and the two growth rates. A high-growth rate that is maintained for
    too long, or a stable growth rate too close to the discount rate, will produce unrealistic
    valuations.
    - Unlike the single-stage Gordon Growth Model, this model allows the high-growth phase to
    exceed the discount rate (i.e. high_growth_rate can be greater than rate_of_return),
    since that phase is discounted explicitly rather than valued as a perpetuity.
    """
    if not isinstance(dividends_per_share, int | float):
        raise TypeError(
            "dividends_per_share must be a float or int, "
            f"not {type(dividends_per_share)}."
        )

    if rate_of_return <= stable_growth_rate:
        raise ValueError(
            f"The Rate of Return ({rate_of_return}) must be greater than the Stable Growth Rate "
            f"({stable_growth_rate}): dividends growing at or above the discount rate forever have an "
            "infinite Terminal Value."
        )

    dividend_projection = [dividends_per_share]

    for _ in range(1, high_growth_periods + 1):
        dividend_projection.append(dividend_projection[-1] * (1 + high_growth_rate))

    # fsum rather than sum so the total is correctly rounded rather than accumulation-order dependent (builtin sum only compensates on 3.12+)  # noqa: E501
    high_growth_present_value = fsum(
        dividend / (1 + rate_of_return) ** period
        for period, dividend in enumerate(dividend_projection[1:], start=1)
    )

    final_high_growth_dividend = dividend_projection[-1]

    # The growth rate steps down in equal parts from the high to the stable rate, reaching the stable rate in the first period after the transition.  # noqa: E501
    transition_growth_rates = [
        high_growth_rate
        - (high_growth_rate - stable_growth_rate) * period / (transition_periods + 1)
        for period in range(1, transition_periods + 1)
    ]
    for growth_rate in transition_growth_rates:
        dividend_projection.append(dividend_projection[-1] * (1 + growth_rate))

    transition_present_value = fsum(
        dividend / (1 + rate_of_return) ** period
        for period, dividend in enumerate(
            dividend_projection[high_growth_periods + 1 :],
            start=high_growth_periods + 1,
        )
    )

    explicit_periods = high_growth_periods + transition_periods
    terminal_value = (
        dividend_projection[-1]
        * (1 + stable_growth_rate)
        / (rate_of_return - stable_growth_rate)
    )
    terminal_value_present_value = (
        terminal_value / (1 + rate_of_return) ** explicit_periods
    )

    intrinsic_value = (
        high_growth_present_value
        + transition_present_value
        + terminal_value_present_value
    )

    components = {
        "Final High-Growth Dividend": final_high_growth_dividend,
        "High-Growth Phase Present Value": high_growth_present_value,
    }
    if transition_periods:
        components["Transition Phase Present Value"] = transition_present_value
    components |= {
        "Terminal Value": terminal_value,
        "Terminal Value Present Value": terminal_value_present_value,
        "Intrinsic Value": intrinsic_value,
    }

    return pd.DataFrame.from_dict(
        components,
        orient="index",
        columns=[
            f"High-Growth Periods = {high_growth_periods}"
            + (
                f", Transition Periods = {transition_periods}"
                if transition_periods
                else ""
            )
        ],
    )


def get_h_model(
    dividends_per_share: float | pd.Series | pd.DataFrame,
    rate_of_return: float,
    high_growth_rate: float,
    stable_growth_rate: float,
    high_growth_periods: int = 10,
) -> float | pd.Series | pd.DataFrame:
    """
    The H-Model of Fuller and Hsia (1984) shows the value of a stock whose dividend growth
    declines linearly to a stable rate.

    The growth starts at a high rate and declines over a number of periods, after which it
    grows at the stable rate forever. It approximates the three-stage dividend discount
    model in one closed-form expression: the Gordon Growth Model at the stable rate, plus
    the value of the extra growth during the decline.

    The formula is as follows:

        - H = High Growth Periods / 2 (the half-life of the high-growth period)
        - Intrinsic Value = Dividends Per Share * (1 + Stable Growth Rate) / (Rate of Return -
        Stable Growth Rate) + Dividends Per Share * H * (High Growth Rate - Stable Growth Rate)
        / (Rate of Return - Stable Growth Rate)

    Also known as: H-Model, Fuller-Hsia model.

    Args:
        dividends_per_share (float | pd.Series | pd.DataFrame): The current (annual) dividends
        per share.
        rate_of_return (float): The required rate of return.
        high_growth_rate (float): The growth rate at the start of the decline.
        stable_growth_rate (float): The growth rate the decline ends at and that holds forever
        after it.
        high_growth_periods (int, optional): The number of years over which the growth rate
        declines from the high to the stable rate. Defaults to 10.

    Returns:
        float | pd.Series | pd.DataFrame: The intrinsic value per share.

    Notes:
    - The Rate of Return must be greater than the Stable Growth Rate; a ValueError is raised
    when it is not.
    - The H-Model is an approximation: it is close to the exact value of a linearly declining
    growth rate when the high-growth period is short or the gap between the two growth rates
    is small, and overstates it as either grows.
    - Fuller, R. J. and C.-C. Hsia, "A Simplified Common Stock Valuation Model," Financial
    Analysts Journal 40(5), 1984, pp. 49-56.
    """
    if rate_of_return <= stable_growth_rate:
        raise ValueError(
            f"The Rate of Return ({rate_of_return}) must be greater than the Stable Growth Rate "
            f"({stable_growth_rate}): dividends growing at or above the discount rate forever "
            "have an infinite value."
        )

    half_life = high_growth_periods / 2
    spread = rate_of_return - stable_growth_rate

    return (
        dividends_per_share * (1 + stable_growth_rate) / spread
        + dividends_per_share
        * half_life
        * (high_growth_rate - stable_growth_rate)
        / spread
    )


def get_justified_price_to_earnings(
    payout_ratio: pd.Series | pd.DataFrame,
    rate_of_return: float | pd.DataFrame,
    growth_rate: pd.Series | pd.DataFrame,
    leading: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    The justified Price-to-Earnings ratio is the P/E the Gordon Growth Model implies for a
    company that pays out a constant share of its earnings, which grow at a constant rate.

    The formula is as follows:

        - Justified Trailing P/E = Payout Ratio * (1 + Growth Rate) / (Rate of Return - Growth
        Rate)
        - Justified Leading P/E = Payout Ratio / (Rate of Return - Growth Rate)

    Args:
        payout_ratio (pd.Series | pd.DataFrame): The dividends as a share of the earnings.
        rate_of_return (float | pd.DataFrame): The required rate of return, one number or one
        per company and period.
        growth_rate (pd.Series | pd.DataFrame): The constant growth rate of the earnings and
        dividends.
        leading (bool, optional): Whether to return the leading P/E (on next year's earnings)
        instead of the trailing P/E (on the last year's earnings). Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: The justified P/E. NaN where the growth rate is at or above
        the rate of return, where the model has no finite value.
    """
    spread = (rate_of_return - growth_rate).where(rate_of_return > growth_rate)
    if leading:
        return payout_ratio / spread
    return payout_ratio * (1 + growth_rate) / spread


def get_justified_price_to_book(
    return_on_equity: pd.Series | pd.DataFrame,
    rate_of_return: float | pd.DataFrame,
    growth_rate: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The justified Price-to-Book ratio is the P/B the residual income model implies for a
    company with a constant Return on Equity whose book value grows at a constant rate.

    The formula is as follows:

        - Justified P/B = (Return on Equity - Growth Rate) / (Rate of Return - Growth Rate)

    Args:
        return_on_equity (pd.Series | pd.DataFrame): The Return on Equity, on the equity at the
        beginning of the period.
        rate_of_return (float | pd.DataFrame): The required rate of return, one number or one
        per company and period.
        growth_rate (pd.Series | pd.DataFrame): The constant growth rate of the book value.

    Returns:
        pd.Series | pd.DataFrame: The justified P/B. NaN where the growth rate is at or above
        the rate of return, where the model has no finite value.
    """
    spread = (rate_of_return - growth_rate).where(rate_of_return > growth_rate)
    return (return_on_equity - growth_rate) / spread


def get_residual_income(
    net_income: float | pd.Series | pd.DataFrame,
    cost_of_equity: float | pd.Series | pd.DataFrame,
    book_value_of_equity: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Residual Income is a measure of the profit a company generates in excess of the return
    required by its equity holders.

    It is the equity-side counterpart to Economic Value Added (EVA): where EVA nets the cost
    of both debt and equity capital off of NOPAT, Residual Income nets only the cost of
    equity capital off of Net Income, since equity holders' claim is on Net Income (a profit
    already net of interest paid to debt holders). It underpins the Residual Income Model,
    an alternative equity valuation lens to a traditional Discounted Cash Flow (DCF) that is
    particularly useful when a company's free cash flows are negative or unpredictable but
    its accounting earnings are more stable.

    The formula is as follows:

        - Residual Income_t = Net Income_t - (Cost of Equity * Book Value of Equity_t-1)

    Also known as: RI, economic profit (equity variant), abnormal earnings.

    Args:
        net_income (float | pd.Series | pd.DataFrame): The net income (or loss) of a company
        for the period.
        cost_of_equity (float | pd.Series | pd.DataFrame): The cost of equity, i.e. the
        return required by the company's shareholders. See `get_cost_of_equity` in the WACC
        module.
        book_value_of_equity (float | pd.Series | pd.DataFrame): The book value of common
        equity of the company (Total Shareholder Equity, excluding preferred equity) at the
        beginning of the period, i.e. the capital the shareholders had invested while the Net
        Income was earned (Ohlson, 1995).

    Returns:
        float | pd.Series | pd.DataFrame: The Residual Income.

    Notes:
    - A positive Residual Income indicates that the company generated more profit than
    equity holders required given the capital they have invested, i.e. it created value for
    shareholders beyond their opportunity cost.
    - A negative Residual Income indicates the company failed to earn its equity holders'
    required return, even though it may still have reported a positive Net Income.
    - Unlike Economic Value Added, which is scaled by Invested Capital (debt + equity),
    Residual Income is scaled only by the Book Value of Equity, making it the natural
    equity-only analogue used together with a Residual Income valuation model, in the same
    way FCFE is the equity-only analogue of FCFF in a cash-flow-based valuation.

    References:
    - Ohlson, James A. "Earnings, Book Values, and Dividends in Equity Valuation."
    Contemporary Accounting Research, Vol. 11, No. 2, 1995, pp. 661-687.
    """
    if not isinstance(net_income, int | float | pd.Series | pd.DataFrame):
        raise TypeError(
            f"net_income must be a float, pd.Series or pd.DataFrame, not {type(net_income)}."
        )

    return net_income - (cost_of_equity * book_value_of_equity)


def _equity_value_per_share(
    cash_flow: float | np.ndarray,
    growth_rate: float | np.ndarray,
    perpetual_growth_rate: float | np.ndarray,
    weighted_average_cost_of_capital: float | np.ndarray,
    net_cash: float,
    shares_outstanding: float,
    periods: int,
) -> float | np.ndarray:
    """
    The DCF value per share of `get_intrinsic_value`, written so that every rate may also be
    an array of scenarios.
    """
    years = np.arange(1, periods + 1)
    growth = np.asarray(growth_rate, dtype=float)[..., None]
    perpetual = np.asarray(perpetual_growth_rate, dtype=float)
    wacc = np.asarray(weighted_average_cost_of_capital, dtype=float)
    projections = cash_flow * (1 + growth) ** years
    present_value = (projections / (1 + wacc[..., None]) ** years).sum(axis=-1)
    terminal = projections[..., -1] * (1 + perpetual) / (wacc - perpetual)
    enterprise_value = present_value + terminal / (1 + wacc) ** periods
    # Without shares (0 reported) there is no value per share rather than an infinite one.
    if not shares_outstanding > 0:
        return np.full(np.shape(enterprise_value), np.nan)

    return (enterprise_value + net_cash) / shares_outstanding


def get_implied_growth_rate(
    share_price: float,
    cash_flow: float,
    perpetual_growth_rate: float,
    weighted_average_cost_of_capital: float,
    cash_and_cash_equivalents: float,
    total_debt: float,
    shares_outstanding: float,
    periods: int = 5,
    short_term_investments: float = 0.0,
    minority_interest: float = 0.0,
    preferred_stock: float = 0.0,
) -> float:
    """
    The growth rate the share price implies (a reverse DCF).

    It is the yearly growth of the cash flow over the projection periods at which the
    Discounted Cash Flow value per share of `get_intrinsic_value` equals the share price.
    Rather than asking what a company is worth, it asks what the market expects, which can
    then be judged against the company's history and prospects.

    For more information, see the following book:

    - Rappaport, A. and Mauboussin, M.J. (2001). "Expectations Investing: Reading Stock
      Prices for Better Returns." Harvard Business School Press.

    Args:
        share_price (float): The current share price.
        cash_flow (float): The base cash flow, e.g. the latest Free Cash Flow.
        perpetual_growth_rate (float): The growth after the projection periods.
        weighted_average_cost_of_capital (float): The discount rate.
        cash_and_cash_equivalents (float): The cash and cash equivalents.
        total_debt (float): The total debt.
        shares_outstanding (float): The number of shares outstanding.
        periods (int, optional): The number of projection periods. Defaults to 5.
        short_term_investments (float, optional): The marketable securities that are not cash
            equivalents, added with the cash. Defaults to 0.0.
        minority_interest (float, optional): The minority interest, deducted with the debt.
            Defaults to 0.0.
        preferred_stock (float, optional): The preferred stock, deducted with the debt.
            Defaults to 0.0.

    Returns:
        float: The implied yearly growth rate, or NaN when no growth rate between -90% and
        +500% a year reproduces the share price (e.g. a negative cash flow).
    """
    if cash_flow <= 0 or weighted_average_cost_of_capital <= perpetual_growth_rate:
        return np.nan

    # The same net claims as get_intrinsic_value, so that the implied growth rate put back
    # into it returns the share price.
    net_cash = (
        cash_and_cash_equivalents
        + short_term_investments
        - total_debt
        - minority_interest
        - preferred_stock
    )

    def difference(growth: float) -> float:
        return (
            float(
                _equity_value_per_share(
                    cash_flow,
                    growth,
                    perpetual_growth_rate,
                    weighted_average_cost_of_capital,
                    net_cash,
                    shares_outstanding,
                    periods,
                )
            )
            - share_price
        )

    lower, upper = -0.9, 5.0
    # No sign change, or a missing input (NaN), means no growth rate gives the price.
    if not difference(lower) * difference(upper) <= 0:
        return np.nan

    return float(brentq(difference, lower, upper, maxiter=500))


def get_earnings_power_value(
    revenue: pd.Series,
    operating_income: pd.Series,
    income_tax_expense: pd.Series,
    income_before_tax: pd.Series,
    weighted_average_cost_of_capital: float,
    cash_and_cash_equivalents: float,
    total_debt: float,
    shares_outstanding: float,
    short_term_investments: float = 0.0,
    minority_interest: float = 0.0,
    preferred_stock: float = 0.0,
) -> pd.Series:
    """
    The Earnings Power Value (Greenwald): the value of the company's current, sustainable
    earnings if they never grew, NOPAT / WACC.

    Earnings are normalized by applying the average operating margin over the given periods
    to the latest revenue, which smooths out a single good or bad year. Comparing it with
    the market value shows how much of the price rests on growth: a price far above the EPV
    requires growth that creates value. The NOPAT is capitalized as a yearly amount, so
    every period should cover a year (for quarterly statements, the sum of four quarters).

    For more information, see the following book:

    - Greenwald, B.C.N., Kahn, J., Sonkin, P.D. and van Biema, M. (2001). "Value
      Investing: From Graham to Buffett and Beyond." Wiley.

    Args:
        revenue (pd.Series): The revenue over the normalization periods, each a year.
        operating_income (pd.Series): The operating income over the same periods.
        income_tax_expense (pd.Series): The income tax expense over the same periods.
        income_before_tax (pd.Series): The income before tax over the same periods.
        weighted_average_cost_of_capital (float): The discount rate.
        cash_and_cash_equivalents (float): The latest cash and cash equivalents.
        total_debt (float): The latest total debt.
        shares_outstanding (float): The number of shares outstanding.
        short_term_investments (float, optional): The marketable securities that are not cash
            equivalents, added with the cash. Defaults to 0.0.
        minority_interest (float, optional): The minority interest, deducted with the debt.
            Defaults to 0.0.
        preferred_stock (float, optional): The preferred stock, deducted with the debt.
            Defaults to 0.0.

    Returns:
        pd.Series: The Average Operating Margin, Normalized Operating Income, Tax Rate,
        Normalized NOPAT, Earnings Power Value (of the firm), Equity Value and Earnings
        Power Value per Share.

    Notes:
    - Maintenance capital expenditure is assumed to equal depreciation, so operating income
      stands in for the cash the business can distribute without shrinking.
    """
    # A period without revenue has no margin and one without income before tax no tax
    # rate: left out of the averages rather than turning them infinite.
    margin = float((operating_income / revenue.where(revenue > 0)).mean())
    normalized_operating_income = margin * float(revenue.iloc[-1])
    tax_rate = float(
        np.clip(
            (
                income_tax_expense / income_before_tax.where(income_before_tax != 0)
            ).mean(),
            0,
            1,
        )
    )
    nopat = normalized_operating_income * (1 - tax_rate)
    earnings_power_value = (
        nopat / weighted_average_cost_of_capital
        if weighted_average_cost_of_capital > 0
        else np.nan
    )
    # The Equity Value as in get_intrinsic_value: the cash and marketable securities added,
    # the claims ahead of the common shareholders deducted.
    equity_value = (
        earnings_power_value
        + cash_and_cash_equivalents
        + short_term_investments
        - total_debt
        - minority_interest
        - preferred_stock
    )

    return pd.Series(
        {
            "Average Operating Margin": margin,
            "Normalized Operating Income": normalized_operating_income,
            "Tax Rate": tax_rate,
            "Normalized NOPAT": nopat,
            "Earnings Power Value": earnings_power_value,
            "Equity Value": equity_value,
            "Earnings Power Value per Share": (
                equity_value / shares_outstanding if shares_outstanding > 0 else np.nan
            ),
        }
    )


def get_monte_carlo_intrinsic_value(
    cash_flow: float,
    growth_rate: float,
    growth_rate_volatility: float,
    perpetual_growth_rate: float,
    perpetual_growth_rate_volatility: float,
    weighted_average_cost_of_capital: float,
    weighted_average_cost_of_capital_volatility: float,
    cash_and_cash_equivalents: float,
    total_debt: float,
    shares_outstanding: float,
    periods: int = 5,
    simulations: int = 10000,
    seed: int | None = 0,
    short_term_investments: float = 0.0,
    minority_interest: float = 0.0,
    preferred_stock: float = 0.0,
) -> np.ndarray:
    """
    The distribution of the Discounted Cash Flow value per share when the growth rate, the
    perpetual growth rate and the discount rate are uncertain.

    Each simulation draws the three rates from normal distributions around their expected
    values and calculates the value as in `get_intrinsic_value`. Draws in which the discount
    rate does not exceed the perpetual growth rate are redrawn, as the terminal value is
    undefined there.

    Args:
        cash_flow (float): The base cash flow, e.g. the latest Free Cash Flow.
        growth_rate (float): The expected yearly growth over the projection periods.
        growth_rate_volatility (float): The standard deviation of the growth rate.
        perpetual_growth_rate (float): The expected growth after the projection periods.
        perpetual_growth_rate_volatility (float): Its standard deviation.
        weighted_average_cost_of_capital (float): The expected discount rate.
        weighted_average_cost_of_capital_volatility (float): Its standard deviation.
        cash_and_cash_equivalents (float): The cash and cash equivalents.
        total_debt (float): The total debt.
        shares_outstanding (float): The number of shares outstanding.
        periods (int, optional): The number of projection periods. Defaults to 5.
        simulations (int, optional): The number of simulations. Defaults to 10,000.
        seed (int | None, optional): The seed of the random numbers, for reproducible
            results. Defaults to 0.
        short_term_investments (float, optional): The marketable securities that are not cash
            equivalents, added with the cash. Defaults to 0.0.
        minority_interest (float, optional): The minority interest, deducted with the debt.
            Defaults to 0.0.
        preferred_stock (float, optional): The preferred stock, deducted with the debt.
            Defaults to 0.0.

    Returns:
        np.ndarray: The simulated values per share.
    """
    rng = np.random.default_rng(seed)
    growth = rng.normal(growth_rate, growth_rate_volatility, simulations)
    perpetual = rng.normal(
        perpetual_growth_rate, perpetual_growth_rate_volatility, simulations
    )
    wacc = rng.normal(
        weighted_average_cost_of_capital,
        weighted_average_cost_of_capital_volatility,
        simulations,
    )

    # Redraw the discount rates that do not exceed the perpetual growth rate (with a small
    # margin, as a spread close to zero makes the terminal value explode).
    for _ in range(100):
        invalid = wacc <= perpetual + 0.005
        if not invalid.any():
            break
        wacc[invalid] = rng.normal(
            weighted_average_cost_of_capital,
            weighted_average_cost_of_capital_volatility,
            int(invalid.sum()),
        )
    else:
        raise ValueError(
            "The discount rate is too often below the perpetual growth rate; lower the "
            "volatilities or widen the gap between the two."
        )

    return _equity_value_per_share(
        cash_flow,
        growth,
        perpetual,
        wacc,
        cash_and_cash_equivalents
        + short_term_investments
        - total_debt
        - minority_interest
        - preferred_stock,
        shares_outstanding,
        periods,
    )


def get_owner_earnings(
    net_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    maintenance_capital_expenditure: pd.Series | pd.DataFrame,
    change_in_working_capital: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Buffett's owner earnings: the cash a business could hand its owners while keeping its
    competitive position and unit volume, the earnings the owners of the business own.

    The formula is as follows:

        - Owner Earnings = Net Income + Depreciation and Amortization - Maintenance Capital
          Expenditure - Change in Working Capital

    Args:
        net_income (pd.Series | pd.DataFrame): The reported net income.
        depreciation_and_amortization (pd.Series | pd.DataFrame): The depreciation and
            amortization (Buffett also adds other non-cash charges).
        maintenance_capital_expenditure (pd.Series | pd.DataFrame): The capital expenditure
            needed to keep the business as it is, as a positive amount.
        change_in_working_capital (pd.Series | pd.DataFrame): The increase in working capital,
            positive when working capital grows.

    Returns:
        pd.Series | pd.DataFrame: The owner earnings.

    References:
    - Buffett, W. E. (1986). "Chairman's Letter." Berkshire Hathaway Inc. Annual Report 1986,
      appendix on purchase-price accounting adjustments and owner earnings.
    """
    return (
        net_income
        + depreciation_and_amortization
        - maintenance_capital_expenditure
        - change_in_working_capital
    )


def get_greenwald_maintenance_capital_expenditure(
    capital_expenditure: pd.Series | pd.DataFrame,
    property_plant_and_equipment_to_revenue: pd.Series | pd.DataFrame,
    change_in_revenue: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Greenwald's estimate of the maintenance capital expenditure.

    It is the capital expenditure minus the investment that the growth in revenue required
    at the company's ratio of fixed assets to revenue (the growth capital expenditure).

    The formula is as follows:

        - Growth Capital Expenditure = PP&E / Revenue * Change in Revenue (0 when revenue falls)
        - Maintenance Capital Expenditure = Capital Expenditure - Growth Capital Expenditure,
          at least 0

    Args:
        capital_expenditure (pd.Series | pd.DataFrame): The capital expenditure, positive.
        property_plant_and_equipment_to_revenue (pd.Series | pd.DataFrame): The property, plant
            and equipment per unit of revenue.
        change_in_revenue (pd.Series | pd.DataFrame): The change in revenue over the period.

    Returns:
        pd.Series | pd.DataFrame: The maintenance capital expenditure.

    References:
    - Greenwald, B. C. N., Kahn, J., Sonkin, P. D. and van Biema, M. (2001). "Value Investing:
      From Graham to Buffett and Beyond." Wiley, chapter 6.
    """
    growth_capital_expenditure = property_plant_and_equipment_to_revenue * (
        change_in_revenue.clip(lower=0)
    )
    # Growth capital expenditure above the actual capital expenditure leaves no maintenance part, not a negative one.  # noqa: E501
    return (capital_expenditure - growth_capital_expenditure).clip(lower=0)
