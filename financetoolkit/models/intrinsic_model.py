"""Intrinsic Value Module"""

__docformat__ = "google"

from math import fsum

import numpy as np
import pandas as pd
from scipy.optimize import brentq

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
    short_term_investments: float = 0.0,
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
        - Equity Value = Enterprise Value — Total Debt + Cash and Cash Equivalents + Short Term
        Investments
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

    Returns:
        pd.DataFrame: A DataFrame containing the Terminal Value, Cash Flow Projection, Enterprise
        Value, Equity Value and Intrinsic Value.

    Notes:
    - The results are highly dependent on the input. Therefore, think carefully about each input
    parameter to ensure the results are accurate (given your beliefs).
    - The Weighted Average Cost of Capital must be greater than the Perpetual Growth Rate,
    otherwise the Terminal Value formula divides by a non-positive number. A ValueError is
    raised when it is not.
    - The Weighted Average Cost of Capital is the discount rate of the cash flow to all capital
    providers, so the base cash flow should be the Free Cash Flow to the Firm. Debt is subtracted
    from the Enterprise Value afterwards, which counts interest twice when the base cash flow is
    already after interest (such as the reported Free Cash Flow).
    - The cash and the short term investments are added back together, the mirror image of the
    Enterprise Value that deducts cash and marketable securities together (Damodaran). The
    operating cash flows do not include the income on these assets, so they are valued separately.
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
    cash_flow_present_value.append(
        terminal_value / (1 + weighted_average_cost_of_capital) ** periods
    )

    # Calculate the Enterprise Value, using fsum so the total is correctly rounded rather than accumulation-order dependent (builtin sum only compensates on 3.12+)  # noqa: E501
    enterprise_value = fsum(cash_flow_present_value)

    # Calculate the Equity Value: the cash and the marketable securities belong to the shareholders on top of the operating Enterprise Value.  # noqa: E501
    equity_value = (
        enterprise_value
        + cash_and_cash_equivalents
        + short_term_investments
        - total_debt
    )

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

    The Gordon Growth Model (this function is spelled "Gorden" to match the controller method
    `get_gorden_growth_model`) is a method for calculating the intrinsic
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


def get_free_cash_flow_to_firm(
    net_operating_profit_after_taxes: float | pd.Series | pd.DataFrame,
    depreciation_and_amortization: float | pd.Series | pd.DataFrame,
    capital_expenditure: float | pd.Series | pd.DataFrame,
    change_in_net_working_capital: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Free Cash Flow to the Firm (FCFF) is the cash flow available to all providers of
    capital, both debt and equity holders, after the company has paid all of its operating
    expenses and invested in the assets (fixed assets and working capital) needed to sustain
    its operations. Because it is measured before any financing cash flows (interest, debt
    repayments or issuances, dividends), FCFF is capital-structure neutral, which makes it
    the appropriate cash flow to discount at the Weighted Average Cost of Capital (WACC) when
    valuing the Enterprise Value of a company directly (as opposed to FCFE, which values
    equity directly and should be discounted at the Cost of Equity).

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
    holders after all operating expenses, reinvestment needs (capital expenditure and working
    capital), and net payments to (or from) debt holders have been accounted for. Unlike
    FCFF, FCFE is a levered cash flow measure, so it should be discounted at the Cost of
    Equity, not the Weighted Average Cost of Capital, when used in a valuation.

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
) -> pd.DataFrame:
    """
    The Two-Stage Dividend Discount Model extends the (single-stage) Gordon Growth Model to
    companies that are not expected to grow at a constant rate forever. It explicitly
    projects and discounts dividends over an initial "high-growth" phase (e.g. a young,
    fast-growing company), and then, from the end of that phase onward, assumes dividends
    settle into perpetual growth at a lower, more sustainable "stable" rate — valued with the
    Gordon Growth Model as a terminal value.

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

    Returns:
        pd.DataFrame: A DataFrame containing the final high-growth dividend, the present
        value of the high-growth phase, the Terminal Value (and its present value), and the
        resulting Intrinsic Value.

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

    terminal_value = (
        dividend_projection[-1]
        * (1 + stable_growth_rate)
        / (rate_of_return - stable_growth_rate)
    )
    terminal_value_present_value = (
        terminal_value / (1 + rate_of_return) ** high_growth_periods
    )

    intrinsic_value = high_growth_present_value + terminal_value_present_value

    components = {
        "Final High-Growth Dividend": dividend_projection[-1],
        "High-Growth Phase Present Value": high_growth_present_value,
        "Terminal Value": terminal_value,
        "Terminal Value Present Value": terminal_value_present_value,
        "Intrinsic Value": intrinsic_value,
    }

    return pd.DataFrame.from_dict(
        components,
        orient="index",
        columns=[f"High-Growth Periods = {high_growth_periods}"],
    )


def get_residual_income(
    net_income: float | pd.Series | pd.DataFrame,
    cost_of_equity: float | pd.Series | pd.DataFrame,
    book_value_of_equity: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Residual Income is a measure of the profit a company generates in excess of the return
    required by its equity holders. It is the equity-side counterpart to Economic Value Added
    (EVA): where EVA nets the cost of both debt and equity capital off of NOPAT, Residual
    Income nets only the cost of equity capital off of Net Income, since equity holders'
    claim is on Net Income (a profit already net of interest paid to debt holders). It
    underpins the Residual Income Model, an alternative equity valuation lens to a
    traditional Discounted Cash Flow (DCF) that is particularly useful when a company's free
    cash flows are negative or unpredictable but its accounting earnings are more stable.

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
) -> float:
    """
    The growth rate the share price implies (a reverse DCF): the yearly growth of the cash
    flow over the projection periods at which the Discounted Cash Flow value per share of
    `get_intrinsic_value` equals the share price. Rather than asking what a company is
    worth, it asks what the market expects, which can then be judged against the company's
    history and prospects.

    For more information, see the following book:

    - Mauboussin, M.J. and Rappaport, A. (2001). "Expectations Investing: Reading Stock
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

    Returns:
        float: The implied yearly growth rate, or NaN when no growth rate between -90% and
        +500% a year reproduces the share price (e.g. a negative cash flow).
    """
    if cash_flow <= 0 or weighted_average_cost_of_capital <= perpetual_growth_rate:
        return np.nan

    net_cash = cash_and_cash_equivalents - total_debt

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
) -> pd.Series:
    """
    The Earnings Power Value (Greenwald): the value of the company's current, sustainable
    earnings if they never grew, NOPAT / WACC. Earnings are normalized by applying the
    average operating margin over the given periods to the latest revenue, which smooths out
    a single good or bad year. Comparing it with the market value shows how much of the
    price rests on growth: a price far above the EPV requires growth that creates value.
    The NOPAT is capitalized as a yearly amount, so every period should cover a year (for
    quarterly statements, the sum of four quarters).

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

    Returns:
        pd.Series: The Average Operating Margin, Normalized Operating Income, Tax Rate,
        Normalized NOPAT, Earnings Power Value (of the firm), Equity Value and Earnings
        Power Value per Share.

    Notes:
    - Maintenance capital expenditure is assumed to equal depreciation, so operating income
      stands in for the cash the business can distribute without shrinking.
    """
    margin = float((operating_income / revenue).mean())
    normalized_operating_income = margin * float(revenue.iloc[-1])
    tax_rate = float(np.clip((income_tax_expense / income_before_tax).mean(), 0, 1))
    nopat = normalized_operating_income * (1 - tax_rate)
    earnings_power_value = nopat / weighted_average_cost_of_capital
    equity_value = earnings_power_value + cash_and_cash_equivalents - total_debt

    return pd.Series(
        {
            "Average Operating Margin": margin,
            "Normalized Operating Income": normalized_operating_income,
            "Tax Rate": tax_rate,
            "Normalized NOPAT": nopat,
            "Earnings Power Value": earnings_power_value,
            "Equity Value": equity_value,
            "Earnings Power Value per Share": equity_value / shares_outstanding,
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
) -> np.ndarray:
    """
    The distribution of the Discounted Cash Flow value per share when the growth rate, the
    perpetual growth rate and the discount rate are uncertain. Each simulation draws the
    three rates from normal distributions around their expected values and calculates the
    value as in `get_intrinsic_value`. Draws in which the discount rate does not exceed the
    perpetual growth rate are redrawn, as the terminal value is undefined there.

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
        cash_and_cash_equivalents - total_debt,
        shares_outstanding,
        periods,
    )
