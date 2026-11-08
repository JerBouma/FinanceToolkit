"""Profitability Module"""

__docformat__ = "google"


import pandas as pd


def get_gross_margin(
    revenue: pd.Series | pd.DataFrame, cost_of_goods_sold: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
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


def get_operating_margin(
    operating_income: pd.Series | pd.DataFrame, revenue: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
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


def get_net_profit_margin(
    net_income: pd.Series | pd.DataFrame, revenue: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
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


def get_ebitda_margin(
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    revenue: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the EBITDA margin, a profitability ratio that measures the percentage of
    revenue that remains as earnings before interest, taxes, depreciation and
    amortization (EBITDA).

    EBITDA margin approximates a company's core operating profitability before the
    effects of financing decisions (interest), tax jurisdictions, and non-cash
    accounting choices around fixed and intangible assets (depreciation and
    amortization). This makes it a commonly used metric to compare operating
    performance across companies with different capital structures, tax regimes, and
    depreciation policies.

    The formula is as follows:

        EBITDA Margin = (Operating Income + Depreciation and Amortization) / Revenue

    Also known as: EBITDA-to-revenue ratio.

    Args:
        operating_income (float or pd.Series): Operating income of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and amortization of the company.
        revenue (float or pd.Series): Total revenue of the company.

    Returns:
        float | pd.Series: The EBITDA margin percentage value.
    """
    return (operating_income + depreciation_and_amortization) / revenue


def get_interest_burden_ratio(
    income_before_tax: pd.Series | pd.DataFrame,
    operating_income: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Compute the Interest Burden Ratio, the component of the extended (five-step) DuPont
    decomposition that isolates the drag interest expense places on operating profit.

    It divides earnings before tax (EBT) by earnings before interest and taxes (EBIT,
    proxied by operating income), so it expresses the share of operating profit that
    survives interest expense. For a company with debt it sits between 0 and 1, where a
    value close to 1 means interest barely dents operating profit and a low value
    signals a heavy interest load. Values slightly above 1 occur when non-operating
    income (e.g. interest income) exceeds interest expense.

    It should not be confused with the interest coverage ratio, which divides operating
    profit by interest expense and is unbounded above.

    The formula is as follows:

        Interest Burden Ratio = Income Before Tax / Operating Income

    Also known as: EBT to EBIT ratio.

    Args:
        income_before_tax (float or pd.Series): Income before tax of the company.
        operating_income (float or pd.Series): Operating income of the company.

    Returns:
        float | pd.Series: The Interest Burden Ratio
    """
    return income_before_tax / operating_income


def get_income_before_tax_profit_margin(
    income_before_tax: pd.Series | pd.DataFrame, revenue: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
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
    income_tax_expense: pd.Series | pd.DataFrame,
    income_before_tax: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the effective tax rate, a financial ratio that measures the percentage of pretax income
    that is paid as taxes.

    The rate is only meaningful for a positive income before tax: with a pre-tax loss the
    ratio of a tax charge (or benefit) to the loss flips sign. The Ratios module passes
    NaN for those periods; this function divides the inputs as given.

    Args:
        income_tax_expense (float or pd.Series): The income tax expense of the company.
        income_before_tax (float or pd.Series): The company's income before taxes.

    Returns:
        float | pd.Series: The effective tax rate value.
    """
    return income_tax_expense / income_before_tax


def get_return_on_assets(
    net_income: pd.Series | pd.DataFrame, average_total_assets: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the return on assets (ROA), a profitability ratio that measures how
    efficiently a company uses its assets to generate profits.

    Args:
        net_income (float or pd.Series): Net income of the company.
        average_total_assets (float or pd.Series): Average total assets of the company.
            This is typically calculated as (beginning total assets + ending total assets) / 2.

    Returns:
        float | pd.Series: The ROA percentage value.
    """
    return net_income / average_total_assets


def get_cash_return_on_assets(
    operating_cash_flow: pd.Series | pd.DataFrame,
    average_total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the cash return on assets (Cash ROA), a profitability ratio that measures
    how efficiently a company uses its assets to generate operating cash flow.

    Unlike the return on assets, which uses accrual-based net income, the cash return
    on assets uses operating cash flow, making it less sensitive to non-cash accounting
    choices (e.g. depreciation method, revenue recognition timing, working capital
    accruals). Comparing cash ROA to ROA is a useful earnings-quality cross-check: a
    cash ROA that persistently trails ROA can indicate that reported profits are not
    being converted into cash.

    The formula is as follows:

        Cash Return on Assets = Operating Cash Flow / Average Total Assets

    Also known as: cash ROA.

    Args:
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.
        average_total_assets (float or pd.Series): Average total assets of the company.
            This is typically calculated as (beginning total assets + ending total assets) / 2.

    Returns:
        float | pd.Series: The cash return on assets percentage value.
    """
    return operating_cash_flow / average_total_assets


def get_return_on_equity(
    net_income: pd.Series | pd.DataFrame, average_total_equity: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the return on equity (ROE), a profitability ratio that measures how
    efficiently a company generates profits using its shareholders' equity.

    Args:
        net_income (float or pd.Series): Net income of the company.
        average_total_equity (float or pd.Series): Average total equity of the company.
            This is typically calculated as (beginning total equity + ending total equity) / 2.

    Returns:
        float | pd.Series: The ROE percentage value.
    """
    return net_income / average_total_equity


def get_return_on_invested_capital(
    operating_income: pd.Series | pd.DataFrame,
    income_tax_expense: pd.Series | pd.DataFrame,
    income_before_tax: pd.Series | pd.DataFrame,
    average_total_equity: pd.Series | pd.DataFrame,
    average_total_debt: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the return on invested capital, a financial ratio that measures the company's
    after-tax operating return on the capital invested in it, including both equity and debt.

    The numerator is the net operating profit after taxes (NOPAT): operating income (EBIT)
    taxed at the effective tax rate (Damodaran). Because EBIT is before interest, the
    return does not depend on how the capital is financed. The effective tax rate is
    Income Tax Expense / Income Before Tax, bounded to [0, 1]; with a pre-tax loss it is
    undefined and set to 0, so an operating loss is not offset by a tax credit. A missing
    tax expense or income before tax gives NaN.

    The formula is as follows:

        Return on Invested Capital = Operating Income x (1 - Effective Tax Rate) /
            (Average Total Equity + Average Total Debt)

    Also known as: ROIC, return on capital (ROC).

    Args:
        operating_income (float or pd.Series): The company's operating income (EBIT).
        income_tax_expense (float or pd.Series): The company's income tax expense.
        income_before_tax (float or pd.Series): The company's income before taxes.
        average_total_equity (float or pd.Series): Average total equity of the company.
            This is typically calculated as (beginning total equity + ending total equity) / 2.
        average_total_debt (float or pd.Series): Average total debt of the company.
            This is typically calculated as (beginning total debt + ending total debt) / 2.

    Returns:
        float | pd.Series: The return on invested capital value.
    """
    # Only a pre-tax loss (or zero) sets the rate to 0; a missing tax expense or income before
    # tax stays NaN, so an unreported figure is not read as a tax-free year.
    effective_tax_rate = (
        income_tax_expense / income_before_tax.where(income_before_tax > 0)
    ).clip(lower=0, upper=1)
    effective_tax_rate = effective_tax_rate.mask(income_before_tax <= 0, 0)

    net_operating_profit_after_taxes = operating_income * (1 - effective_tax_rate)

    return net_operating_profit_after_taxes / (
        average_total_equity + average_total_debt
    )


def get_income_quality_ratio(
    cash_flow_from_operating_activities: pd.Series | pd.DataFrame,
    net_income: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
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
    net_income: pd.Series | pd.DataFrame,
    average_total_assets: pd.Series | pd.DataFrame,
    average_goodwill_and_intangible_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the return on tangible assets, which measures the amount of profit
    generated relative to a company's tangible assets: total assets minus goodwill and
    other intangible assets.

    Liabilities are not deducted, so the ratio measures the return on the whole tangible
    asset base, however it is financed, and stays comparable between companies with
    different leverage. `get_tangible_asset_value` in the valuation module makes the same
    deduction of goodwill and intangible assets but also deducts the liabilities.

    The formula is as follows:

        Return on Tangible Assets = Net Income / (Average Total Assets -
            Average Goodwill and Intangible Assets)

    Args:
        net_income (float or pd.Series): The net income of the company.
        average_total_assets (float or pd.Series): Average total assets of the company.
            This is typically calculated as (beginning total assets + ending total assets) / 2.
        average_goodwill_and_intangible_assets (float or pd.Series): Average goodwill and
            other intangible assets of the company.

    Returns:
        float | pd.Series: The return on tangible assets value.
    """
    average_tangible_assets = (
        average_total_assets - average_goodwill_and_intangible_assets
    )

    return net_income / average_tangible_assets


def get_return_on_capital_employed(
    operating_income: pd.Series | pd.DataFrame,
    total_assets: pd.Series | pd.DataFrame,
    total_current_liabilities: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the return on capital employed (ROCE), a profitability ratio that measures
    the amount of return a company generates from the capital it has invested in the business.

    The formula is as follows:

        Return on Capital Employed = EBIT / (Total Assets - Total Current Liabilities)

    Operating income is used as EBIT, so non-operating income and expenses do not enter
    the return.

    Args:
        operating_income (float or pd.Series): Operating income (EBIT) of the company.
        total_assets (float or pd.Series): Total assets of the company.
        total_current_liabilities (float or pd.Series): Total current liabilities of the company.

    Returns:
        float | pd.Series: The ROCE value.
    """
    return operating_income / (total_assets - total_current_liabilities)


def get_net_income_per_ebt(
    net_income: pd.Series | pd.DataFrame, income_tax_expense: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
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
    free_cash_flow: pd.Series | pd.DataFrame,
    operating_cash_flow: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
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


def get_free_cash_flow_margin(
    free_cash_flow: pd.Series | pd.DataFrame, revenue: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the free cash flow margin, a profitability ratio that measures the
    percentage of revenue that is converted into free cash flow.

    Unlike the net profit margin, which can be distorted by non-cash accounting items
    (e.g. depreciation, stock-based compensation, deferred taxes) and by working
    capital timing, the free cash flow margin reflects the cash a company actually
    generates, after capital expenditures, for every dollar of revenue. A persistently
    low or declining free cash flow margin relative to the net profit margin can be a
    quality-of-earnings warning sign.

    The formula is as follows:

        Free Cash Flow Margin = Free Cash Flow / Revenue

    Args:
        free_cash_flow (float or pd.Series): Free cash flow of the company.
        revenue (float or pd.Series): Total revenue of the company.

    Returns:
        float | pd.Series: The free cash flow margin percentage value.
    """
    return free_cash_flow / revenue


def get_tax_burden_ratio(
    net_income: pd.Series | pd.DataFrame, income_before_tax: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
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
    earnings_before_tax: pd.Series | pd.DataFrame,
    earnings_before_interest_and_taxes: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
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
    earnings_before_interest_and_taxes: pd.Series | pd.DataFrame,
    revenue: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
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


def get_cash_tax_rate(
    income_taxes_paid: pd.Series | pd.DataFrame,
    income_before_tax: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the cash tax rate, which measures the percentage of pretax income that
    is actually paid out in cash taxes, as opposed to the accrual-based effective tax
    rate.

    As for the effective tax rate, the rate is only meaningful for a positive income before
    tax. The Ratios module passes NaN for periods with a pre-tax loss; this function divides
    the inputs as given.

    Args:
        income_taxes_paid (float or pd.Series): Income taxes paid by the company, as
            reported in the Cash Flow Statement.
        income_before_tax (float or pd.Series): The company's income before taxes.

    Returns:
        float | pd.Series: The cash tax rate value.
    """
    return income_taxes_paid / income_before_tax


def get_tax_rate_divergence(
    cash_tax_rate: pd.Series | pd.DataFrame,
    effective_tax_rate: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the tax rate divergence, which measures the difference between the cash
    tax rate and the accrual-based effective tax rate.

    A persistently positive divergence indicates the company is paying more in cash
    taxes than it is recognizing as tax expense (e.g. due to the reversal of deferred
    tax liabilities), while a persistently negative divergence indicates the opposite
    and can be a quality-of-earnings red flag if it stems from aggressive tax
    deferral rather than timing differences.

    Args:
        cash_tax_rate (float or pd.Series): The cash tax rate of the company.
        effective_tax_rate (float or pd.Series): The effective (accrual-based) tax rate
            of the company.

    Returns:
        float | pd.Series: The tax rate divergence value.
    """
    return cash_tax_rate - effective_tax_rate


def get_gross_profitability(
    revenue: pd.Series | pd.DataFrame,
    cost_of_goods_sold: pd.Series | pd.DataFrame,
    total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the gross profitability of Novy-Marx (2013): the gross profit a company
    earns on its assets. Gross profit is the cleanest accounting measure of economic
    profitability, as everything below it (selling, research and administrative
    expenses, interest, taxes) is partly investment or financing, and it predicts
    returns as well as book-to-market does.

    The formula is as follows:

        Gross Profitability = (Revenue - Cost of Goods Sold) / Total Assets

    The paper takes the total assets at the end of the fiscal year, the same date as
    the gross profit, so the assets are given as a point in time and not averaged.

    Also known as: GP/A, gross profits to assets.

    Args:
        revenue (float or pd.Series): Total revenue of the company.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        total_assets (float or pd.Series): Total assets at the end of the period.

    Returns:
        float | pd.Series: The gross profitability value.
    """
    return (revenue - cost_of_goods_sold) / total_assets


def get_operating_profitability(
    revenue: pd.Series | pd.DataFrame,
    cost_of_goods_sold: pd.Series | pd.DataFrame,
    selling_general_and_administrative_expenses: pd.Series | pd.DataFrame,
    research_and_development_expenses: pd.Series | pd.DataFrame,
    interest_expense: pd.Series | pd.DataFrame,
    total_shareholder_equity: pd.Series | pd.DataFrame,
    deferred_tax_liabilities: pd.Series | pd.DataFrame,
    preferred_stock: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the operating profitability of Fama and French (2015), the profitability
    measure behind the RMW (robust minus weak) factor of the five-factor model: the
    operating profit after interest that a company earns on its book equity.

    The formula is as follows:

        Operating Profitability = (Revenue - Cost of Goods Sold - Selling, General and
            Administrative Expenses - Interest Expense) / Book Equity

        Book Equity = Total Shareholder Equity + Deferred Tax Liabilities - Preferred Stock

    The selling, general and administrative expenses of Compustat (XSGA), which Fama
    and French use, include research and development, while the statements here report
    research and development separately, so both are deducted. Book equity follows
    Fama and French: the stockholders' equity (without minority interest) plus the
    balance sheet deferred taxes, minus the preferred stock, at the end of the fiscal
    year (a point in time, not averaged). A company with a book equity of zero or less
    is left out by Fama and French and has no value (NaN).

    Also known as: OP, RMW profitability, Fama-French profitability.

    Args:
        revenue (float or pd.Series): Total revenue of the company.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        selling_general_and_administrative_expenses (float or pd.Series): Selling,
            general and administrative expenses, without research and development.
        research_and_development_expenses (float or pd.Series): Research and development
            expenses.
        interest_expense (float or pd.Series): Interest expense of the company.
        total_shareholder_equity (float or pd.Series): Stockholders' equity without
            minority interest, at the end of the period.
        deferred_tax_liabilities (float or pd.Series): Balance sheet deferred tax
            liabilities, at the end of the period.
        preferred_stock (float or pd.Series): Preferred stock, at the end of the period.

    Returns:
        float | pd.Series: The operating profitability value.
    """
    book_equity = total_shareholder_equity + deferred_tax_liabilities - preferred_stock

    operating_profit = (
        revenue
        - cost_of_goods_sold
        - selling_general_and_administrative_expenses
        - research_and_development_expenses
        - interest_expense
    )

    return operating_profit / book_equity.where(book_equity > 0)


def get_cash_based_operating_profitability(
    revenue: pd.Series | pd.DataFrame,
    cost_of_goods_sold: pd.Series | pd.DataFrame,
    selling_general_and_administrative_expenses: pd.Series | pd.DataFrame,
    change_in_receivables: pd.Series | pd.DataFrame,
    change_in_inventory: pd.Series | pd.DataFrame,
    change_in_prepaids: pd.Series | pd.DataFrame,
    change_in_deferred_revenue: pd.Series | pd.DataFrame,
    change_in_accounts_payable: pd.Series | pd.DataFrame,
    change_in_accrued_expenses: pd.Series | pd.DataFrame,
    beginning_total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the cash-based operating profitability of Ball, Gerakos, Linnainmaa and
    Nikolaev (2016): the operating profitability of Ball et al. (2015) with the
    working capital accruals taken out, so that it measures the operating profit
    received in cash. It subsumes accruals in predicting returns and predicts them
    better than the measures that include accruals.

    The formula is as follows:

        Cash-Based Operating Profitability = (Revenue - Cost of Goods Sold - Selling,
            General and Administrative Expenses (without research and development)
            - Δ Receivables - Δ Inventory - Δ Prepaids + Δ Deferred Revenue
            + Δ Accounts Payable + Δ Accrued Expenses) / Beginning Total Assets

    Research and development is not deducted, as the paper adds it back (XSGA - XRD in
    Compustat): it is an investment rather than an expense of the period. The changes
    are those of the balances over the window of the flows, and the total assets are
    those at the start of it (the end of the period before), as in Ball et al. (2016).

    Also known as: CbOP, cash operating profitability, cash-based profitability.

    Args:
        revenue (float or pd.Series): Total revenue of the company.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        selling_general_and_administrative_expenses (float or pd.Series): Selling,
            general and administrative expenses, without research and development.
        change_in_receivables (float or pd.Series): The change in the total receivables.
        change_in_inventory (float or pd.Series): The change in the inventory.
        change_in_prepaids (float or pd.Series): The change in the prepaid expenses.
        change_in_deferred_revenue (float or pd.Series): The change in the current and
            non-current deferred revenue.
        change_in_accounts_payable (float or pd.Series): The change in the accounts payable.
        change_in_accrued_expenses (float or pd.Series): The change in the accrued expenses.
        beginning_total_assets (float or pd.Series): Total assets at the start of the
            window of the flows.

    Returns:
        float | pd.Series: The cash-based operating profitability value.
    """
    operating_profit = (
        revenue - cost_of_goods_sold - selling_general_and_administrative_expenses
    )

    cash_based_operating_profit = (
        operating_profit
        - change_in_receivables
        - change_in_inventory
        - change_in_prepaids
        + change_in_deferred_revenue
        + change_in_accounts_payable
        + change_in_accrued_expenses
    )

    return cash_based_operating_profit / beginning_total_assets


def get_cash_return_on_invested_capital(
    free_cash_flow: pd.Series | pd.DataFrame,
    average_total_equity: pd.Series | pd.DataFrame,
    average_total_debt: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the cash return on invested capital (CROIC), the free cash flow a company
    generates on the capital invested in it by shareholders and lenders.

    The formula is as follows:

        Cash Return on Invested Capital = Free Cash Flow / (Average Total Equity +
            Average Total Debt)

    The invested capital is the same as in the return on invested capital, so the two
    can be compared directly: a CROIC well below the ROIC means the operating profit is
    not turned into free cash flow.

    Also known as: CROIC, cash ROIC, free cash flow return on invested capital.

    Args:
        free_cash_flow (float or pd.Series): Free cash flow of the company.
        average_total_equity (float or pd.Series): Average total equity of the company.
        average_total_debt (float or pd.Series): Average total debt of the company.

    Returns:
        float | pd.Series: The cash return on invested capital value.
    """
    return free_cash_flow / (average_total_equity + average_total_debt)


def get_free_cash_flow_conversion(
    free_cash_flow: pd.Series | pd.DataFrame, net_income: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the free cash flow conversion, the share of net income that turns into
    free cash flow.

    The formula is as follows:

        Free Cash Flow Conversion = Free Cash Flow / Net Income

    With a loss the ratio has no meaning (it flips sign), so a net income of zero or
    less gives NaN.

    Also known as: FCF conversion, cash conversion ratio, FCF to net income.

    Args:
        free_cash_flow (float or pd.Series): Free cash flow of the company.
        net_income (float or pd.Series): Net income of the company.

    Returns:
        float | pd.Series: The free cash flow conversion value.
    """
    return free_cash_flow / net_income.where(net_income > 0)


def get_free_cash_flow_to_ebitda_ratio(
    free_cash_flow: pd.Series | pd.DataFrame,
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the free cash flow to EBITDA ratio, the share of EBITDA that remains as
    free cash flow after interest, taxes, working capital and capital expenditure.

    The formula is as follows:

        Free Cash Flow to EBITDA = Free Cash Flow / (Operating Income + Depreciation
            and Amortization)

    A non-positive EBITDA gives NaN, as the ratio flips sign.

    Also known as: FCF/EBITDA, EBITDA cash conversion, cash conversion of EBITDA.

    Args:
        free_cash_flow (float or pd.Series): Free cash flow of the company.
        operating_income (float or pd.Series): Operating income (EBIT) of the company.
        depreciation_and_amortization (float or pd.Series): Depreciation and
            amortization of the company.

    Returns:
        float | pd.Series: The free cash flow to EBITDA value.
    """
    ebitda = operating_income + depreciation_and_amortization

    return free_cash_flow / ebitda.where(ebitda > 0)


def get_percent_accruals(
    net_income: pd.Series | pd.DataFrame,
    operating_cash_flow: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the percent accruals of Hafzalla, Lundholm and Van Winkle (2011): the
    accruals (the part of earnings not received in cash) relative to the size of the
    earnings rather than to total assets. It separates the companies whose earnings
    are mostly accruals, which go on to underperform, better than accruals scaled by
    total assets do.

    The formula is as follows:

        Percent Accruals = (Net Income - Cash Flow from Operations) / |Net Income|

    A net income of zero gives NaN.

    Also known as: percent operating accruals, PctAcc, accruals to earnings.

    Args:
        net_income (float or pd.Series): Net income of the company.
        operating_cash_flow (float or pd.Series): Cash flow from operations of the company.

    Returns:
        float | pd.Series: The percent accruals value.
    """
    return (net_income - operating_cash_flow) / net_income.abs().where(net_income != 0)
