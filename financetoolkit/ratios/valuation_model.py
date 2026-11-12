"""Valuation Module"""

__docformat__ = "google"


import numpy as np
import pandas as pd


def get_earnings_per_share(
    net_income: pd.Series | pd.DataFrame,
    preferred_dividends: pd.Series | float,
    average_outstanding_shares: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the earnings per share (EPS), a valuation ratio that measures the
    amount of net income earned per share of outstanding common stock.

    Args:
        net_income (float or pd.Series): Net income of the company.
        preferred_dividends (float or pd.Series): Preferred dividends of the company.
        average_outstanding_shares (float or pd.Series): Average outstanding shares of the company.

    Returns:
        float | pd.Series: The EPS value.
    """
    return (net_income - preferred_dividends) / average_outstanding_shares


def get_revenue_per_share(
    total_revenue: pd.Series | pd.DataFrame,
    shares_outstanding: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the revenue per share, a valuation ratio that measures the amount of
    revenue generated per outstanding share of a company's stock.

    Args:
        total_revenue (float or pd.Series): Total revenue of the company.
        shares_outstanding (float or pd.Series): Total number of outstanding shares of the company.

    Returns:
        float | pd.Series: The revenue per share value.
    """
    return total_revenue / shares_outstanding


def get_price_to_earnings_ratio(
    stock_price: pd.Series | pd.DataFrame, earnings_per_share: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the price earnings ratio (P/E), a valuation ratio that compares a company's
    stock price to its earnings per share.

    Args:
        stock_price (float or pd.Series): Stock price of the company.
        earnings_per_share (float or pd.Series): Earnings per share of the company.

    Returns:
        float | pd.Series: The P/E ratio value.
    """
    return stock_price / earnings_per_share


def get_price_to_earnings_growth_ratio(
    price_earnings: pd.Series | pd.DataFrame,
    growth_rate: pd.Series | pd.DataFrame | float,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the price earnings to growth (PEG) ratio, a valuation metric that measures the ratio
    of the price-to-earnings ratio to a company specific growth rate.

    The company specific growth rate is typically the expected growth rate of the company's earnings
    over a specific period of time, usually 5 to 10 years. This can be defined by it's EBITDA,
    EPS, or other growth metrics.

    Args:
        price_earnings (float or pd.Series): The company's price-to-earnings ratio.
        growth_rate (float or pd.Series): The growth rate of the company.

    Returns:
        float | pd.Series: The PEG ratio value.
    """
    return price_earnings / growth_rate


def get_estimated_eps_growth_rate(
    estimated_eps: float | pd.Series | pd.DataFrame,
    trailing_eps: float | pd.Series | pd.DataFrame,
    years: float | pd.Series | pd.DataFrame = 1,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculate the estimated EPS growth rate, i.e. the annualized growth implied by the
    analyst consensus (average) EPS estimate for a future period relative to the company's
    most recently reported actual (trailing) EPS.

    Unlike the generic `growth=True` parameter available on every ratio and model
    method — which compares two already-reported historical periods — this bridges an
    actual reported figure to a forward-looking analyst estimate, which is what a
    forward-looking metric such as the PEG ratio needs. The growth is annualized over
    the years between the two, so the estimates of later years give a yearly rate that
    is comparable with that of the next year.

    The formula is as follows:

        Estimated EPS Growth Rate = (Estimated EPS / Trailing EPS) ^ (1 / Years) - 1

    A compound rate only exists when both EPS figures are positive and the estimate lies
    in the future (Years > 0); otherwise the growth rate is NaN.

    Args:
        estimated_eps (float | pd.Series | pd.DataFrame): The analyst consensus
            (average) EPS estimate for a future period.
        trailing_eps (float | pd.Series | pd.DataFrame): The company's most recently
            reported actual EPS.
        years (float | pd.Series | pd.DataFrame, optional): The number of years between the
            trailing EPS and the estimate. Defaults to 1.

    Returns:
        float | pd.Series | pd.DataFrame: The annualized estimated EPS growth rate.
    """
    valid = (estimated_eps > 0) & (trailing_eps > 0) & (years > 0)

    # NumPy division also handles scalar zero denominators, which are masked below.
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_ratio = np.divide(estimated_eps, trailing_eps)
        growth_rate = np.power(eps_ratio, np.divide(1, years)) - 1

    if isinstance(growth_rate, pd.Series | pd.DataFrame):
        return growth_rate.where(valid)

    return growth_rate if valid else float("nan")


def get_book_value_per_share(
    total_shareholder_equity: pd.Series | pd.DataFrame,
    preferred_equity: pd.Series | pd.DataFrame,
    common_shares_outstanding: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the book value per share, a valuation ratio that measures the amount of
    common equity value per share outstanding.

    Args:
        total_shareholder_equity (float or pd.Series): Total equity of the company.
        preferred_equity (float or pd.Series): Preferred equity of the company.
        common_shares_outstanding (float or pd.Series): Common shares outstanding of the company.

    Returns:
        float | pd.Series: The book value per share value.
    """
    return (total_shareholder_equity - preferred_equity) / common_shares_outstanding


def get_price_to_book_ratio(
    price_per_share: pd.Series | pd.DataFrame,
    book_value_per_share: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the price to book ratio, a valuation ratio that compares a company's market
    price to its book value per share.

    Args:
        price_per_share (float or pd.Series): Price per share of the company.
        book_value_per_share (float or pd.Series): Book value per share of the company.

    Returns:
        float | pd.Series: The price to book ratio value.
    """
    return price_per_share / book_value_per_share


def get_interest_debt_per_share(
    interest_expense: pd.Series | pd.DataFrame,
    total_debt: pd.Series | pd.DataFrame,
    shares_outstanding: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the interest debt per share, a valuation ratio that measures the
    combined interest expense and debt burden of a company per outstanding share
    of its stock.

    The formula is as follows:

        Interest Debt per Share = (Interest Expense + Total Debt) / Shares Outstanding

    For more information about the method, see the following source:
    https://site.financialmodelingprep.com/developer/docs/formula

    Args:
        interest_expense (float or pd.Series): Interest expense of the company.
        total_debt (float or pd.Series): Total debt of the company.
        shares_outstanding (float or pd.Series): Total number of outstanding shares of the company.

    Returns:
        float | pd.Series: The interest debt per share value.
    """
    return (interest_expense + total_debt) / shares_outstanding


def get_capex_per_share(
    capital_expenditures: pd.Series | pd.DataFrame,
    shares_outstanding: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the capex per share, a valuation ratio that measures the amount of
    capital expenditures made per outstanding share of a company's stock.

    Args:
        capital_expenditures (float or pd.Series): Total capital expenditures made by the company.
        shares_outstanding (float or pd.Series): Total number of outstanding shares of the company.

    Returns:
        float | pd.Series: The capex per share value.
    """
    return capital_expenditures / shares_outstanding


def get_dividend_yield(
    dividends: pd.Series | pd.DataFrame, stock_price: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the dividend yield ratio, a valuation ratio that measures the amount of
    dividends distributed per share of stock relative to the stock's price.

    Args:
        dividends (float or pd.Series): Dividend paid out by the company.
        stock_price (float or pd.Series): Stock price of the company.

    Returns:
        float | pd.Series: The dividend yield percentage value.
    """
    return dividends / stock_price


def get_weighted_dividend_yield(
    dividends_paid: pd.Series | pd.DataFrame,
    shares_outstanding: pd.Series | pd.DataFrame,
    stock_price: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the weighted dividend yield ratio, a valuation ratio that measures the amount of
    dividends distributed per share of stock relative to the stock's price.

    This dividend yield differs from the dividend yield ratio in that it takes into account the
    (diluted) weighted average shares and actual dividends paid as found in the cash flow statement.

    Args:
        dividends_paid (float or pd.Series): Dividends Paid as reported in the Cash Flow Statement.
        shares_outstanding (float or pd.Series): Total number of outstanding shares of the company.
        stock_price (float or pd.Series): Stock price of the company.

    Returns:
        float | pd.Series: The weighted dividend yield percentage value.
    """
    return (dividends_paid / shares_outstanding) / stock_price


def get_price_to_cash_flow_ratio(
    market_cap: pd.Series | pd.DataFrame, operations_cash_flow: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the price to cash flow ratio, a valuation ratio that compares a company's market
    price to its cash flow per share.

    Args:
        market_cap (float or pd.Series): Market capitalization of the company.
        operations_cash_flow (float or pd.Series): Operations cash flow of the company.

    Returns:
        float | pd.Series: The price to cash flow ratio value.
    """
    return market_cap / operations_cash_flow


def get_price_to_free_cash_flow_ratio(
    market_cap: pd.Series | pd.DataFrame, free_cash_flow: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the price to free cash flow ratio, a valuation ratio that compares a company's market
    price to its free cash flow per share.

    Args:
        market_cap (float or pd.Series): Market capitalization of the company.
        free_cash_flow (float or pd.Series): Free cash flow of the company

    Returns:
        float | pd.Series: The price to free cash flow ratio value.
    """
    return market_cap / free_cash_flow


def get_price_to_sales_ratio(
    market_cap: pd.Series | pd.DataFrame, total_revenue: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the price to sales ratio (P/S), a valuation ratio that compares a
    company's market capitalization to its total revenue.

    The price to sales ratio is particularly useful for valuing companies that are not
    yet profitable (and therefore have no meaningful P/E ratio), since revenue is
    typically positive even when earnings are not, and is less susceptible to
    accounting distortions than earnings-based multiples. It is, however, less
    informative than earnings- or cash-flow-based multiples for mature, profitable
    companies since it ignores profitability and cost structure entirely.

    The formula is as follows:

        Price to Sales Ratio = Market Capitalization / Total Revenue

    This is equivalent to Stock Price / Revenue per Share.

    Also known as: P/S ratio, sales multiple.

    Args:
        market_cap (float or pd.Series): Market capitalization of the company.
        total_revenue (float or pd.Series): Total revenue of the company.

    Returns:
        float | pd.Series: The price to sales ratio value.
    """
    return market_cap / total_revenue


def get_market_cap(
    share_price: pd.Series | pd.DataFrame,
    total_shares_outstanding: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the market capitalization of the company.

    Notes: All the inputs must be in the same currency and unit for accurate calculations.

    Args:
        share_price (float | pd.Series): The share price of the company.
        total_shares_outstanding (float | pd.Series): The total number of shares outstanding of the company.

    Returns:
        float | pd.Series: The market capitalization of the company.
    """
    return share_price * total_shares_outstanding


def get_basket_market_cap(
    market_cap: pd.DataFrame, share: bool = False
) -> pd.DataFrame:
    """
    Adds the total market capitalization of a basket of companies, or expresses each company
    as its share of that total.

    The total of a date is missing when a company of the basket has no market cap on that
    date, as the sum of the others would understate the basket.

    Notes: All the market caps must be in the same currency.

    Args:
        market_cap (pd.DataFrame): The market capitalization with the dates (or periods) as
            index and the companies as columns.
        share (bool, optional): Whether to return each company's share of the total (the
            Total is then 1) instead of the market caps. Defaults to False.

    Returns:
        pd.DataFrame: The market caps (or shares) with a "Total" column added.
    """
    total = market_cap.sum(axis=1, skipna=False)

    basket = market_cap.div(total, axis=0) if share else market_cap.copy()
    basket["Total"] = total.div(total) if share else total

    return basket


def get_market_cap_to_gdp(
    market_cap: pd.Series,
    gross_domestic_product: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates the market capitalization to Gross Domestic Product (GDP) ratio, the
    Buffett indicator: a market value divided by the annual output of an economy.

    Each date (or period) is divided by the GDP of the calendar year before the year it
    falls in, the last complete year, so that no GDP is used before its year has ended.

    Notes: The market cap and GDP must be in the same currency and unit.

    Args:
        market_cap (pd.Series): The market capitalization with dates (DatetimeIndex) or
            periods (PeriodIndex) as index.
        gross_domestic_product (pd.DataFrame): The annual GDP with years (PeriodIndex) as
            index and the countries as columns.

    Returns:
        pd.DataFrame: The ratio with the dates (or periods) as index and the countries as
        columns.
    """
    previous_year_gdp = gross_domestic_product.set_axis(
        gross_domestic_product.index.year
    ).reindex(market_cap.index.year - 1)
    previous_year_gdp.index = market_cap.index

    return previous_year_gdp.rdiv(market_cap, axis=0)


def get_enterprise_value(
    market_cap: pd.Series | pd.DataFrame,
    total_debt: pd.Series | pd.DataFrame,
    minority_interest: pd.Series | pd.DataFrame,
    preferred_equity: pd.Series | pd.DataFrame,
    cash_and_cash_equivalents: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Enterprise Value (EV) of a company. The Enterprise Value (EV)
    is a measure of a company's total value, often used as a more comprehensive
    alternative to market capitalization. It is calculated as the sum of a company's
    market capitalization, outstanding debt, minority interest, and
    preferred equity, minus the cash and cash equivalents.

    Notes: All the inputs must be in the same currency and unit for accurate calculations.

    Args:
        market_cap (float or pd.Series): The market capitalization of the company.
        total_debt (float or pd.Series): The total debt of the company.
        minority_interest (float or pd.Series): The value of minority interest in the company.
        preferred_equity (float or pd.Series): The value of the preferred equity in the company.
        cash_and_cash_equivalents (float or pd.Series): The value of cash and cash equivalents of the
            company, or of cash and short-term investments to also deduct marketable securities.

    Returns:
        float | pd.Series: The Enterprise Value (EV) of the company.
    """
    return (
        market_cap
        + total_debt
        + minority_interest
        + preferred_equity
        - cash_and_cash_equivalents
    )


def get_ev_to_sales_ratio(
    enterprise_value: pd.Series | pd.DataFrame, total_revenue: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the EV to sales ratio, a valuation ratio that compares a company's enterprise value
    (EV) to its total revenue.

    Args:
        enterprise_value (float or pd.Series): Enterprise value of the company.
        total_revenue (float or pd.Series): Total revenue of the company.

    Returns:
        float | pd.Series: The EV to sales ratio value.
    """
    return enterprise_value / total_revenue


def get_ev_to_ebitda_ratio(
    enterprise_value: pd.Series | pd.DataFrame,
    operating_income: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the enterprise value over EBITDA ratio, which is a valuation ratio
    that measures a company's total value (including debt and equity) relative to its
    EBITDA.

    Args:
        enterprise_value (float or pd.Series): The total value of a company (market capitalization
            plus debt minus cash and cash equivalents).
        operating_income (float or pd.Series): The operating income of the company.
        depreciation_and_amortization (float or pd.Series): The depreciation and amortization of the company.

    Returns:
        float | pd.Series: The enterprise value over EBITDA ratio.
    """
    return enterprise_value / (operating_income + depreciation_and_amortization)


def get_ev_to_operating_cashflow_ratio(
    enterprise_value: pd.Series | pd.DataFrame,
    operating_cashflow: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the enterprise value over operating cash flow ratio, which is a valuation ratio
    that measures a company's total value (including debt and equity) relative to its
    operating cash flow.

    Args:
        enterprise_value (float or pd.Series): The total value of a company (market capitalization
            plus debt minus cash and cash equivalents).
        operating_cashflow (float or pd.Series): The cash generated by the company's operations.

    Returns:
        float | pd.Series: The enterprise value over operating cash flow ratio.
    """
    return enterprise_value / operating_cashflow


def get_earnings_yield(
    earnings_per_share: pd.Series | pd.DataFrame,
    market_price_per_share: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the earnings yield ratio, which measures the earnings per share relative
    to the market price per share.

    Args:
        earnings_per_share (float or pd.Series): Earnings per share of the company.
        market_price_per_share (float or pd.Series): Market price per share of the company.

    Returns:
        float | pd.Series: The earnings yield ratio.
    """
    return earnings_per_share / market_price_per_share


def get_dividend_payout_ratio(
    dividends: pd.Series | pd.DataFrame, net_income: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculates the dividend payout ratio, which measures the proportion of net income paid out as
    dividends to shareholders.

    Args:
        dividends (float or pd.Series): Dividends paid by the company.
        net_income (float or pd.Series): Net income of the company.

    Returns:
        float | pd.Series: The dividend payout ratio.
    """
    return abs(dividends) / net_income


def get_reinvestment_ratio(
    dividend_payout_ratio: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the reinvestment ratio, which measures the proportion of net income
    retained by the company to reinvest in the business.

    Args:
        dividend_payout_ratio (float or pd.Series): The dividend payout ratio.

    Returns:
        float | pd.Series: The reinvestment ratio.
    """
    return 1 - dividend_payout_ratio


def get_tangible_asset_value(
    total_assets: pd.Series | pd.DataFrame,
    total_liabilities: pd.Series | pd.DataFrame,
    goodwill_and_intangible_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the tangible asset value, the book value of a company's equity after
    deducting goodwill and other intangible assets.

    The formula is as follows:

        Tangible Asset Value = Total Assets - Total Liabilities - Goodwill and Intangible Assets

    Args:
        total_assets (float or pd.Series): The total assets of the company.
        total_liabilities (float or pd.Series): The total liabilities of the company.
        goodwill_and_intangible_assets (float or pd.Series): The goodwill and other intangible
            assets of the company.

    Returns:
        float | pd.Series: The tangible asset value.
    """
    return total_assets - total_liabilities - goodwill_and_intangible_assets


def get_net_current_asset_value(
    total_current_assets: pd.Series | pd.DataFrame,
    total_liabilities: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the net current asset value (Benjamin Graham's NCAV), which is the total value
    of a company's current assets minus its total liabilities (not just its current liabilities).

    Args:
        total_current_assets (float or pd.Series): The current assets of the company.
        total_liabilities (float or pd.Series): The total liabilities of the company.

    Returns:
        float | pd.Series: The net current asset value.
    """
    return total_current_assets - total_liabilities


def get_ev_to_ebit(
    enterprise_value: pd.Series | pd.DataFrame,
    earnings_before_interest_and_taxes: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the enterprise value multiplier, a financial ratio that measures the total value of a
    company's operations (including debt and equity) relative to its earnings before interest and taxes.

    Args:
        enterprise_value (float or pd.Series): The total value of a company's operations.
        earnings_before_interest_and_taxes (float or pd.Series): The company's earnings before interest and taxes.

    Returns:
        float | pd.Series: The enterprise value multiplier value.
    """
    return enterprise_value / earnings_before_interest_and_taxes


def get_ev_to_free_cash_flow_ratio(
    enterprise_value: pd.Series | pd.DataFrame, free_cash_flow: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the EV to free cash flow ratio, a valuation ratio that compares a company's
    enterprise value (EV) to its free cash flow.

    Args:
        enterprise_value (float or pd.Series): Enterprise value of the company.
        free_cash_flow (float or pd.Series): Free cash flow of the company.

    Returns:
        float | pd.Series: The EV to free cash flow ratio value.
    """
    return enterprise_value / free_cash_flow


def get_buyback_yield(
    repurchase_of_common_stock: pd.Series | pd.DataFrame,
    issuance_of_common_stock: pd.Series | pd.DataFrame,
    market_capitalization: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the buyback yield, a valuation ratio that measures the net amount of
    common stock repurchased (net of new shares issued) relative to the company's
    market capitalization.

    Repurchases are typically reported as a cash outflow (negative) and issuances as a
    cash inflow (positive) on the cash flow statement, so netting and negating the two
    yields a positive value when the company is a net repurchaser of its own stock and
    a negative value when it is a net issuer (i.e. diluting shareholders).

    Args:
        repurchase_of_common_stock (float or pd.Series): Common stock repurchased by the company,
            as reported in the Cash Flow Statement.
        issuance_of_common_stock (float or pd.Series): Common stock issued by the company,
            as reported in the Cash Flow Statement.
        market_capitalization (float or pd.Series): Market capitalization of the company.

    Returns:
        float | pd.Series: The buyback yield value.
    """
    return -(repurchase_of_common_stock + issuance_of_common_stock) / (
        market_capitalization
    )


def get_shareholder_yield(
    dividend_yield: pd.Series | pd.DataFrame, buyback_yield: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the total shareholder yield, a valuation ratio that combines the dividend
    yield and the buyback yield to measure the total cash returned to shareholders
    relative to the company's market capitalization.

    Args:
        dividend_yield (float or pd.Series): Dividend yield of the company.
        buyback_yield (float or pd.Series): Buyback yield of the company.

    Returns:
        float | pd.Series: The total shareholder yield value.
    """
    return dividend_yield + buyback_yield


def get_net_share_issuance(
    shares: pd.Series | pd.DataFrame, beginning_shares: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the net share issuance of Pontiff and Woodgate (2008): the log change in
    the split-adjusted number of shares. It is positive for a net issuer of shares and
    negative for a company that buys back more shares than it issues.

    The formula is as follows:

        Net Share Issuance = ln(Shares / Beginning Shares)

    The shares must be adjusted for splits, so that only issuance and repurchases move
    them. A number of shares of zero or less at either end gives NaN.

    Also known as: share issuance, ISSUE, change in shares outstanding.

    Args:
        shares (float or pd.Series): Split-adjusted number of shares at the end of the window.
        beginning_shares (float or pd.Series): Split-adjusted number of shares at the
            start of the window.

    Returns:
        float | pd.Series: The net share issuance value.
    """
    return np.log(
        shares.where(shares > 0) / beginning_shares.where(beginning_shares > 0)
    )


def get_composite_equity_issuance(
    market_cap: pd.Series | pd.DataFrame,
    beginning_market_cap: pd.Series | pd.DataFrame,
    total_return_price: pd.Series | pd.DataFrame,
    beginning_total_return_price: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the composite equity issuance of Daniel and Titman (2006): the part of
    the growth in market capitalization that the stock return does not explain. It
    captures every way equity is issued or retired (seasoned offerings, employee
    stock, stock-financed acquisitions, buybacks and dividends), and companies that
    issue predict lower returns.

    The formula is as follows:

        Composite Equity Issuance = ln(Market Cap / Beginning Market Cap)
            - ln(Total Return Price / Beginning Total Return Price)

    The second term is the log total return over the window (dividends reinvested), so
    a company that pays dividends without issuing shares has a negative value. A value
    of zero or less in either ratio gives NaN.

    Also known as: CEI, iota, composite share issuance.

    Args:
        market_cap (float or pd.Series): Market capitalization at the end of the window.
        beginning_market_cap (float or pd.Series): Market capitalization at the start of
            the window.
        total_return_price (float or pd.Series): Price adjusted for dividends and splits
            at the end of the window.
        beginning_total_return_price (float or pd.Series): Price adjusted for dividends
            and splits at the start of the window.

    Returns:
        float | pd.Series: The composite equity issuance value.
    """
    market_cap_growth = market_cap / beginning_market_cap
    total_return = total_return_price / beginning_total_return_price

    return np.log(market_cap_growth.where(market_cap_growth > 0)) - np.log(
        total_return.where(total_return > 0)
    )


def get_net_external_financing(
    net_stock_issued: pd.Series | pd.DataFrame,
    dividends_paid: pd.Series | pd.DataFrame,
    net_debt_issued: pd.Series | pd.DataFrame,
    average_total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the net external financing of Bradshaw, Richardson and Sloan (2006): the
    cash a company raises from (or returns to) shareholders and lenders, relative to
    its average total assets. Companies that raise the most external financing earn
    the lowest returns afterwards.

    The formula is as follows:

        Net External Financing = (Net Equity Financing + Net Debt Financing)
            / Average Total Assets

        Net Equity Financing = Net Stock Issued + Dividends Paid
        Net Debt Financing = Net Debt Issued

    The flows carry the signs of the cash flow statement: stock repurchases and
    dividends are negative outflows, so the net equity financing is the stock issued
    minus the stock repurchased minus the dividends paid. Average total assets of zero
    or less give NaN.

    Also known as: XFIN, ΔXFIN, net financing.

    Args:
        net_stock_issued (float or pd.Series): Common and preferred stock issued minus
            repurchased, as reported in the Cash Flow Statement.
        dividends_paid (float or pd.Series): Common and preferred dividends paid, as
            reported (negative) in the Cash Flow Statement.
        net_debt_issued (float or pd.Series): Long and short term debt issued minus
            repaid, as reported in the Cash Flow Statement.
        average_total_assets (float or pd.Series): Average of the total assets at the
            start and the end of the window.

    Returns:
        float | pd.Series: The net external financing value.
    """
    return (net_stock_issued + dividends_paid + net_debt_issued) / (
        average_total_assets.where(average_total_assets > 0)
    )


def get_sbc_adjusted_free_cash_flow(
    free_cash_flow: pd.Series | pd.DataFrame,
    stock_based_compensation: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the stock-based compensation (SBC) adjusted free cash flow, which
    deducts non-cash SBC expenses from free cash flow to give a more conservative
    view of the cash actually available to shareholders.

    Free cash flow already excludes SBC as a cash expense (it is added back in the
    cash flow from operations), which can overstate the cash available to
    shareholders. Subtracting SBC treats it as if it were a real cash cost, which is
    a common quality-of-earnings adjustment, especially for companies that rely
    heavily on equity compensation.

    Args:
        free_cash_flow (float or pd.Series): Free cash flow of the company.
        stock_based_compensation (float or pd.Series): Stock-based compensation of the company.

    Returns:
        float | pd.Series: The SBC-adjusted free cash flow value.
    """
    return free_cash_flow - stock_based_compensation


def get_free_cash_flow_payout_ratio(
    dividends: pd.Series | pd.DataFrame, free_cash_flow: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the free cash flow payout ratio, the share of free cash flow paid out as
    dividends.

    The formula is as follows:

        Free Cash Flow Payout Ratio = |Dividends Paid| / Free Cash Flow

    Dividends are reported as a negative cash outflow, so their magnitude is used. With
    a free cash flow of zero or less the ratio has no meaning, so those periods are NaN.

    Also known as: FCF payout ratio, dividends to free cash flow, cash dividend payout ratio.

    Args:
        dividends (float or pd.Series): Dividends paid by the company.
        free_cash_flow (float or pd.Series): Free cash flow of the company.

    Returns:
        float | pd.Series: The free cash flow payout ratio, NaN where the free cash flow is
            zero or negative.
    """
    return abs(dividends) / free_cash_flow.where(free_cash_flow > 0)


def get_ev_to_invested_capital_ratio(
    enterprise_value: pd.Series | pd.DataFrame,
    total_equity: pd.Series | pd.DataFrame,
    total_debt: pd.Series | pd.DataFrame,
    cash_and_short_term_investments: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the enterprise value to invested capital ratio, the market value of the
    operations against the capital invested in them.

    The formula is as follows (Koller, Goedhart and Wessels):

        Invested Capital = Total Equity + Total Debt - Cash and Short Term Investments

        EV to Invested Capital = Enterprise Value / Invested Capital

    The invested capital is taken from the financing side, with the same items as the
    enterprise value: the equity includes minority interest and preferred stock, and the
    cash and short-term investments are deducted from both. An invested capital of zero
    or less gives NaN.

    Also known as: EV/IC, enterprise value to invested capital, market-to-book of the firm.

    Args:
        enterprise_value (float or pd.Series): Enterprise value of the company.
        total_equity (float or pd.Series): Total equity, including minority interest.
        total_debt (float or pd.Series): Total debt of the company.
        cash_and_short_term_investments (float or pd.Series): Cash, cash equivalents and
            short-term investments.

    Returns:
        float | pd.Series: The EV to invested capital value, NaN where the invested capital
            is zero or negative.
    """
    invested_capital = total_equity + total_debt - cash_and_short_term_investments

    return enterprise_value / invested_capital.where(invested_capital > 0)


def get_price_to_tangible_book_ratio(
    market_cap: pd.Series | pd.DataFrame,
    total_shareholder_equity: pd.Series | pd.DataFrame,
    preferred_stock: pd.Series | pd.DataFrame,
    goodwill_and_intangible_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the price to tangible book ratio, the market value of the common equity
    against its book value without goodwill and other intangible assets.

    The formula is as follows (CFA Institute):

        Tangible Book Value = Total Shareholder Equity - Preferred Stock - Goodwill and
            Intangible Assets

        Price to Tangible Book = Market Capitalization / Tangible Book Value

    A tangible book value of zero or less gives NaN, as the ratio has no meaning then.

    Also known as: P/TBV, price to tangible book value, price to tangible equity.

    Args:
        market_cap (float or pd.Series): Market capitalization of the company.
        total_shareholder_equity (float or pd.Series): Shareholders' equity, without
            minority interest.
        preferred_stock (float or pd.Series): Preferred stock of the company.
        goodwill_and_intangible_assets (float or pd.Series): Goodwill and intangible assets.

    Returns:
        float | pd.Series: The price to tangible book value, NaN where the tangible book
            value is zero or negative.
    """
    tangible_book_value = (
        total_shareholder_equity - preferred_stock - goodwill_and_intangible_assets
    )

    return market_cap / tangible_book_value.where(tangible_book_value > 0)
