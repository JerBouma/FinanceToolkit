"""Valuation Module"""

__docformat__ = "google"

import pandas as pd


def get_earnings_per_share(
    net_income: pd.Series,
    preferred_dividends: pd.Series,
    average_outstanding_shares: pd.Series,
) -> pd.Series:
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
    total_revenue: pd.Series, shares_outstanding: pd.Series
) -> pd.Series:
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
    stock_price: pd.Series, earnings_per_share: pd.Series
) -> pd.Series:
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
    price_earnings: pd.Series, growth_rate: pd.Series
) -> pd.Series:
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


def get_book_value_per_share(
    total_shareholder_equity: pd.Series,
    preferred_equity: pd.Series,
    common_shares_outstanding: pd.Series,
) -> pd.Series:
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
    price_per_share: pd.Series, book_value_per_share: pd.Series
) -> pd.Series:
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
    interest_expense: pd.Series,
    total_debt: pd.Series,
    shares_outstanding: pd.Series,
) -> pd.Series:
    """
    Calculate the interest debt per share, a valuation ratio that measures the
    amount of interest expense incurred per outstanding share of a company's stock.

    Args:
        interest_expense (float or pd.Series): Interest expense of the company.
        total_debt (float or pd.Series): Total debt of the company.
        shares_outstanding (float or pd.Series): Total number of outstanding shares of the company.

    Returns:
        float | pd.Series: The interest debt per share value.
    """
    return (interest_expense / total_debt) * shares_outstanding


def get_capex_per_share(
    capital_expenditures: pd.Series, shares_outstanding: pd.Series
) -> pd.Series:
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
    dividends: pd.Series,
    stock_price: pd.Series,
) -> pd.Series:
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
    dividends_paid: pd.Series,
    shares_outstanding: pd.Series,
    stock_price: pd.Series,
) -> pd.Series:
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
    market_cap: pd.Series, operations_cash_flow: pd.Series
) -> pd.Series:
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
    market_cap: pd.Series, free_cash_flow: pd.Series
) -> pd.Series:
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


def get_market_cap(
    share_price: pd.Series,
    total_shares_outstanding: pd.Series,
) -> pd.Series:
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


def get_enterprise_value(
    market_cap: pd.Series,
    total_debt: pd.Series,
    minority_interest: pd.Series,
    preferred_equity: pd.Series,
    cash_and_cash_equivalents: pd.Series,
) -> pd.Series:
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
        cash_and_cash_equivalents (float or pd.Series): The value of cash and cash equivalents of the company.

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
    enterprise_value: pd.Series, total_revenue: pd.Series
) -> pd.Series:
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
    enterprise_value: pd.Series,
    operating_income: pd.Series,
    depreciation_and_amortization: pd.Series,
) -> pd.Series:
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
    enterprise_value: pd.Series, operating_cashflow: pd.Series
) -> pd.Series:
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
    earnings_per_share: pd.Series, market_price_per_share: pd.Series
) -> pd.Series:
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


def get_dividend_payout_ratio(dividends: pd.Series, net_income: pd.Series) -> pd.Series:
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


def get_reinvestment_ratio(dividend_payout_ratio: pd.Series) -> pd.Series:
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
    total_assets: pd.Series,
    total_liabilities: pd.Series,
    goodwill: pd.Series,
) -> pd.Series:
    """
    Calculate the tangible asset value, which represents the total value of a company's assets
    that can be used to generate revenue.

    Args:
        total_assets (float or pd.Series): The total assets of the company.
        total_liabilities (float or pd.Series): The total liabilities of the company.
        goodwill (float or pd.Series): The goodwill of the company.

    Returns:
        float | pd.Series: The tangible asset value.
    """
    return total_assets - total_liabilities - goodwill


def get_net_current_asset_value(
    total_current_assets: pd.Series,
    total_current_liabilities: pd.Series,
) -> pd.Series:
    """
    Calculate the net current asset value, which is the total value of a company's current assets
    minus its current liabilities.

    Args:
        total_current_assets (float or pd.Series): The current assets of the company.
        total_current_liabilities (float or pd.Series): The current liabilities of the company.

    Returns:
        float | pd.Series: The net current asset value.
    """
    return total_current_assets - total_current_liabilities


def get_ev_to_ebit(
    enterprise_value: pd.Series,
    earnings_before_interest_and_taxes: pd.Series,
) -> pd.Series:
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
