"""Overvie Module"""
__docformat__ = "numpy"

import pandas as pd

from financialtoolkit.ratios import (
    efficiency as _efficiency,
    liquidity as _liquidity,
    profitability as _profitability,
    solvency as _solvency,
    valuation as _valuation,
)


def get_efficiency_ratios(
    tickers: str | list[str],
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
):
    """
    Retrieves financial statements (balance, income, or cash flow statements) for one or multiple companies,
    and returns a DataFrame containing the data.

    Args:
        tickers (List[str]): List of company tickers.
        balance_sheet (pd.DataFrame): A DataFrame containing the balance sheet data for the companies.
        income_statement (pd.DataFrame): A DataFrame containing the income statement data for the companies.

    Returns:
        pd.DataFrame: A DataFrame containing the _efficiency ratios. If only one ticker is provided, the
                      returned DataFrame will have a single column containing the data for that ticker. If multiple
                      tickers are provided, the returned DataFrame will have multiple columns, one for each ticker,
                      with the ticker symbol as the column name.
    """
    efficiency_ratios_dict: dict = {}

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if len(ticker_list) == 1:
        balance_sheet = pd.concat({ticker_list[0]: balance_sheet}, axis=1)
        income_statement = pd.concat({ticker_list[0]: income_statement}, axis=1)

    for ticker in ticker_list:
        bst = balance_sheet[ticker]
        ist = income_statement[ticker]

        efficiency_ratios: pd.DataFrame = pd.DataFrame()

        efficiency_ratios[
            "Asset Turnover Ratio"
        ] = _efficiency.get_asset_turnover_ratio(
            ist.loc["Revenue"], bst.loc["Total Assets"]
        )
        efficiency_ratios[
            "Inventory Turnover Ratio"
        ] = _efficiency.get_inventory_turnover_ratio(
            ist.loc["Cost of Goods Sold"], bst.loc["Inventory"].mean()
        )
        efficiency_ratios[
            "Payables Turnover Ratio"
        ] = _efficiency.get_payables_turnover_ratio(
            ist.loc["Cost of Goods Sold"], bst.loc["Accounts Payable"].mean()
        )
        efficiency_ratios[
            "Days of Inventory Outstanding (DIO)"
        ] = _efficiency.get_days_of_inventory_outstanding(
            bst.loc["Inventory"].mean(), ist.loc["Cost of Goods Sold"]
        )
        efficiency_ratios[
            "Days of Sales Outstanding (DSO)"
        ] = _efficiency.get_days_of_sales_outstanding(
            bst.loc["Accounts Receivable"], ist.loc["Revenue"]
        )
        efficiency_ratios["Operating Cycle (CC)"] = _efficiency.get_operating_cycle(
            efficiency_ratios["Days of Inventory Outstanding (DIO)"],
            efficiency_ratios["Days of Sales Outstanding (DSO)"],
        )
        efficiency_ratios[
            "Days of Payables Outstanding (DPO)"
        ] = _efficiency.get_days_payables_outstanding(
            efficiency_ratios["Payables Turnover Ratio"]
        )
        efficiency_ratios[
            "Cash Conversion Cycle (CCC)"
        ] = _efficiency.get_cash_conversion_cycle(
            efficiency_ratios["Days of Inventory Outstanding (DIO)"],
            efficiency_ratios["Days of Sales Outstanding (DSO)"],
            efficiency_ratios["Days of Payables Outstanding (DPO)"],
        )

        efficiency_ratios_dict[ticker] = efficiency_ratios

    efficiency_ratios_total = pd.concat(efficiency_ratios_dict, axis=0)

    if len(ticker_list) == 1:
        efficiency_ratios_total = efficiency_ratios_total[ticker_list[0]]

    return efficiency_ratios_total


def get_liquidity_ratios(
    tickers: str | list[str],
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
    cash_flow_statement: pd.DataFrame,
):
    """
    Calculates various liquidity ratios for each of the companies. It returns a DataFrame
    containing the calculated ratios.

    The function calculates the following liquidity ratios for each company:
        Current Ratio
        Quick Ratio
        Cash Ratio
        Working Capital
        Working Capital Ratio
        Operating Cash Flow Ratio
        Operating Cash Flow to Sales Ratio
        Short Term Coverage Ratio

    Args:
        tickers (List[str]): List of company tickers.
        balance_sheet (pd.DataFrame): A DataFrame containing the balance sheet data for the companies.
        income_statement (pd.DataFrame): A DataFrame containing the income statement data for the companies.
        cash_flow_statement (pd.DataFrame): A DataFrame containing the cash flow statement data for the companies.

    Returns:
        pd.DataFrame: A DataFrame containing the _liquidity ratios for the companies. If
                      only one ticker is provided, the returned DataFrame will have a single column
                      containing the data for that ticker. If multiple tickers are provided, the returned DataFrame
                      will have multiple columns, one for each ticker, with the ticker symbol as the column name.
    """
    liquidity_ratios_dict: dict = {}

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if len(ticker_list) == 1:
        balance_sheet = pd.concat({ticker_list[0]: balance_sheet}, axis=1)
        income_statement = pd.concat({ticker_list[0]: income_statement}, axis=1)
        cash_flow_statement = pd.concat({ticker_list[0]: cash_flow_statement}, axis=1)

    for ticker in ticker_list:
        bst = balance_sheet[ticker]
        ist = income_statement[ticker]
        cf = cash_flow_statement[ticker]

        liquidity_ratios: pd.DataFrame = pd.DataFrame()

        liquidity_ratios["Current Ratio"] = _liquidity.get_current_ratio(
            bst.loc["Current Assets"], bst.loc["Total Current Liabilities"]
        )
        liquidity_ratios["Quick Ratio"] = _liquidity.get_quick_ratio(
            bst.loc["Current Assets"],
            bst.loc["Inventory"],
            bst.loc["Total Current Liabilities"],
        )
        liquidity_ratios["Cash Ratio"] = _liquidity.get_cash_ratio(
            bst.loc["Cash and Cash Equivalents"], bst.loc["Total Current Liabilities"]
        )
        liquidity_ratios["Working Capital"] = _liquidity.get_working_capital(
            bst.loc["Current Assets"], bst.loc["Total Current Liabilities"]
        )
        liquidity_ratios[
            "Working Capital Ratio"
        ] = _liquidity.get_working_capital_ratio(
            bst.loc["Current Assets"], bst.loc["Total Current Liabilities"]
        )
        liquidity_ratios[
            "Operating Cash Flow Ratio"
        ] = _liquidity.get_operating_cash_flow_ratio(
            cf.loc["Operating Cash Flow"], bst.loc["Total Current Liabilities"]
        )
        liquidity_ratios[
            "Operating Cash Flow to Sales Ratio"
        ] = _liquidity.get_operating_cash_flow_sales_ratio(
            cf.loc["Operating Cash Flow"], ist.loc["Revenue"]
        )
        liquidity_ratios[
            "Short Term Coverage Ratio"
        ] = _liquidity.get_short_term_coverage_ratio(
            cf.loc["Operating Cash Flow"],
            bst.loc["Accounts Receivable"],
            bst.loc["Inventory"],
            bst.loc["Accounts Payable"],
        )

        liquidity_ratios_dict[ticker] = liquidity_ratios

    liquidity_ratios_total = pd.concat(liquidity_ratios_dict, axis=0)

    if len(ticker_list) == 1:
        liquidity_ratios_total = liquidity_ratios_total[ticker_list[0]]

    return liquidity_ratios_total


def get_profitability_ratios(
    tickers: str | list[str],
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
    cash_flow_statement: pd.DataFrame,
):
    """
    Calculates various profitability ratios for each of the companies.
    It returns a DataFrame containing the calculated ratios.

    The function calculates the following profitability ratios for each company:
        Gross Margin
        Operating Margin
        Net Profit Margin
        Pre Tax Profit
        Effective Tax Rate
        Return on Assets (ROA)
        Return on Equity (ROE)
        Return on Invested Capital (ROIC)
        Return on Tangible Assets (ROTA)
        Return on Capital Employed (ROCE)
        Income Quality Ratio

    Args:
        tickers (List[str]): List of company tickers.
        balance_sheet (pd.DataFrame): A DataFrame containing the balance sheet data for the companies.
        income_statement (pd.DataFrame): A DataFrame containing the income statement data for the companies.
        cash_flow_statement (pd.DataFrame): A DataFrame containing the cash flow statement data for the companies.

    Returns:
        pd.DataFrame: A DataFrame containing the _liquidity ratios for the companies. If
                      only one ticker is provided, the returned DataFrame will have a single column
                      containing the data for that ticker. If multiple tickers are provided, the returned DataFrame
                      will have multiple columns, one for each ticker, with the ticker symbol as the column name.
    """
    profitability_ratios_dict: dict = {}

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if len(ticker_list) == 1:
        balance_sheet = pd.concat({ticker_list[0]: balance_sheet}, axis=1)
        income_statement = pd.concat({ticker_list[0]: income_statement}, axis=1)
        cash_flow_statement = pd.concat({ticker_list[0]: cash_flow_statement}, axis=1)

    for ticker in ticker_list:
        bst = balance_sheet[ticker]
        ist = income_statement[ticker]
        cf = cash_flow_statement[ticker]

        profitability_ratios: pd.DataFrame = pd.DataFrame()

        profitability_ratios["Gross Margin"] = _profitability.get_gross_margin(
            ist.loc["Revenue"], ist.loc["Cost of Goods Sold"]
        )

        profitability_ratios["Operating Margin"] = _profitability.get_operating_margin(
            ist.loc["Operating Income"], ist.loc["Revenue"]
        )

        profitability_ratios[
            "Net Profit Margin"
        ] = _profitability.get_net_profit_margin(
            ist.loc["Net Income"], ist.loc["Revenue"]
        )

        profitability_ratios[
            "Income Before Tax Profit Margin"
        ] = _profitability.get_income_before_tax_profit_margin(
            ist.loc["Income Before Tax"], ist.loc["Revenue"]
        )

        profitability_ratios[
            "Effective Tax Rate"
        ] = _profitability.get_effective_tax_rate(
            ist.loc["Income Tax Expense"], ist.loc["Income Before Tax"]
        )

        profitability_ratios[
            "Return on Assets (ROA)"
        ] = _profitability.get_return_on_assets(
            ist.loc["Net Income"], bst.loc["Total Assets"]
        )

        profitability_ratios[
            "Return on Equity (ROE)"
        ] = _profitability.get_return_on_equity(
            ist.loc["Net Income"], bst.loc["Total Equity"]
        )

        profitability_ratios[
            "Return on Invested Capital (ROIC)"
        ] = _profitability.get_return_on_invested_capital(
            ist.loc["Net Income"],
            cf.loc["Dividends Paid"],
            bst.loc["Total Equity"] + bst.loc["Total Debt"],
        )

        profitability_ratios[
            "Return on Tangible Assets (ROTA)"
        ] = _profitability.get_return_on_tangible_assets(
            ist.loc["Net Income"],
            bst.loc["Total Assets"],
            bst.loc["Intangible Assets"],
            bst.loc["Total Liabilities"],
        )

        profitability_ratios[
            "Return on Capital Employed (ROCE)"
        ] = _profitability.get_return_on_capital_employed(
            ist.loc["Net Income"],
            ist.loc["Interest Expense"],
            ist.loc["Income Tax Expense"],
            bst.loc["Total Assets"],
            bst.loc["Total Current Liabilities"],
        )

        profitability_ratios[
            "Income Quality Ratio"
        ] = _profitability.get_income_quality_ratio(
            cf.loc["Cash Flow from Operations"], ist.loc["Net Income"]
        )

        profitability_ratios_dict[ticker] = profitability_ratios

    profitability_ratios_total = pd.concat(profitability_ratios_dict, axis=0)

    if len(ticker_list) == 1:
        profitability_ratios_total = profitability_ratios_total[ticker_list[0]]

    return profitability_ratios_total


def get_solvency_ratios(
    tickers: str | list[str],
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
):
    """
    Calculates various solvency ratios for each of the companies. It returns a
    DataFrame containing the calculated ratios.

    The function calculates the following solvency ratios for each company:
        debt_to_assets_ratio
        debt_to_equity_ratio
        interest_coverage_ratio
        debt_service_coverage_ratio
        financial_leverage

    Args:
        tickers (List[str]): List of company tickers.
        balance_sheet (pd.DataFrame): A DataFrame containing the balance sheet data for the companies.
        income_statement (pd.DataFrame): A DataFrame containing the income statement data for the companies.
        cash_flow_statement (pd.DataFrame): A DataFrame containing the cash flow statement data for the companies.

    Returns:
        pd.DataFrame: A DataFrame containing the _liquidity ratios for the companies. If
                      only one ticker is provided, the returned DataFrame will have a single column
                      containing the data for that ticker. If multiple tickers are provided, the returned DataFrame
                      will have multiple columns, one for each ticker, with the ticker symbol as the column name.
    """
    solvency_ratios_dict: dict = {}

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if len(ticker_list) == 1:
        balance_sheet = pd.concat({ticker_list[0]: balance_sheet}, axis=1)
        income_statement = pd.concat({ticker_list[0]: income_statement}, axis=1)

    for ticker in ticker_list:
        bst = balance_sheet[ticker]
        ist = income_statement[ticker]

        solvency_ratios: pd.DataFrame = pd.DataFrame()

        solvency_ratios["Debt-to-Assets Ratio"] = _solvency.get_debt_to_assets_ratio(
            bst.loc["Total Debt"], bst.loc["Total Assets"]
        )

        solvency_ratios["Debt-to-Equity Ratio"] = _solvency.get_debt_to_equity_ratio(
            bst.loc["Total Debt"], bst.loc["Total Equity"]
        )

        solvency_ratios["Financial Leverage"] = _solvency.get_financial_leverage(
            bst.loc["Total Assets"], bst.loc["Total Equity"]
        )

        solvency_ratios[
            "Interest Coverage Ratio"
        ] = _solvency.get_interest_coverage_ratio(
            ist.loc["EBITDA"], ist.loc["Interest Expense"]
        )

        solvency_ratios[
            "Debt Service Coverage Ratio"
        ] = _solvency.get_debt_service_coverage_ratio(
            ist.loc["Operating Income"], bst.loc["Total Current Liabilities"]
        )

        solvency_ratios_dict[ticker] = solvency_ratios

    solvency_ratios_total = pd.concat(solvency_ratios_dict, axis=0)

    if len(ticker_list) == 1:
        solvency_ratios_total = solvency_ratios_total[ticker_list[0]]

    return solvency_ratios_total


def get_valuation_ratios(
    tickers: str | list[str],
    historical_data: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
    cash_flow_statement: pd.DataFrame,
):
    """
    Calculates various valuation ratios for each of the companies.
    It returns a DataFrame containing the calculated ratios.

    The function calculates the following valuation ratios for each company:
        earnings_per_share
        revenue_per_share
        price_earnings_ratio,
        earnings_per_share_growth
        price_to_earnings_growth_ratio
        book_value_per_share
        price_to_book_ratio
        payout_ratio
        dividend_yield
        earnings_yield

    Args:
        tickers (List[str]): List of company tickers.
        historical_data (pd.DataFrame):A DataFrame containing historical data for the companies.
        balance_sheet (pd.DataFrame): A DataFrame containing the balance sheet data for the companies.
        income_statement (pd.DataFrame): A DataFrame containing the income statement data for the companies.
        cash_flow_statement (pd.DataFrame): A DataFrame containing the cash flow statement data for the companies.

    Returns:
        pd.DataFrame: A DataFrame containing the _liquidity ratios for the companies. If
                      only one ticker is provided, the returned DataFrame will have a single column
                      containing the data for that ticker. If multiple tickers are provided, the returned DataFrame
                      will have multiple columns, one for each ticker, with the ticker symbol as the column name.
    """
    valuation_ratios_dict: dict = {}

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if len(ticker_list) == 1:
        historical_data = pd.concat({ticker_list[0]: historical_data}, axis=1)
        historical_data.columns = historical_data.columns.swaplevel(0, 1)
        balance_sheet = pd.concat({ticker_list[0]: balance_sheet}, axis=1)
        income_statement = pd.concat({ticker_list[0]: income_statement}, axis=1)
        cash_flow_statement = pd.concat({ticker_list[0]: cash_flow_statement}, axis=1)

    for ticker in ticker_list:
        bst = balance_sheet[ticker]
        ist = income_statement[ticker]
        cft = cash_flow_statement[ticker]
        hst = historical_data["Adj Close"][ticker].loc[
            [str(year) for year in bst.columns]
        ]

        valuation_ratios: pd.DataFrame = pd.DataFrame()

        # Preferred Dividends is defined as zero
        valuation_ratios[
            "Earnings Per Share (EPS)"
        ] = _valuation.get_earnings_per_share(
            ist.loc["Net Income"], 0, ist.loc["Weighted Average Shares"]
        )

        valuation_ratios["Revenue per Share (RPS)"] = _valuation.get_revenue_per_share(
            ist.loc["Revenue"], ist.loc["Weighted Average Shares"]
        )

        valuation_ratios[
            "Price-to-Earnings Ratio (PE)"
        ] = _valuation.get_price_earnings_ratio(
            hst.values, valuation_ratios["Earnings Per Share (EPS)"]
        )

        valuation_ratios[
            "Earnings Per Share Growth"
        ] = _valuation.get_earnings_per_share_growth(
            valuation_ratios["Earnings Per Share (EPS)"]
        )

        valuation_ratios[
            "Price-to-Earnings-Growth Ratio (PEG)"
        ] = _valuation.get_price_to_earnings_growth_ratio(
            valuation_ratios["Price-to-Earnings Ratio (PE)"],
            valuation_ratios["Earnings Per Share Growth"],
        )

        valuation_ratios["Book Value per Share"] = _valuation.get_book_value_per_share(
            bst.loc["Total Equity"], 0, ist.loc["Weighted Average Shares"]
        )

        valuation_ratios[
            "Price-to-Book Ratio (PB)"
        ] = _valuation.get_price_to_book_ratio(
            hst.values, valuation_ratios["Book Value per Share"]
        )

        valuation_ratios["Payout Ratio"] = _valuation.get_payout_ratio(
            cft.loc["Dividends Paid"], ist.loc["Net Income"]
        )

        valuation_ratios["Dividend Yield"] = _valuation.get_dividend_yield(
            abs(cft.loc["Dividends Paid"]) / ist.loc["Weighted Average Shares"],
            hst.values,
        )

        valuation_ratios["Earnings Yield"] = _valuation.get_earnings_yield(
            valuation_ratios["Earnings Per Share (EPS)"], hst.values
        )

        valuation_ratios_dict[ticker] = valuation_ratios

    valuation_ratios_total = pd.concat(valuation_ratios_dict, axis=0)

    if len(ticker_list) == 1:
        valuation_ratios_total = valuation_ratios_total[ticker_list[0]]

    return valuation_ratios_total
