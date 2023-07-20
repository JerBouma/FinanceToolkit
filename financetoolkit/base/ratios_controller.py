"""Ratios Module"""
__docformat__ = "numpy"

import pandas as pd

from financetoolkit.base.helpers import handle_errors
from financetoolkit.ratios import (
    efficiency,
    liquidity,
    profitability,
    solvency,
    valuation,
)

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-public-methods


class Ratios:
    """
    Ratios Controller Class
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical: pd.DataFrame,
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cash: pd.DataFrame,
    ):
        self._tickers = tickers
        self._historical_data: pd.DataFrame = historical
        self._balance_sheet_statement: pd.DataFrame = balance
        self._income_statement: pd.DataFrame = income
        self._cash_flow_statement: pd.DataFrame = cash

        # Initialization of Fundamentals Variables
        self._all_ratios: pd.DataFrame = pd.DataFrame()
        self._efficiency_ratios: pd.DataFrame = pd.DataFrame()
        self._liquidity_ratios: pd.DataFrame = pd.DataFrame()
        self._profitability_ratios: pd.DataFrame = pd.DataFrame()
        self._solvency_ratios: pd.DataFrame = pd.DataFrame()
        self._valuation_ratios: pd.DataFrame = pd.DataFrame()

    def collect_all_ratios(
        self, include_dividends: bool = False, diluted: bool = True, days: int = 365
    ):
        """
        Calculates all Ratios based on the data provided.
        """
        if self._efficiency_ratios.empty:
            self.collect_efficiency_ratios(days=days)
        if self._liquidity_ratios.empty:
            self.collect_liquidity_ratios()
        if self._profitability_ratios.empty:
            self.collect_profitability_ratios()
        if self._solvency_ratios.empty:
            self.collect_solvency_ratios(diluted=diluted)
        if self._valuation_ratios.empty:
            self.collect_valuation_ratios(
                include_dividends=include_dividends, diluted=diluted
            )

        self._all_ratios = pd.concat(
            [
                self._efficiency_ratios,
                self._liquidity_ratios,
                self._profitability_ratios,
                self._solvency_ratios,
                self._valuation_ratios,
            ]
        )

        return self._all_ratios

    def collect_efficiency_ratios(self, days: int = 365):
        """
        Calculates all Efficiency Ratios based on the data provided.
        """
        if self._efficiency_ratios.empty:
            efficiency_ratios: dict = {}

            efficiency_ratios[
                "Days of Inventory Outstanding (DIO)"
            ] = self.get_days_of_inventory_outstanding(days=days)
            efficiency_ratios[
                "Days of Sales Outstanding (DSO)"
            ] = self.get_days_of_sales_outstanding(days=days)
            efficiency_ratios["Operating Cycle (CC)"] = self.get_operating_cycle()
            efficiency_ratios[
                "Days of Accounts Payable Outstanding (DPO)"
            ] = self.get_days_of_accounts_payable_outstanding(days=days)
            efficiency_ratios[
                "Cash Conversion Cycle (CCC)"
            ] = self.get_cash_conversion_cycle(days=days)
            efficiency_ratios["Receivables Turnover"] = self.get_receivables_turnover()
            efficiency_ratios[
                "Inventory Turnover Ratio"
            ] = self.get_inventory_turnover_ratio()
            efficiency_ratios[
                "Accounts Payable Turnover Ratio"
            ] = self.get_accounts_payables_turnover_ratio()
            efficiency_ratios["SGA-to-Revenue Ratio"] = self.get_sga_to_revenue_ratio()
            efficiency_ratios["Fixed Asset Turnover"] = self.get_fixed_asset_turnover()
            efficiency_ratios["Asset Turnover Ratio"] = self.get_asset_turnover_ratio()
            efficiency_ratios["Operating Ratio"] = self.get_operating_ratio()

            self._efficiency_ratios = (
                pd.concat(efficiency_ratios)
                .swaplevel(0, 1)
                .sort_index(level=0, sort_remaining=False)
                .dropna(axis="columns", how="all")
            )

        if len(self._tickers) == 1:
            return self._efficiency_ratios.loc[self._tickers[0]]

        return self._efficiency_ratios

    def collect_liquidity_ratios(self) -> pd.DataFrame:
        """
        Calculates all Liquidity Ratios based on the data provided.
        """
        if self._liquidity_ratios.empty:
            liquidity_ratios: dict = {}

            liquidity_ratios["Current Ratio"] = self.get_current_ratio()
            liquidity_ratios["Quick Ratio"] = self.get_quick_ratio()
            liquidity_ratios["Cash Ratio"] = self.get_cash_ratio()
            liquidity_ratios["Working Capital"] = self.get_working_capital()
            liquidity_ratios[
                "Operating Cash Flow Ratio"
            ] = self.get_operating_cash_flow_ratio()
            liquidity_ratios[
                "Operating Cash Flow to Sales Ratio"
            ] = self.get_operating_cash_flow_sales_ratio()
            liquidity_ratios[
                "Short Term Coverage Ratio"
            ] = self.get_short_term_coverage_ratio()

            self._liquidity_ratios = (
                pd.concat(liquidity_ratios)
                .swaplevel(0, 1)
                .sort_index(level=0, sort_remaining=False)
                .dropna(axis="columns", how="all")
            )

        if len(self._tickers) == 1:
            return self._liquidity_ratios.loc[self._tickers[0]]

        return self._liquidity_ratios

    def collect_profitability_ratios(self) -> pd.DataFrame:
        """
        Calculates all Profitability Ratios based on the data provided.
        """
        if self._profitability_ratios.empty:
            profitability_ratios: dict = {}

            profitability_ratios["Gross Margin"] = self.get_gross_margin()
            profitability_ratios["Operating Margin"] = self.get_operating_margin()
            profitability_ratios["Net Profit Margin"] = self.get_net_profit_margin()
            profitability_ratios[
                "Interest Coverage Ratio"
            ] = self.get_interest_coverage_ratio()
            profitability_ratios[
                "Income Before Tax Profit Margin"
            ] = self.get_income_before_tax_profit_margin()
            profitability_ratios["Effective Tax Rate"] = self.get_effective_tax_rate()
            profitability_ratios["Return on Assets (ROA)"] = self.get_return_on_assets()
            profitability_ratios["Return on Equity (ROE)"] = self.get_return_on_equity()
            profitability_ratios[
                "Return on Invested Capital (ROIC)"
            ] = self.get_return_on_invested_capital()
            profitability_ratios[
                "Return on Capital Employed (ROCE)"
            ] = self.get_return_on_capital_employed()
            profitability_ratios[
                "Return on Tangible Assets"
            ] = self.get_return_on_tangible_assets()
            profitability_ratios[
                "Income Quality Ratio"
            ] = self.get_income_quality_ratio()
            profitability_ratios["Net Income per EBT"] = self.get_net_income_per_ebt()
            profitability_ratios[
                "Free Cash Flow to Operating Cash Flow Ratio"
            ] = self.get_free_cash_flow_operating_cash_flow_ratio()
            profitability_ratios["EBT to EBIT Ratio"] = self.get_EBT_to_EBIT()
            profitability_ratios["EBIT to Revenue"] = self.get_EBIT_to_revenue()

            self._profitability_ratios = (
                pd.concat(profitability_ratios)
                .swaplevel(0, 1)
                .sort_index(level=0, sort_remaining=False)
                .dropna(axis="columns", how="all")
            )

        if len(self._tickers) == 1:
            return self._profitability_ratios.loc[self._tickers[0]]

        return self._profitability_ratios

    def collect_solvency_ratios(self, diluted: bool = True):
        """
        Calculates all Solvency Ratios based on the data provided.
        """
        if self._solvency_ratios.empty:
            solvency_ratios: dict = {}

            solvency_ratios["Debt-to-Assets Ratio"] = self.get_debt_to_assets_ratio()
            solvency_ratios["Debt-to-Equity Ratio"] = self.get_debt_to_equity_ratio()
            solvency_ratios[
                "Interest Coverage Ratio"
            ] = self.get_interest_coverage_ratio()
            solvency_ratios[
                "Debt Service Coverage Ratio"
            ] = self.get_debt_service_coverage_ratio()
            solvency_ratios["Equity Multiplier"] = self.get_equity_multiplier()
            solvency_ratios["Free Cash Flow Yield"] = self.get_free_cash_flow_yield(
                diluted=diluted
            )
            solvency_ratios[
                "Net-Debt to EBITDA Ratio"
            ] = self.get_net_debt_to_ebitda_ratio()
            solvency_ratios["Cash Flow Coverage Ratio"] = self.get_free_cash_flow_yield(
                diluted=diluted
            )
            solvency_ratios["CAPEX Coverage Ratio"] = self.get_capex_coverage_ratio()
            solvency_ratios[
                "Dividend CAPEX Coverage Ratio"
            ] = self.get_capex_dividend_coverage_ratio()

            self._solvency_ratios = (
                pd.concat(solvency_ratios)
                .swaplevel(0, 1)
                .sort_index(level=0, sort_remaining=False)
                .dropna(axis="columns", how="all")
            )

        if len(self._tickers) == 1:
            return self._solvency_ratios.loc[self._tickers[0]]

        return self._solvency_ratios

    def collect_valuation_ratios(
        self, include_dividends: bool = False, diluted: bool = True
    ):
        """
        Calculates all Valuation Ratios based on the data provided.
        """
        if self._valuation_ratios.empty:
            valuation_ratios: dict = {}

            valuation_ratios["Earnings per Share (EPS)"] = self.get_earnings_per_share(
                include_dividends=include_dividends, diluted=diluted
            )
            valuation_ratios["Revenue per Share (RPS)"] = self.get_revenue_per_share(
                diluted=diluted
            )
            valuation_ratios["Price-to-Earnings (PE)"] = self.get_price_earnings_ratio(
                include_dividends=include_dividends, diluted=diluted
            )
            valuation_ratios[
                "Earnings per Share Growth"
            ] = self.get_earnings_per_share_growth(
                include_dividends=include_dividends, diluted=diluted
            )
            valuation_ratios[
                "Price-to-Earnings-Growth (PEG)"
            ] = self.get_price_to_earnings_growth_ratio(
                include_dividends=include_dividends, diluted=diluted
            )
            valuation_ratios["Book Value per Share"] = self.get_book_value_per_share(
                diluted=diluted
            )
            valuation_ratios["Price-to-Book (PB)"] = self.get_price_to_book_ratio(
                diluted=diluted
            )
            valuation_ratios[
                "Interest Debt per Share"
            ] = self.get_interest_debt_per_share(diluted=diluted)
            valuation_ratios["CAPEX per Share"] = self.get_capex_per_share(
                diluted=diluted
            )
            valuation_ratios["Earnings Yield"] = self.get_earnings_yield(
                include_dividends=include_dividends, diluted=diluted
            )
            valuation_ratios["Payout Ratio"] = self.get_payout_ratio()
            valuation_ratios["Dividend Yield"] = self.get_dividend_yield(
                diluted=diluted
            )
            valuation_ratios[
                "Price-to-Cash-Flow (P/CF)"
            ] = self.get_price_to_cash_flow_ratio(diluted=diluted)
            valuation_ratios[
                "Price-to-Free-Cash-Flow (P/FCF)"
            ] = self.get_price_to_free_cash_flow_ratio(diluted=diluted)
            valuation_ratios["Market Cap"] = self.get_market_cap(diluted=diluted)
            valuation_ratios["Enterprise Value"] = self.get_enterprise_value(
                diluted=diluted
            )
            valuation_ratios["EV-to-Sales"] = self.get_ev_to_sales_ratio(
                diluted=diluted
            )
            valuation_ratios["EV-to-EBIT"] = self.get_ev_to_ebit(diluted=diluted)
            valuation_ratios["EV-to-EBITDA"] = self.get_ev_to_ebitda_ratio(
                diluted=diluted
            )
            valuation_ratios[
                "EV-to-Operating-Cash-Flow"
            ] = self.get_ev_to_operating_cashflow_ratio(diluted=diluted)
            valuation_ratios["Tangible Asset Value"] = self.get_tangible_asset_value()
            valuation_ratios[
                "Net Current Asset Value"
            ] = self.get_net_current_asset_value()

            self._valuation_ratios = (
                pd.concat(valuation_ratios)
                .swaplevel(0, 1)
                .sort_index(level=0, sort_remaining=False)
                .dropna(axis="columns", how="all")
            )

        if len(self._tickers) == 1:
            return self._valuation_ratios.loc[self._tickers[0]]

        return self._valuation_ratios

    @handle_errors
    def get_asset_turnover_ratio(self):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.
        """
        return efficiency.get_asset_turnover_ratio(
            self._income_statement.loc[:, "Revenue", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Assets", :],
        )

    @handle_errors
    def get_inventory_turnover_ratio(self):
        """
        Calculate the inventory turnover ratio, an efficiency ratio that measures
        how quickly a company sells its inventory.
        """
        return efficiency.get_inventory_turnover_ratio(
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            self._balance_sheet_statement.loc[:, "Inventory", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Inventory", :],
        )

    @handle_errors
    def get_days_of_inventory_outstanding(self, days: int = 365):
        """
        Calculate the days sales in inventory ratio, an efficiency ratio that measures
        how long it takes a company to sell its inventory.
        """
        return efficiency.get_days_of_inventory_outstanding(
            self._balance_sheet_statement.loc[:, "Inventory", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Inventory", :],
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            days,
        )

    @handle_errors
    def get_days_of_sales_outstanding(self, days: int = 365):
        """
        Calculate the days of sales outstanding, an efficiency ratio that measures
        the average number of days it takes a company to collect payment on its
        credit sales.
        """
        return efficiency.get_days_of_sales_outstanding(
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :].shift(
                axis=1
            ),
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._income_statement.loc[:, "Revenue", :],
            days,
        )

    @handle_errors
    def get_operating_cycle(self, days: int = 365):
        """
        Calculate the operating cycle, an efficiency ratio that measures the average
        number of days it takes a company to turn its inventory into cash.
        """
        days_of_inventory = efficiency.get_days_of_inventory_outstanding(
            self._balance_sheet_statement.loc[:, "Inventory", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Inventory", :],
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            days,
        )
        days_of_sales = efficiency.get_days_of_sales_outstanding(
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :].shift(
                axis=1
            ),
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._income_statement.loc[:, "Revenue", :],
            days,
        )

        return efficiency.get_operating_cycle(days_of_inventory, days_of_sales)

    @handle_errors
    def get_accounts_payables_turnover_ratio(self):
        """
        Calculate the accounts payable turnover ratio is an efficiency ratio that measures how
        quickly a company pays its suppliers.
        """
        return efficiency.get_accounts_payables_turnover_ratio(
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            self._balance_sheet_statement.loc[:, "Accounts Payable", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Accounts Payable", :],
        )

    @handle_errors
    def get_days_of_accounts_payable_outstanding(self, days: int = 365):
        """
        Calculate the days payables outstanding, an efficiency ratio that measures the
        number of days it takes a company to pay its suppliers.
        """
        return efficiency.get_days_of_accounts_payable_outstanding(
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            self._balance_sheet_statement.loc[:, "Accounts Payable", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Accounts Payable", :],
            days,
        )

    @handle_errors
    def get_cash_conversion_cycle(self, days: int = 365):
        """
        Calculate the Cash Conversion Cycle, which measures the amount of time it takes for a company to convert
        its investments in inventory and accounts receivable into cash, while considering the time it takes to pay
        its accounts payable.
        """
        days_of_inventory = efficiency.get_days_of_inventory_outstanding(
            self._balance_sheet_statement.loc[:, "Inventory", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Inventory", :],
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            days,
        )
        days_of_sales = efficiency.get_days_of_sales_outstanding(
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :].shift(
                axis=1
            ),
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._income_statement.loc[:, "Revenue", :],
            days,
        )

        days_of_payables = efficiency.get_days_of_accounts_payable_outstanding(
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            self._balance_sheet_statement.loc[:, "Accounts Payable", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Accounts Payable", :],
            days,
        )

        return efficiency.get_cash_conversion_cycle(
            days_of_inventory, days_of_sales, days_of_payables
        )

    @handle_errors
    def get_receivables_turnover(self):
        """
        Calculate the receivables turnover, a ratio that measures how efficiently a
        company uses its assets by comparing the amount of credit extended to customers to
        the amount of sales generated.
        """
        return efficiency.get_receivables_turnover(
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :].shift(
                axis=1
            ),
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._income_statement.loc[:, "Revenue", :],
        )

    @handle_errors
    def get_sga_to_revenue_ratio(self):
        """
        Calculates the sales, general, and administrative (SG&A) expenses to revenue ratio,
        which measures the SG&A expenses relative to the revenue of the company.
        """
        return efficiency.get_sga_to_revenue_ratio(
            self._income_statement.loc[
                :, "Selling, General and Administrative Expenses", :
            ],
            self._income_statement.loc[:, "Revenue", :],
        )

    @handle_errors
    def get_fixed_asset_turnover(self):
        """
        Calculate the Fixed Asset Turnover ratio, an efficiency ratio that
        measures how efficiently a company uses its fixed assets to generate sales.
        """
        return efficiency.get_fixed_asset_turnover(
            self._income_statement.loc[:, "Revenue", :],
            self._balance_sheet_statement.loc[:, "Fixed Assets", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Fixed Assets", :],
        )

    @handle_errors
    def get_operating_ratio(self):
        """
        Calculate the operating ratio, a financial metric that measures the efficiency
        of a company's operations by comparing its operating expenses to its revenue.
        """
        return efficiency.get_operating_ratio(
            self._income_statement.loc[:, "Operating Expenses", :],
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            self._income_statement.loc[:, "Revenue", :],
        )

    @handle_errors
    def get_current_ratio(self):
        """
        Calculate the current ratio, a liquidity ratio that measures a company's ability
        to pay off its short-term liabilities with its current assets.
        """
        return liquidity.get_current_ratio(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

    @handle_errors
    def get_quick_ratio(self):
        """
        Calculate the quick ratio (also known as the acid-test ratio), a more stringent
        measure of liquidity that excludes inventory from current assets.

        This ratio is also referred to as the Acid Test Ratio.
        """
        return liquidity.get_quick_ratio(
            self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
            self._balance_sheet_statement.loc[:, "Short Term Investments", :],
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

    @handle_errors
    def get_cash_ratio(self):
        """
        Calculate the cash ratio, a liquidity ratio that measures a company's ability
        to pay off its short-term liabilities with its cash and cash equivalents.
        """
        return liquidity.get_cash_ratio(
            self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
            self._balance_sheet_statement.loc[:, "Short Term Investments", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

    @handle_errors
    def get_working_capital(self):
        """
        Calculate the working capital, which is the difference between a company's current assets
        and current liabilities.
        """
        return liquidity.get_working_capital(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

    @handle_errors
    def get_operating_cash_flow_ratio(self):
        """
        Calculate the operating cash flow ratio, a liquidity ratio that measures a company's
        ability to pay off its current liabilities with its operating cash flow.
        """
        return liquidity.get_operating_cash_flow_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

    @handle_errors
    def get_operating_cash_flow_sales_ratio(self):
        """
        Calculate the operating cash flow to sales ratio, a liquidity ratio that
        measures the ability of a company to generate cash from its sales.
        """
        return liquidity.get_operating_cash_flow_sales_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._income_statement.loc[:, "Revenue", :],
        )

    @handle_errors
    def get_short_term_coverage_ratio(self):
        """
        Calculate the short term coverage ratio, a liquidity ratio that measures a
        company's ability to pay off its short-term obligations with its operating cash flow.
        """
        return liquidity.get_short_term_coverage_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._balance_sheet_statement.loc[:, "Inventory", :],
            self._balance_sheet_statement.loc[:, "Accounts Payable", :],
        )

    @handle_errors
    def get_gross_margin(self):
        """
        Calculate the gross margin, a profitability ratio that measures the percentage of
        revenue that exceeds the cost of goods sold.
        """
        return profitability.get_gross_margin(
            self._income_statement.loc[:, "Revenue", :],
            self._income_statement.loc[:, "Cost of Goods Sold", :],
        )

    @handle_errors
    def get_operating_margin(self):
        """
        Calculate the operating margin, a profitability ratio that measures the percentage of
        revenue that remains after deducting operating expenses.
        """
        return profitability.get_operating_margin(
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Revenue", :],
        )

    @handle_errors
    def get_net_profit_margin(self):
        """
        Calculate the net profit margin, a profitability ratio that measures the percentage
        of profit a company earns per dollar of revenue.
        """
        return profitability.get_net_profit_margin(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Revenue", :],
        )

    @handle_errors
    def get_interest_burden_ratio(self):
        """
        Compute the Interest Coverage Ratio, a metric that reveals a company's
        ability to cover its interest expenses with its pre-tax profits.
        This ratio measures the proportion of pre-tax profits required to
        pay for interest payments and is crucial in determining a
        company's financial health.
        """
        return profitability.get_interest_burden_ratio(
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Interest Expense", :],
        )

    @handle_errors
    def get_income_before_tax_profit_margin(self):
        """
        Calculate the Pretax Profit Margin, which is the ratio of a company's pre-tax profit to
        its revenue, indicating how much profit a company makes before paying taxes on its earnings.
        """
        return profitability.get_income_before_tax_profit_margin(
            self._income_statement.loc[:, "Income Before Tax", :],
            self._income_statement.loc[:, "Revenue", :],
        )

    @handle_errors
    def get_effective_tax_rate(self):
        """
        Calculate the effective tax rate, a financial ratio that measures the
        percentage of pretax income that is paid as taxes.
        """
        return profitability.get_effective_tax_rate(
            self._income_statement.loc[:, "Income Tax Expense", :],
            self._income_statement.loc[:, "Income Before Tax", :],
        )

    @handle_errors
    def get_return_on_assets(self):
        """
        Calculate the return on assets (ROA), a profitability ratio that measures how
        efficiently a company uses its assets to generate profits.
        """
        return profitability.get_return_on_assets(
            self._income_statement.loc[:, "Net Income", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :],
        )

    @handle_errors
    def get_return_on_equity(self):
        """
        Calculate the return on equity (ROE), a profitability ratio that measures how
        efficiently a company generates profits using its shareholders' equity.
        """
        return profitability.get_return_on_equity(
            self._income_statement.loc[:, "Net Income", :],
            self._balance_sheet_statement.loc[:, "Total Equity", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Equity", :],
        )

    @handle_errors
    def get_return_on_invested_capital(self):
        """
        Calculate the return on invested capital, a financial ratio that measures
        the company's return on the capital invested in it, including both equity and debt.
        """
        effective_tax_rate = self.get_effective_tax_rate()

        return profitability.get_return_on_invested_capital(
            self._income_statement.loc[:, "Net Income", :],
            self._cash_flow_statement.loc[:, "Dividends Paid", :],
            effective_tax_rate,
            self._balance_sheet_statement.loc[:, "Total Equity", :],
            self._balance_sheet_statement.loc[:, "Total Debt", :],
        )

    @handle_errors
    def get_income_quality_ratio(self):
        """
        Calculates the income quality ratio, which measures the cash flow from operating
        activities relative to the net income of the company.
        """
        return profitability.get_income_quality_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._cash_flow_statement.loc[:, "Net Income", :],
        )

    @handle_errors
    def get_return_on_tangible_assets(self):
        """
        Calculate the return on tangible assets, which measures the amount of profit
        generated by a company's tangible assets.
        """
        return profitability.get_return_on_tangible_assets(
            self._income_statement.loc[:, "Net Income", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Intangible Assets", :],
            self._balance_sheet_statement.loc[:, "Total Liabilities", :],
        )

    @handle_errors
    def get_return_on_capital_employed(self):
        """
        Calculate the return on capital employed (ROCE), a profitability ratio that
        measures the amount of return a company generates from the capital it has
        invested in the business.
        """
        return profitability.get_return_on_capital_employed(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Interest Expense", :],
            self._income_statement.loc[:, "Income Tax Expense", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

    @handle_errors
    def get_net_income_per_ebt(self):
        """
        Calculate the net income per earnings before taxes (EBT), a profitability ratio that
        measures the net income generated for each dollar of EBT.
        """
        return profitability.get_net_income_per_ebt(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Income Tax Expense", :],
        )

    @handle_errors
    def get_free_cash_flow_operating_cash_flow_ratio(self):
        """
        Calculate the free cash flow to operating cash flow ratio, a profitability ratio that
        measures the amount of free cash flow a company generates for every dollar of operating cash flow.
        """
        return profitability.get_free_cash_flow_operating_cash_flow_ratio(
            self._cash_flow_statement.loc[:, "Free Cash Flow", :],
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
        )

    @handle_errors
    def get_tax_burden_ratio(self):
        """
        Calculate the tax burden ratio, which is the ratio of a company's
        net income to its income before tax, indicating how much of a
        company's income is retained after taxes.
        """
        return profitability.get_tax_burden_ratio(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Income Before Tax", :],
        )

    @handle_errors
    def get_EBT_to_EBIT(self):
        """
        Calculate the EBT to EBIT, which is the ratio of a company's earnings before tax to its
        earnings before interest and taxes, indicating how much of a company's earnings are
        generated before paying interest on debt.
        """
        return profitability.get_EBT_to_EBIT(
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :],
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :]
            + +self._income_statement.loc[:, "Interest Expense", :],
        )

    @handle_errors
    def get_EBIT_to_revenue(self):
        """
        Calculate the EBIT per Revenue, which is the ratio of a company's earnings
        before interest and taxes to its revenue, indicating how much profit a
        company generates from its operations before paying interest on debt
        and taxes on its earnings.
        """
        return profitability.get_EBT_to_EBIT(
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :]
            + +self._income_statement.loc[:, "Interest Expense", :],
            self._income_statement.loc[:, "Revenue", :],
        )

    @handle_errors
    def get_debt_to_assets_ratio(self):
        """
        Calculate the debt to assets ratio, a solvency ratio that measures the proportion
        of a company's assets that are financed by debt.

        This ratio is also known as the debt ratio.
        """
        return solvency.get_debt_to_assets_ratio(
            self._balance_sheet_statement.loc[:, "Total Debt", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :],
        )

    @handle_errors
    def get_debt_to_equity_ratio(self):
        """
        Calculate the debt to equity ratio, a solvency ratio that measures the
        proportion of a company's equity that is financed by debt.
        """
        return solvency.get_debt_to_equity_ratio(
            self._balance_sheet_statement.loc[:, "Total Debt", :],
            self._balance_sheet_statement.loc[:, "Total Equity", :],
        )

    @handle_errors
    def get_interest_coverage_ratio(self):
        """
        Calculate the interest coverage ratio, a solvency ratio that measures a company's
        ability to pay its interest expenses on outstanding debt.
        """
        return solvency.get_interest_coverage_ratio(
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Depreciation and Amortization", :],
            self._income_statement.loc[:, "Interest Expense", :],
        )

    @handle_errors
    def get_equity_multiplier(self):
        """
        Calculate the equity multiplier, a solvency ratio that measures the degree to which
        a company uses borrowed money (debt) to finance its operations and growth.
        """
        return solvency.get_equity_multiplier(
            self._balance_sheet_statement.loc[:, "Total Assets", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Total Equity", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Equity", :],
        )

    @handle_errors
    def get_debt_service_coverage_ratio(self):
        """
        Calculate the debt service coverage ratio, a solvency ratio that measures a company's
        ability to service its debt with its net operating income.
        """
        return solvency.get_debt_service_coverage_ratio(
            self._income_statement.loc[:, "Operating Income", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

    @handle_errors
    def get_free_cash_flow_yield(self, diluted: bool = True):
        """
        Calculates the free cash flow yield ratio, which measures the free cash flow
        relative to the market capitalization of the company.
        """
        years = self._balance_sheet_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        market_cap = valuation.get_market_cap(share_prices, average_shares)

        return solvency.get_free_cash_flow_yield(
            self._cash_flow_statement.loc[:, "Free Cash Flow", :],
            market_cap,
        )

    @handle_errors
    def get_net_debt_to_ebitda_ratio(self):
        """
        Calculates the net debt to EBITDA ratio, which measures the net debt of the company
        relative to its EBITDA.
        """
        return solvency.get_net_debt_to_ebitda_ratio(
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Depreciation and Amortization", :],
            self._balance_sheet_statement.loc[:, "Net Debt", :],
        )

    @handle_errors
    def get_cash_flow_coverage_ratio(self):
        """
        Calculate the cash flow coverage ratio, a solvency ratio that measures a company's
        ability to pay off its debt with its operating cash flow.
        """
        return solvency.get_cash_flow_coverage_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._balance_sheet_statement.loc[:, "Total Debt", :],
        )

    @handle_errors
    def get_capex_coverage_ratio(self):
        """
        Calculate the capital expenditure coverage ratio, a solvency ratio that
        measures a company's ability to cover its capital expenditures with its
        cash flow from operations.
        """
        return solvency.get_capex_coverage_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._cash_flow_statement.loc[:, "Capital Expenditure", :],
        )

    @handle_errors
    def get_capex_dividend_coverage_ratio(self):
        """
        Calculate the dividend paid and capex coverage ratio, a solvency ratio that
        measures a company's ability to cover both its capital expenditures and
        dividend payments with its cash flow from operations.
        """
        return solvency.get_dividend_capex_coverage_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._cash_flow_statement.loc[:, "Capital Expenditure", :],
            self._cash_flow_statement.loc[:, "Dividends Paid", :],
        )

    @handle_errors
    def get_earnings_per_share(
        self, include_dividends: bool = False, diluted: bool = True
    ):
        """
        Calculate the earnings per share (EPS), a valuation ratio that measures the
        amount of net income earned per share of outstanding common stock.
        """
        dividends = (
            self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :]
            if include_dividends
            else 0
        )

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        return valuation.get_earnings_per_share(
            self._income_statement.loc[:, "Net Income", :], dividends, average_shares
        )

    @handle_errors
    def get_earnings_per_share_growth(
        self, include_dividends: bool = False, diluted: bool = True
    ):
        """
        Calculate the earnings per share growth.
        """
        eps = self.get_earnings_per_share(include_dividends, diluted)

        return valuation.get_earnings_per_share_growth(eps)

    @handle_errors
    def get_revenue_per_share(self, diluted: bool = True):
        """
        Calculate the revenue per share, a valuation ratio that measures the amount
        of revenue generated per outstanding share of a company's stock.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        return valuation.get_revenue_per_share(
            self._income_statement.loc[:, "Revenue", :], average_shares
        )

    @handle_errors
    def get_price_earnings_ratio(
        self, include_dividends: bool = False, diluted: bool = True
    ):
        """
        Calculate the price earnings ratio (P/E), a valuation ratio that compares a
        company's stock price to its earnings per share.
        """
        eps = self.get_earnings_per_share(include_dividends, diluted)

        years = eps.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        return valuation.get_price_earnings_ratio(share_prices, eps)

    @handle_errors
    def get_price_to_earnings_growth_ratio(
        self, include_dividends: bool = False, diluted: bool = True
    ):
        """
        Calculate the price earnings to growth (PEG) ratio, a valuation metric that
        measures the ratio of the price-to-earnings ratio to earnings growth rate.
        """
        eps_growth = self.get_earnings_per_share(include_dividends, diluted)
        price_earnings = self.get_price_earnings_ratio(include_dividends, diluted)

        return valuation.get_price_to_earnings_growth_ratio(price_earnings, eps_growth)

    @handle_errors
    def get_book_value_per_share(self, diluted: bool = True):
        """
        Calculate the book value per share, a valuation ratio that measures the
        amount of common equity value per share outstanding.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        return valuation.get_book_value_per_share(
            self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :],
            self._balance_sheet_statement.loc[:, "Preferred Stock", :],
            average_shares,
        )

    @handle_errors
    def get_price_to_book_ratio(self, diluted: bool = True):
        """
        Calculate the price to book ratio, a valuation ratio that compares a
        company's market price to its book value per share.
        """
        book_value_per_share = self.get_book_value_per_share(diluted)

        years = book_value_per_share.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        return valuation.get_price_to_book_ratio(share_prices, book_value_per_share)

    @handle_errors
    def get_interest_debt_per_share(self, diluted: bool = True):
        """
        Calculate the interest debt per share, a valuation ratio that measures the
        amount of interest expense incurred per outstanding share of a company's stock.
        """

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        return valuation.get_interest_debt_per_share(
            self._income_statement.loc[:, "Interest Expense", :],
            self._balance_sheet_statement.loc[:, "Total Debt", :],
            average_shares,
        )

    @handle_errors
    def get_capex_per_share(self, diluted: bool = True):
        """
        Calculate the capex per share, a valuation ratio that measures the amount of
        capital expenditures made per outstanding share of a company's stock.
        """

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        return valuation.get_capex_per_share(
            self._cash_flow_statement.loc[:, "Capital Expenditure", :], average_shares
        )

    @handle_errors
    def get_dividend_yield(self, diluted: bool = True):
        """
        Calculate the dividend yield ratio, a valuation ratio that measures the
        amount of dividends distributed per share of stock relative to the stock's price.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        return valuation.get_dividend_yield(
            abs(self._cash_flow_statement.loc[:, "Dividends Paid", :]),
            average_shares,
            share_prices,
        )

    @handle_errors
    def get_price_to_cash_flow_ratio(self, diluted: bool = True):
        """
        Calculate the price to cash flow ratio, a valuation ratio that compares a
        company's market price to its operating cash flow per share.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        market_cap = valuation.get_market_cap(share_prices, average_shares)

        return valuation.get_price_to_cash_flow_ratio(
            market_cap, self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
        )

    @handle_errors
    def get_price_to_free_cash_flow_ratio(self, diluted: bool = True):
        """
        Calculate the price to free cash flow ratio, a valuation ratio that compares a
        company's market price to its free cash flow per share.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        market_cap = valuation.get_market_cap(share_prices, average_shares)

        return valuation.get_price_to_free_cash_flow_ratio(
            market_cap, self._cash_flow_statement.loc[:, "Free Cash Flow", :]
        )

    @handle_errors
    def get_market_cap(self, diluted: bool = True):
        """
        Calculates the market capitalization of the company.

        Note: All the inputs must be in the same currency and unit for accurate calculations.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        return valuation.get_market_cap(share_prices, average_shares)

    @handle_errors
    def get_enterprise_value(self, diluted: bool = True):
        """
        Calculates the Enterprise Value (EV) of a company. The Enterprise Value (EV)
        is a measure of a company's total value, often used as a more comprehensive
        alternative to market capitalization. It is calculated as the sum of a company's
        market capitalization, outstanding debt, minority interest, and
        preferred equity, minus the cash and cash equivalents.

        Note: All the inputs must be in the same currency and unit for accurate calculations.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        market_cap = valuation.get_market_cap(share_prices, average_shares)

        return valuation.get_enterprise_value(
            market_cap,
            self._balance_sheet_statement.loc[:, "Total Debt", :],
            self._balance_sheet_statement.loc[:, "Minority Interest", :],
            self._balance_sheet_statement.loc[:, "Preferred Stock", :],
            self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
        )

    @handle_errors
    def get_ev_to_sales_ratio(self, diluted: bool = True):
        """
        Calculate the EV to sales ratio, a valuation ratio that compares a company's
        enterprise value (EV) to its total revenue.
        """
        enterprise_value = self.get_enterprise_value(diluted)

        return valuation.get_ev_to_sales_ratio(
            enterprise_value, self._income_statement.loc[:, "Revenue", :]
        )

    @handle_errors
    def get_ev_to_ebitda_ratio(self, diluted: bool = True):
        """
        Calculates the enterprise value over EBITDA ratio, which is a valuation ratio that
        measures a company's total value (including debt and equity) relative to its EBITDA.
        """
        enterprise_value = self.get_enterprise_value(diluted)

        return valuation.get_ev_to_ebitda_ratio(
            enterprise_value,
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Depreciation and Amortization", :],
        )

    @handle_errors
    def get_ev_to_operating_cashflow_ratio(self, diluted: bool = True):
        """
        Calculates the enterprise value over operating cash flow ratio, which is a valuation
        ratio that measures a company's total value (including debt and equity) relative
        to its operating cash flow.
        """
        enterprise_value = self.get_enterprise_value(diluted)

        return valuation.get_ev_to_operating_cashflow_ratio(
            enterprise_value,
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
        )

    @handle_errors
    def get_earnings_yield(self, include_dividends: bool = False, diluted: bool = True):
        """
        Calculates the earnings yield ratio, which measures the earnings per share
        relative to the market price per share.
        """
        eps = self.get_earnings_per_share(include_dividends, diluted)

        years = eps.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        return valuation.get_earnings_yield(eps, share_prices)

    @handle_errors
    def get_payout_ratio(self):
        """
        Calculates the (dividend) payout ratio, which measures the proportion of earnings
        paid out as dividends to shareholders.
        """
        return valuation.get_payout_ratio(
            self._cash_flow_statement.loc[:, "Dividends Paid", :],
            self._income_statement.loc[:, "Net Income", :],
        )

    @handle_errors
    def get_tangible_asset_value(self):
        """
        Calculate the tangible asset value, which represents the total value of a company's
        assets that can be used to generate revenue.
        """
        return valuation.get_tangible_asset_value(
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Total Liabilities", :],
            self._balance_sheet_statement.loc[:, "Goodwill", :],
        )

    @handle_errors
    def get_net_current_asset_value(self):
        """
        Calculate the net current asset value, which is the total value of a company's
        current assets minus its current liabilities.
        """
        return valuation.get_net_current_asset_value(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

    @handle_errors
    def get_ev_to_ebit(self, diluted: bool = True):
        """
        Calculate the net current asset value, which is the total value of a company's
        current assets minus its current liabilities.
        """
        enterprise_value = self.get_enterprise_value(diluted)

        return valuation.get_ev_to_ebit(
            enterprise_value,
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :]
            + +self._income_statement.loc[:, "Interest Expense", :],
        )
