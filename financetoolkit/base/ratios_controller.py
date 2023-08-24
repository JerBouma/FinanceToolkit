"""Ratios Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import calculate_growth, handle_errors
from financetoolkit.ratios import (
    efficiency,
    liquidity,
    profitability,
    solvency,
    valuation,
)

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-public-methods,too-many-locals,eval-used


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
        custom_ratios_dict: dict | None = None,
        rounding: int | None = 4,
    ):
        """
        Initializes the Ratios Controller Class.
        """
        self._tickers = tickers
        self._historical_data: pd.DataFrame = historical
        self._balance_sheet_statement: pd.DataFrame = balance
        self._income_statement: pd.DataFrame = income
        self._cash_flow_statement: pd.DataFrame = cash
        self._custom_ratios_dict: dict = (
            custom_ratios_dict if custom_ratios_dict else {}
        )
        self._custom_ratios: pd.DataFrame = pd.DataFrame()
        self._custom_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._rounding: int | None = rounding

        # Initialization of Fundamentals Variables
        self._all_ratios: pd.DataFrame = pd.DataFrame()
        self._all_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._efficiency_ratios: pd.DataFrame = pd.DataFrame()
        self._efficiency_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._liquidity_ratios: pd.DataFrame = pd.DataFrame()
        self._liquidity_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._profitability_ratios: pd.DataFrame = pd.DataFrame()
        self._profitability_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._solvency_ratios: pd.DataFrame = pd.DataFrame()
        self._solvency_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._valuation_ratios: pd.DataFrame = pd.DataFrame()
        self._valuation_ratios_growth: pd.DataFrame = pd.DataFrame()

    def collect_all_ratios(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        days: int = 365,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates all Ratios based on the data provided.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculations. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for the calculation. Defaults to True.
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_all_ratios()
        ```
        """
        self._all_ratios = pd.concat(
            [
                self.collect_efficiency_ratios(days=days),
                self.collect_liquidity_ratios(),
                self.collect_profitability_ratios(),
                self.collect_solvency_ratios(diluted=diluted),
                self.collect_valuation_ratios(
                    include_dividends=include_dividends, diluted=diluted
                ),
            ]
        )

        self._all_ratios = self._all_ratios.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._all_ratios_growth = calculate_growth(
                self._all_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return self._all_ratios_growth if growth else self._all_ratios

    def collect_efficiency_ratios(
        self,
        days: int = 365,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates all Efficiency Ratios based on the data provided.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_efficiency_ratios()
        ```
        """
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

        self._efficiency_ratios = self._efficiency_ratios.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._efficiency_ratios_growth = calculate_growth(
                self._efficiency_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        if len(self._tickers) == 1:
            return (
                self._efficiency_ratios_growth[self._tickers[0]]
                if growth
                else self._efficiency_ratios.loc[self._tickers[0]]
            )

        return self._efficiency_ratios_growth if growth else self._efficiency_ratios

    def collect_liquidity_ratios(
        self,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        Calculates all Liquidity Ratios based on the data provided.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_liquidity_ratios()
        ```
        """
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

        self._liquidity_ratios = self._liquidity_ratios.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._liquidity_ratios_growth = calculate_growth(
                self._liquidity_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        if len(self._tickers) == 1:
            return (
                self._liquidity_ratios_growth[self._tickers[0]]
                if growth
                else self._liquidity_ratios.loc[self._tickers[0]]
            )

        return self._liquidity_ratios_growth if growth else self._liquidity_ratios

    def collect_profitability_ratios(
        self,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        Calculates all Profitability Ratios based on the data provided.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_profitability_ratios()
        ```
        """
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
        profitability_ratios["Income Quality Ratio"] = self.get_income_quality_ratio()
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

        self._profitability_ratios = self._profitability_ratios.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._profitability_ratios_growth = calculate_growth(
                self._profitability_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        if len(self._tickers) == 1:
            return (
                self._profitability_ratios_growth[self._tickers[0]]
                if growth
                else self._profitability_ratios.loc[self._tickers[0]]
            )

        return (
            self._profitability_ratios_growth if growth else self._profitability_ratios
        )

    def collect_solvency_ratios(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates all Solvency Ratios based on the data provided.

        Args:
            diluted (bool, optional): Whether to use diluted shares for the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_solvency_ratios()
        ```
        """
        solvency_ratios: dict = {}

        solvency_ratios["Debt-to-Assets Ratio"] = self.get_debt_to_assets_ratio()
        solvency_ratios["Debt-to-Equity Ratio"] = self.get_debt_to_equity_ratio()
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

        self._solvency_ratios = self._solvency_ratios.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._solvency_ratios_growth = calculate_growth(
                self._solvency_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        if len(self._tickers) == 1:
            return (
                self._solvency_ratios_growth[self._tickers[0]]
                if growth
                else self._solvency_ratios.loc[self._tickers[0]]
            )

        return self._solvency_ratios_growth if growth else self._solvency_ratios

    def collect_valuation_ratios(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates all Valuation Ratios based on the data provided.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculations. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite the existing ratios. Defaults to False.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.collect_valuation_ratios()
        ```
        """
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
        valuation_ratios["Interest Debt per Share"] = self.get_interest_debt_per_share(
            diluted=diluted
        )
        valuation_ratios["CAPEX per Share"] = self.get_capex_per_share(diluted=diluted)
        valuation_ratios["Earnings Yield"] = self.get_earnings_yield(
            include_dividends=include_dividends, diluted=diluted
        )
        valuation_ratios["Payout Ratio"] = self.get_payout_ratio()
        valuation_ratios["Dividend Yield"] = self.get_dividend_yield()
        valuation_ratios["Weighted Dividend Yield"] = self.get_weighted_dividend_yield(
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
        valuation_ratios["EV-to-Sales"] = self.get_ev_to_sales_ratio(diluted=diluted)
        valuation_ratios["EV-to-EBIT"] = self.get_ev_to_ebit(diluted=diluted)
        valuation_ratios["EV-to-EBITDA"] = self.get_ev_to_ebitda_ratio(diluted=diluted)
        valuation_ratios[
            "EV-to-Operating-Cash-Flow"
        ] = self.get_ev_to_operating_cashflow_ratio(diluted=diluted)
        valuation_ratios["Tangible Asset Value"] = self.get_tangible_asset_value()
        valuation_ratios["Net Current Asset Value"] = self.get_net_current_asset_value()

        self._valuation_ratios = (
            pd.concat(valuation_ratios)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._valuation_ratios = self._valuation_ratios.round(
            rounding if rounding else self._rounding
        )

        if growth:
            self._valuation_ratios_growth = calculate_growth(
                self._valuation_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        if len(self._tickers) == 1:
            return (
                self._valuation_ratios_growth[self._tickers[0]]
                if growth
                else self._valuation_ratios.loc[self._tickers[0]]
            )

        return self._valuation_ratios_growth if growth else self._valuation_ratios

    @handle_errors
    def collect_custom_ratios(
        self,
        custom_ratios_dict: dict | None = None,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates all Custom Ratios based on the data provided.

        Note that any of the following characters are considered as operators:
            +, -, *, /, **, %, //, <, >, ==, !=, >=, <=, (, )
        using any of the above characters as part of the column naming will result into an error.

        Args:
            custom_ratios (dict, optional): A dictionary containing the custom ratios to calculate. This is
            an optional parameter given that you can also define the custom ratios through the Toolkit initialization.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        custom_ratios = {
            'WC / Net Income as %': '(Working Capital / Net Income) * 100',
            'Large Revenues': 'Revenue > 1000000000',
            'Quick Assets': 'Cash and Short Term Investments + Accounts Receivable',
            'Cash Op Expenses':'Cost of Goods Sold + Selling, General and Administrative Expenses '
            '- Depreciation and Amortization',
            'Daily Cash Op Expenses': 'Cash Op Expenses / 365',
            'Defensive Interval':'Quick Assets / Daily Cash Op Expenses'
        }

        companies = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key=API_KEY, start_date="2022-10-01",
            custom_ratios=custom_ratios, quarterly=True
        )

        custom_ratios = companies.ratios.collect_custom_ratios()

        custom_ratios.loc['AMZN']
        ```

        Which returns:

        |                        |         2022Q4 |         2023Q1 |         2023Q2 |   2023Q3 |
        |:-----------------------|---------------:|---------------:|---------------:|---------:|
        | Cash Op Expenses       |    2.1856e+10  |    1.9972e+10  |    2.1322e+10  |      nan |
        | Daily Cash Op Expenses |    5.98795e+07 |    5.47178e+07 |    5.84164e+07 |      nan |
        | Defensive Interval     | 2260.22        | 2592.34        | 2738.1         |      nan |
        | Large Revenues         |    1           |    1           |    1           |        0 |
        | Quick Assets           |    1.35341e+11 |    1.41847e+11 |    1.5995e+11  |      nan |
        | WC / Net Income as %   |  463.349       |  427.335       |  398.924       |      nan |
        """
        if not self._custom_ratios_dict and not custom_ratios_dict:
            print(
                "Please define custom ratios through the Toolkit initialization or include a "
                "dictionary to the custom_ratios_dict parameter. "
                "See https://github.com/JerBouma/FinanceToolkit how to do this."
            )

        custom_ratios_dict = (
            custom_ratios_dict if custom_ratios_dict else self._custom_ratios_dict
        )

        if self._all_ratios.empty:
            self._all_ratios = self.collect_all_ratios()

        custom_ratios = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [self._tickers, custom_ratios_dict.keys()]
            ),
            columns=self._balance_sheet_statement.columns,
        )

        total_financials = pd.concat(
            [
                self._balance_sheet_statement,
                self._income_statement,
                self._cash_flow_statement,
                self._all_ratios,
                custom_ratios,
            ],
            axis=0,
        )

        total_financials = total_financials[
            ~total_financials.index.duplicated(keep="first")
        ]

        formula_dict = {}
        for name, formula in custom_ratios_dict.items():
            # Rearrange the formula dict in case a formula is dependent on another formula
            # and the order would result into errors
            for sub_name, sub_formula in custom_ratios_dict.items():
                if sub_name in formula:
                    formula_dict[sub_name] = sub_formula

            if name not in formula_dict:
                formula_dict[name] = formula

        for name, formula in formula_dict.items():
            formula_names = formula

            for operator in [
                "+",
                "-",
                "*",
                "/",
                "**",
                "%",
                "//",
                "<",
                ">",
                "==",
                "!=",
                ">=",
                "<=",
                "(",
                ")",
            ]:
                formula_names = formula_names.replace(operator, "SPLIT")

            formula_names = formula_names.split("SPLIT")

            formula_names = [
                clean_name
                for clean_name in formula_names
                if clean_name not in ["", " "]
            ]

            formula_adjusted = formula

            for formula_section in formula_names:
                formula_section_stripped = formula_section.strip()
                if formula_section_stripped in total_financials.index.get_level_values(
                    1
                ):
                    formula_adjusted = formula_adjusted.replace(
                        formula_section_stripped,
                        f"total_financials.loc[:, '{formula_section_stripped}', :]",
                    )
                else:
                    try:
                        float(formula_section_stripped)
                    except ValueError:
                        formula_adjusted = None
                        print(
                            f"Column {formula_section_stripped} not found in total_financials and is not a number. "
                            f"Therefore the formula {formula} is invalid."
                        )
                        break

            if formula_adjusted:
                total_financials.loc[:, name, :] = eval(formula_adjusted).to_numpy()

                self._custom_ratios = total_financials.loc[
                    :, list(custom_ratios_dict.keys()), :
                ]
                self._custom_ratios = self._custom_ratios.sort_index(axis=0)

                self._custom_ratios = self._custom_ratios.round(
                    rounding if rounding else self._rounding
                )

        if growth:
            self._custom_ratios_growth = calculate_growth(
                self._custom_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        if len(self._tickers) == 1:
            return (
                self._custom_ratios_growth[self._tickers[0]]
                if growth
                else self._custom_ratios.loc[self._tickers[0]]
            )

        return self._custom_ratios_growth if growth else self._custom_ratios

    @handle_errors
    def get_asset_turnover_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_asset_turnover_ratio()
        ```
        """
        asset_turnover_ratio = efficiency.get_asset_turnover_ratio(
            self._income_statement.loc[:, "Revenue", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Assets", :],
        )

        if growth:
            return calculate_growth(
                asset_turnover_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return asset_turnover_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_inventory_turnover_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the inventory turnover ratio, an efficiency ratio that measures
        how quickly a company sells its inventory.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_inventory_turnover_ratio()
        ```
        """
        inventory_turnover_ratio = efficiency.get_inventory_turnover_ratio(
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            self._balance_sheet_statement.loc[:, "Inventory", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Inventory", :],
        )

        if growth:
            return calculate_growth(
                inventory_turnover_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return inventory_turnover_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_days_of_inventory_outstanding(
        self,
        days: int = 365,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the days sales in inventory ratio, an efficiency ratio that measures
        how long it takes a company to sell its inventory.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_days_of_inventory_outstanding()
        ```
        """
        days_of_inventory_outstanding = efficiency.get_days_of_inventory_outstanding(
            self._balance_sheet_statement.loc[:, "Inventory", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Inventory", :],
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            days,
        )

        if growth:
            return calculate_growth(
                days_of_inventory_outstanding,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return days_of_inventory_outstanding.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_days_of_sales_outstanding(
        self,
        days: int = 365,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the days of sales outstanding, an efficiency ratio that measures
        the average number of days it takes a company to collect payment on its
        credit sales.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_days_of_sales_outstanding()
        ```
        """
        days_of_sales_outstanding = efficiency.get_days_of_sales_outstanding(
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :].shift(
                axis=1
            ),
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._income_statement.loc[:, "Revenue", :],
            days,
        )

        if growth:
            return calculate_growth(
                days_of_sales_outstanding,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return days_of_sales_outstanding.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_operating_cycle(
        self,
        days: int = 365,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the operating cycle, an efficiency ratio that measures the average
        number of days it takes a company to turn its inventory into cash.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_operating_cycle()
        ```
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

        operating_cycle = efficiency.get_operating_cycle(
            days_of_inventory, days_of_sales
        )

        if growth:
            return calculate_growth(
                operating_cycle,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return operating_cycle.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_accounts_payables_turnover_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the accounts payable turnover ratio is an efficiency ratio that measures how
        quickly a company pays its suppliers.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_accounts_payables_turnover_ratio()
        ```
        """
        accounts_payables_turnover_ratio = (
            efficiency.get_accounts_payables_turnover_ratio(
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                self._balance_sheet_statement.loc[:, "Accounts Payable", :].shift(
                    axis=1
                ),
                self._balance_sheet_statement.loc[:, "Accounts Payable", :],
            )
        )

        if growth:
            return calculate_growth(
                accounts_payables_turnover_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return accounts_payables_turnover_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_days_of_accounts_payable_outstanding(
        self,
        days: int = 365,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the days payables outstanding, an efficiency ratio that measures the
        number of days it takes a company to pay its suppliers.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_days_of_accounts_payable_outstanding()
        ```
        """
        days_of_accounts_payable_outstanding = (
            efficiency.get_days_of_accounts_payable_outstanding(
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                self._balance_sheet_statement.loc[:, "Accounts Payable", :].shift(
                    axis=1
                ),
                self._balance_sheet_statement.loc[:, "Accounts Payable", :],
                days,
            )
        )

        if growth:
            return calculate_growth(
                days_of_accounts_payable_outstanding,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return days_of_accounts_payable_outstanding.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_cash_conversion_cycle(
        self,
        days: int = 365,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the Cash Conversion Cycle, which measures the amount of time it takes for a company to convert
        its investments in inventory and accounts receivable into cash, while considering the time it takes to pay
        its accounts payable.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_cash_conversion_cycle()
        ```
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

        cash_conversion_cycle = efficiency.get_cash_conversion_cycle(
            days_of_inventory, days_of_sales, days_of_payables
        )

        if growth:
            return calculate_growth(
                cash_conversion_cycle,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return cash_conversion_cycle.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_receivables_turnover(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the receivables turnover, a ratio that measures how efficiently a
        company uses its assets by comparing the amount of credit extended to customers to
        the amount of sales generated.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_receivables_turnover()
        ```
        """
        receivables_turnover = efficiency.get_receivables_turnover(
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :].shift(
                axis=1
            ),
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._income_statement.loc[:, "Revenue", :],
        )

        if growth:
            return calculate_growth(
                receivables_turnover,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return receivables_turnover.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_sga_to_revenue_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculates the sales, general, and administrative (SG&A) expenses to revenue ratio,
        which measures the SG&A expenses relative to the revenue of the company.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.


        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_sga_to_revenue_ratio()
        ```
        """
        sga_to_revenue_ratio = efficiency.get_sga_to_revenue_ratio(
            self._income_statement.loc[
                :, "Selling, General and Administrative Expenses", :
            ],
            self._income_statement.loc[:, "Revenue", :],
        )

        if growth:
            return calculate_growth(
                sga_to_revenue_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return sga_to_revenue_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_fixed_asset_turnover(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the Fixed Asset Turnover ratio, an efficiency ratio that
        measures how efficiently a company uses its fixed assets to generate sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_fixed_asset_turnover()
        ```
        """
        fixed_asset_turnover = efficiency.get_fixed_asset_turnover(
            self._income_statement.loc[:, "Revenue", :],
            self._balance_sheet_statement.loc[:, "Fixed Assets", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Fixed Assets", :],
        )

        if growth:
            return calculate_growth(
                fixed_asset_turnover,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return fixed_asset_turnover.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_operating_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the operating ratio, a financial metric that measures the efficiency
        of a company's operations by comparing its operating expenses to its revenue.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_operating_ratio()
        ```
        """
        operating_ratio = efficiency.get_operating_ratio(
            self._income_statement.loc[:, "Operating Expenses", :],
            self._income_statement.loc[:, "Cost of Goods Sold", :],
            self._income_statement.loc[:, "Revenue", :],
        )

        if growth:
            return calculate_growth(
                operating_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return operating_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_current_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the current ratio, a liquidity ratio that measures a company's ability
        to pay off its short-term liabilities with its current assets.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_current_ratio()
        ```
        """
        current_ratio = liquidity.get_current_ratio(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if growth:
            return calculate_growth(
                current_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return current_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_quick_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the quick ratio (also known as the acid-test ratio), a more stringent
        measure of liquidity that excludes inventory from current assets.

        This ratio is also referred to as the Acid Test Ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit
        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_quick_ratio()
        ```
        """
        quick_ratio = liquidity.get_quick_ratio(
            self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
            self._balance_sheet_statement.loc[:, "Short Term Investments", :],
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if growth:
            return calculate_growth(
                quick_ratio, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return quick_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_cash_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the cash ratio, a liquidity ratio that measures a company's ability
        to pay off its short-term liabilities with its cash and cash equivalents.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit


        toolkit.ratios.get_cash_ratio()
        ```
        """
        cash_ratio = liquidity.get_cash_ratio(
            self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
            self._balance_sheet_statement.loc[:, "Short Term Investments", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if growth:
            return calculate_growth(
                cash_ratio, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return cash_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_working_capital(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the working capital, which is the difference between a company's current assets
        and current liabilities.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)
        toolkit.ratios.get_working_capital()
        ```
        """
        working_capital = liquidity.get_working_capital(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if growth:
            return calculate_growth(
                working_capital,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return working_capital.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_operating_cash_flow_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the operating cash flow ratio, a liquidity ratio that measures a company's
        ability to pay off its current liabilities with its operating cash flow.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_operating_cash_flow_ratio()
        ```
        """
        operating_cash_flow_ratio = liquidity.get_operating_cash_flow_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if growth:
            return calculate_growth(
                operating_cash_flow_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return operating_cash_flow_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_operating_cash_flow_sales_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the operating cash flow to sales ratio, a liquidity ratio that
        measures the ability of a company to generate cash from its sales.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_operating_cash_flow_sales_ratio()
        ```
        """
        operating_cash_flow_sales_ratio = liquidity.get_operating_cash_flow_sales_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._income_statement.loc[:, "Revenue", :],
        )

        if growth:
            return calculate_growth(
                operating_cash_flow_sales_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return operating_cash_flow_sales_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_short_term_coverage_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the short term coverage ratio, a liquidity ratio that measures a
        company's ability to pay off its short-term obligations with its operating cash flow.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_short_term_coverage_ratio()
        ```
        """
        short_term_coverage_ratio = liquidity.get_short_term_coverage_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
            self._balance_sheet_statement.loc[:, "Inventory", :],
            self._balance_sheet_statement.loc[:, "Accounts Payable", :],
        )

        if growth:
            return calculate_growth(
                short_term_coverage_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return short_term_coverage_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_gross_margin(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the gross margin, a profitability ratio that measures the percentage of
        revenue that exceeds the cost of goods sold.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_gross_margin()
        ```
        """
        gross_margin = profitability.get_gross_margin(
            self._income_statement.loc[:, "Revenue", :],
            self._income_statement.loc[:, "Cost of Goods Sold", :],
        )

        if growth:
            return calculate_growth(
                gross_margin, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return gross_margin.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_operating_margin(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the operating margin, a profitability ratio that measures the percentage of
        revenue that remains after deducting operating expenses.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_operating_margin()
        ```
        """
        operating_margin = profitability.get_operating_margin(
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Revenue", :],
        )

        if growth:
            return calculate_growth(
                operating_margin,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return operating_margin.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_net_profit_margin(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the net profit margin, a profitability ratio that measures the percentage
        of profit a company earns per dollar of revenue.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_net_profit_margin()
        ```
        """
        net_profit_margin = profitability.get_net_profit_margin(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Revenue", :],
        )

        if growth:
            return calculate_growth(
                net_profit_margin,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return net_profit_margin.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_interest_burden_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Compute the Interest Coverage Ratio, a metric that reveals a company's
        ability to cover its interest expenses with its pre-tax profits.
        This ratio measures the proportion of pre-tax profits required to
        pay for interest payments and is crucial in determining a
        company's financial health.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_interest_burden_ratio()
        ```
        """
        interest_burden_ratio = profitability.get_interest_burden_ratio(
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Interest Expense", :],
        )

        if growth:
            return calculate_growth(
                interest_burden_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return interest_burden_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_income_before_tax_profit_margin(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the Pretax Profit Margin, which is the ratio of a company's pre-tax profit to
        its revenue, indicating how much profit a company makes before paying taxes on its earnings.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_income_before_tax_profit_margin()
        ```
        """
        income_before_tax_profit_margin = (
            profitability.get_income_before_tax_profit_margin(
                self._income_statement.loc[:, "Income Before Tax", :],
                self._income_statement.loc[:, "Revenue", :],
            )
        )

        if growth:
            return calculate_growth(
                income_before_tax_profit_margin,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return income_before_tax_profit_margin.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_effective_tax_rate(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the effective tax rate, a financial ratio that measures the
        percentage of pretax income that is paid as taxes.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_effective_tax_rate()
        ```
        """
        effective_tax_rate = profitability.get_effective_tax_rate(
            self._income_statement.loc[:, "Income Tax Expense", :],
            self._income_statement.loc[:, "Income Before Tax", :],
        )

        if growth:
            return calculate_growth(
                effective_tax_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return effective_tax_rate.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_return_on_assets(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the return on assets (ROA), a profitability ratio that measures how
        efficiently a company uses its assets to generate profits.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_return_on_assets()
        ```
        """
        return_on_assets = profitability.get_return_on_assets(
            self._income_statement.loc[:, "Net Income", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :],
        )

        if growth:
            return calculate_growth(
                return_on_assets,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return return_on_assets.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_return_on_equity(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the return on equity (ROE), a profitability ratio that measures how
        efficiently a company generates profits using its shareholders' equity.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_return_on_equity()
        ```
        """
        return_on_equity = profitability.get_return_on_equity(
            self._income_statement.loc[:, "Net Income", :],
            self._balance_sheet_statement.loc[:, "Total Equity", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Equity", :],
        )

        if growth:
            return calculate_growth(
                return_on_equity,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return return_on_equity.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_return_on_invested_capital(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the return on invested capital, a financial ratio that measures
        the company's return on the capital invested in it, including both equity and debt.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_return_on_invested_capital()
        ```
        """
        effective_tax_rate = self.get_effective_tax_rate()

        return_on_invested_capital = profitability.get_return_on_invested_capital(
            self._income_statement.loc[:, "Net Income", :],
            self._cash_flow_statement.loc[:, "Dividends Paid", :],
            effective_tax_rate,
            self._balance_sheet_statement.loc[:, "Total Equity", :],
            self._balance_sheet_statement.loc[:, "Total Debt", :],
        )

        if growth:
            return calculate_growth(
                return_on_invested_capital,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return return_on_invested_capital.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_income_quality_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculates the income quality ratio, which measures the cash flow from operating
        activities relative to the net income of the company.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_income_quality_ratio()
        ```
        """
        income_quality_ratio = profitability.get_income_quality_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._cash_flow_statement.loc[:, "Net Income", :],
        )

        if growth:
            return calculate_growth(
                income_quality_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return income_quality_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_return_on_tangible_assets(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the return on tangible assets, which measures the amount of profit
        generated by a company's tangible assets.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_return_on_tangible_assets()
        ```
        """
        return_on_tangible_assets = profitability.get_return_on_tangible_assets(
            self._income_statement.loc[:, "Net Income", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Intangible Assets", :],
            self._balance_sheet_statement.loc[:, "Total Liabilities", :],
        )

        if growth:
            return calculate_growth(
                return_on_tangible_assets,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return return_on_tangible_assets.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_return_on_capital_employed(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the return on capital employed (ROCE), a profitability ratio that
        measures the amount of return a company generates from the capital it has
        invested in the business.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_return_on_capital_employed()
        ```
        """
        return_on_capital_employed = profitability.get_return_on_capital_employed(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Interest Expense", :],
            self._income_statement.loc[:, "Income Tax Expense", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if growth:
            return calculate_growth(
                return_on_capital_employed,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return return_on_capital_employed.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_net_income_per_ebt(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the net income per earnings before taxes (EBT), a profitability ratio that
        measures the net income generated for each dollar of EBT.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_net_income_per_ebt()
        ```
        """
        net_income_per_ebt = profitability.get_net_income_per_ebt(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Income Tax Expense", :],
        )

        if growth:
            return calculate_growth(
                net_income_per_ebt,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return net_income_per_ebt.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_free_cash_flow_operating_cash_flow_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the free cash flow to operating cash flow ratio, a profitability ratio that
        measures the amount of free cash flow a company generates for every dollar of operating cash flow.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_free_cash_flow_operating_cash_flow_ratio()
        ```
        """
        free_cash_flow_operating_cash_flow_ratio = (
            profitability.get_free_cash_flow_operating_cash_flow_ratio(
                self._cash_flow_statement.loc[:, "Free Cash Flow", :],
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            )
        )

        if growth:
            return calculate_growth(
                free_cash_flow_operating_cash_flow_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return free_cash_flow_operating_cash_flow_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_tax_burden_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the tax burden ratio, which is the ratio of a company's
        net income to its income before tax, indicating how much of a
        company's income is retained after taxes.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_tax_burden_ratio()
        ```
        """
        tax_burden_ratio = profitability.get_tax_burden_ratio(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Income Before Tax", :],
        )

        if growth:
            return calculate_growth(
                tax_burden_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return tax_burden_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_EBT_to_EBIT(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the EBT to EBIT, which is the ratio of a company's earnings before tax to its
        earnings before interest and taxes, indicating how much of a company's earnings are
        generated before paying interest on debt.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_EBT_to_EBIT()
        ```
        """
        EBT_to_EBIT = profitability.get_EBT_to_EBIT(
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :],
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :]
            + self._income_statement.loc[:, "Interest Expense", :],
        )

        if growth:
            return calculate_growth(
                EBT_to_EBIT, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return EBT_to_EBIT.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_EBIT_to_revenue(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the EBIT per Revenue, which is the ratio of a company's earnings
        before interest and taxes to its revenue, indicating how much profit a
        company generates from its operations before paying interest on debt
        and taxes on its earnings.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_EBIT_to_revenue()
        ```
        """
        EBIT_to_revenue = profitability.get_EBIT_to_revenue(
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :]
            + self._income_statement.loc[:, "Interest Expense", :],
            self._income_statement.loc[:, "Revenue", :],
        )

        if growth:
            return calculate_growth(
                EBIT_to_revenue,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return EBIT_to_revenue.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_debt_to_assets_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the debt to assets ratio, a solvency ratio that measures the proportion
        of a company's assets that are financed by debt.

        This ratio is also known as the Debt Ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_debt_to_assets_ratio()
        ```
        """
        debt_to_assets_ratio = solvency.get_debt_to_assets_ratio(
            self._balance_sheet_statement.loc[:, "Total Debt", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :],
        )

        if growth:
            return calculate_growth(
                debt_to_assets_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return debt_to_assets_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_debt_to_equity_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the debt to equity ratio, a solvency ratio that measures the
        proportion of a company's equity that is financed by debt.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_debt_to_equity_ratio()
        ```
        """
        debt_to_equity_ratio = solvency.get_debt_to_equity_ratio(
            self._balance_sheet_statement.loc[:, "Total Debt", :],
            self._balance_sheet_statement.loc[:, "Total Equity", :],
        )

        if growth:
            return calculate_growth(
                debt_to_equity_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return debt_to_equity_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_interest_coverage_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the interest coverage ratio, a solvency ratio that measures a company's
        ability to pay its interest expenses on outstanding debt.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_interest_coverage_ratio()
        ```
        """
        interest_coverage_ratio = solvency.get_interest_coverage_ratio(
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Depreciation and Amortization", :],
            self._income_statement.loc[:, "Interest Expense", :],
        )

        if growth:
            return calculate_growth(
                interest_coverage_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return interest_coverage_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_equity_multiplier(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the equity multiplier, a solvency ratio that measures the degree to which
        a company uses borrowed money (debt) to finance its operations and growth.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_equity_multiplier()
        ```
        """
        equity_multiplier = solvency.get_equity_multiplier(
            self._balance_sheet_statement.loc[:, "Total Assets", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Total Equity", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Equity", :],
        )

        if growth:
            return calculate_growth(
                equity_multiplier,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return equity_multiplier.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_debt_service_coverage_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the debt service coverage ratio, a solvency ratio that measures a company's
        ability to service its debt with its net operating income.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_debt_service_coverage_ratio()
        ```
        """
        debt_service_coverage_ratio = solvency.get_debt_service_coverage_ratio(
            self._income_statement.loc[:, "Operating Income", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if growth:
            return calculate_growth(
                debt_service_coverage_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return debt_service_coverage_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_free_cash_flow_yield(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates the free cash flow yield ratio, which measures the free cash flow
        relative to the market capitalization of the company.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_free_cash_flow_yield()
        ```
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

        free_cash_flow_yield = solvency.get_free_cash_flow_yield(
            self._cash_flow_statement.loc[:, "Free Cash Flow", :],
            market_cap,
        )

        if growth:
            return calculate_growth(
                free_cash_flow_yield,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return free_cash_flow_yield.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_net_debt_to_ebitda_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculates the net debt to EBITDA ratio, which measures the net debt of the company
        relative to its EBITDA.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_net_debt_to_ebitda_ratio()
        ```
        """
        net_debt_to_ebitda_ratio = solvency.get_net_debt_to_ebitda_ratio(
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Depreciation and Amortization", :],
            self._balance_sheet_statement.loc[:, "Net Debt", :],
        )

        if growth:
            return calculate_growth(
                net_debt_to_ebitda_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return net_debt_to_ebitda_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_cash_flow_coverage_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the cash flow coverage ratio, a solvency ratio that measures a company's
        ability to pay off its debt with its operating cash flow.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_cash_flow_coverage_ratio()
        ```
        """
        cash_flow_coverage_ratio = solvency.get_cash_flow_coverage_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._balance_sheet_statement.loc[:, "Total Debt", :],
        )

        if growth:
            return calculate_growth(
                cash_flow_coverage_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return cash_flow_coverage_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_capex_coverage_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the capital expenditure coverage ratio, a solvency ratio that
        measures a company's ability to cover its capital expenditures with its
        cash flow from operations.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_capex_coverage_ratio()
        ```
        """
        capex_coverage_ratio = solvency.get_capex_coverage_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._cash_flow_statement.loc[:, "Capital Expenditure", :],
        )

        if growth:
            return calculate_growth(
                capex_coverage_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return capex_coverage_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_capex_dividend_coverage_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the dividend paid and capex coverage ratio, a solvency ratio that
        measures a company's ability to cover both its capital expenditures and
        dividend payments with its cash flow from operations.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_capex_dividend_coverage_ratio()
        ```
        """
        dividend_capex_coverage_ratio = solvency.get_dividend_capex_coverage_ratio(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
            self._cash_flow_statement.loc[:, "Capital Expenditure", :],
            self._cash_flow_statement.loc[:, "Dividends Paid", :],
        )

        if growth:
            return calculate_growth(
                dividend_capex_coverage_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return dividend_capex_coverage_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_earnings_per_share(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the earnings per share (EPS), a valuation ratio that measures the
        amount of net income earned per share of outstanding common stock.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_earnings_per_share()
        ```
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

        earnings_per_share = valuation.get_earnings_per_share(
            self._income_statement.loc[:, "Net Income", :], dividends, average_shares
        )

        if growth:
            return calculate_growth(
                earnings_per_share,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return earnings_per_share.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_earnings_per_share_growth(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the earnings per share growth.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_earnings_per_share_growth()
        ```
        """
        eps = self.get_earnings_per_share(include_dividends, diluted)

        earnings_per_share_growth = valuation.get_earnings_per_share_growth(eps)

        if growth:
            return calculate_growth(
                earnings_per_share_growth,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return earnings_per_share_growth.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_revenue_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the revenue per share, a valuation ratio that measures the amount
        of revenue generated per outstanding share of a company's stock.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_revenue_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        revenue_per_share = valuation.get_revenue_per_share(
            self._income_statement.loc[:, "Revenue", :], average_shares
        )

        if growth:
            return calculate_growth(
                revenue_per_share,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return revenue_per_share.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_price_earnings_ratio(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the price earnings ratio (P/E), a valuation ratio that compares a
        company's stock price to its earnings per share.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_price_earnings_ratio()
        ```
        """
        eps = self.get_earnings_per_share(include_dividends, diluted)

        years = eps.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        price_earnings_ratio = valuation.get_price_earnings_ratio(share_prices, eps)

        if growth:
            return calculate_growth(
                price_earnings_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return price_earnings_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_price_to_earnings_growth_ratio(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the price earnings to growth (PEG) ratio, a valuation metric that
        measures the ratio of the price-to-earnings ratio to earnings growth rate.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_price_to_earnings_growth_ratio()
        ```
        """
        eps_growth = self.get_earnings_per_share(include_dividends, diluted)
        price_earnings = self.get_price_earnings_ratio(include_dividends, diluted)

        price_to_earnings_growth_ratio = valuation.get_price_to_earnings_growth_ratio(
            price_earnings, eps_growth
        )

        if growth:
            return calculate_growth(
                price_to_earnings_growth_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return price_to_earnings_growth_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_book_value_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the book value per share, a valuation ratio that measures the
        amount of common equity value per share outstanding.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_book_value_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        book_value_per_share = valuation.get_book_value_per_share(
            self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :],
            self._balance_sheet_statement.loc[:, "Preferred Stock", :],
            average_shares,
        )

        if growth:
            return calculate_growth(
                book_value_per_share,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return book_value_per_share.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_price_to_book_ratio(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the price to book ratio, a valuation ratio that compares a
        company's market price to its book value per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_price_to_book_ratio()
        ```
        """
        book_value_per_share = self.get_book_value_per_share(diluted)

        years = book_value_per_share.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        price_to_book_ratio = valuation.get_price_to_book_ratio(
            share_prices, book_value_per_share
        )

        if growth:
            return calculate_growth(
                price_to_book_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return price_to_book_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_interest_debt_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the interest debt per share, a valuation ratio that measures the
        amount of interest expense incurred per outstanding share of a company's stock.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_interest_debt_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        interest_debt_per_share = valuation.get_interest_debt_per_share(
            self._income_statement.loc[:, "Interest Expense", :],
            self._balance_sheet_statement.loc[:, "Total Debt", :],
            average_shares,
        )

        if growth:
            return calculate_growth(
                interest_debt_per_share,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return interest_debt_per_share.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_capex_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the capex per share, a valuation ratio that measures the amount of
        capital expenditures made per outstanding share of a company's stock.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_capex_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        capex_per_share = valuation.get_capex_per_share(
            self._cash_flow_statement.loc[:, "Capital Expenditure", :], average_shares
        )

        if growth:
            return calculate_growth(
                capex_per_share,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return capex_per_share.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_dividend_yield(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the dividend yield ratio, a valuation ratio that measures the
        amount of dividends distributed per share of stock relative to the stock's price.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_dividend_yield()
        ```
        """
        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore
        dividends = self._historical_data.loc[begin:end, "Dividends"].T  # type: ignore

        dividend_yield = valuation.get_dividend_yield(
            dividends,
            share_prices,
        )

        if growth:
            return calculate_growth(
                dividend_yield,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return dividend_yield.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_weighted_dividend_yield(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the dividend yield ratio, a valuation ratio that measures the
        amount of dividends distributed per share of stock relative to the stock's price.

        This dividend yield differs from the dividend yield ratio in that it takes into account the
        (diluted) weighted average shares and actual dividends paid as found in the cash flow statement.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_weighted_dividend_yield()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        weighted_dividend_yield = valuation.get_weighted_dividend_yield(
            abs(self._cash_flow_statement.loc[:, "Dividends Paid", :]),
            average_shares,
            share_prices,
        )

        if growth:
            return calculate_growth(
                weighted_dividend_yield,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return weighted_dividend_yield.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_price_to_cash_flow_ratio(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the price to cash flow ratio, a valuation ratio that compares a
        company's market price to its operating cash flow per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_price_to_cash_flow_ratio()
        ```
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

        price_to_cash_flow_ratio = valuation.get_price_to_cash_flow_ratio(
            market_cap, self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
        )

        if growth:
            return calculate_growth(
                price_to_cash_flow_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return price_to_cash_flow_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_price_to_free_cash_flow_ratio(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the price to free cash flow ratio, a valuation ratio that compares a
        company's market price to its free cash flow per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_price_to_free_cash_flow_ratio()
        ```
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

        price_to_free_cash_flow_ratio = valuation.get_price_to_free_cash_flow_ratio(
            market_cap, self._cash_flow_statement.loc[:, "Free Cash Flow", :]
        )

        if growth:
            return calculate_growth(
                price_to_free_cash_flow_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return price_to_free_cash_flow_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_market_cap(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates the market capitalization of the company.

        Note: All the inputs must be in the same currency and unit for accurate calculations.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_market_cap()
        ```
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

        if growth:
            return calculate_growth(
                market_cap, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return market_cap.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_enterprise_value(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates the Enterprise Value (EV) of a company. The Enterprise Value (EV)
        is a measure of a company's total value, often used as a more comprehensive
        alternative to market capitalization. It is calculated as the sum of a company's
        market capitalization, outstanding debt, minority interest, and
        preferred equity, minus the cash and cash equivalents.

        Note: All the inputs must be in the same currency and unit for accurate calculations.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_enterprise_value()
        ```
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

        enterprise_value = valuation.get_enterprise_value(
            market_cap,
            self._balance_sheet_statement.loc[:, "Total Debt", :],
            self._balance_sheet_statement.loc[:, "Minority Interest", :],
            self._balance_sheet_statement.loc[:, "Preferred Stock", :],
            self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
        )

        if growth:
            return calculate_growth(
                enterprise_value,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return enterprise_value.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_ev_to_sales_ratio(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the EV to sales ratio, a valuation ratio that compares a company's
        enterprise value (EV) to its total revenue.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_ev_to_sales_ratio()
        ```
        """
        enterprise_value = self.get_enterprise_value(diluted)

        ev_to_sales_ratio = valuation.get_ev_to_sales_ratio(
            enterprise_value, self._income_statement.loc[:, "Revenue", :]
        )

        if growth:
            return calculate_growth(
                ev_to_sales_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return ev_to_sales_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_ev_to_ebitda_ratio(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates the enterprise value over EBITDA ratio, which is a valuation ratio that
        measures a company's total value (including debt and equity) relative to its EBITDA.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_ev_to_ebitda_ratio()
        ```
        """
        enterprise_value = self.get_enterprise_value(diluted)

        ev_to_ebitda_ratio = valuation.get_ev_to_ebitda_ratio(
            enterprise_value,
            self._income_statement.loc[:, "Operating Income", :],
            self._income_statement.loc[:, "Depreciation and Amortization", :],
        )

        if growth:
            return calculate_growth(
                ev_to_ebitda_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return ev_to_ebitda_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_ev_to_operating_cashflow_ratio(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates the enterprise value over operating cash flow ratio, which is a valuation
        ratio that measures a company's total value (including debt and equity) relative
        to its operating cash flow.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_ev_to_operating_cashflow_ratio()
        ```
        """
        enterprise_value = self.get_enterprise_value(diluted)

        ev_to_operating_cashflow_ratio = valuation.get_ev_to_operating_cashflow_ratio(
            enterprise_value,
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
        )

        if growth:
            return calculate_growth(
                ev_to_operating_cashflow_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return ev_to_operating_cashflow_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_earnings_yield(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates the earnings yield ratio, which measures the earnings per share
        relative to the market price per share.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_earnings_yield()
        ```
        """
        eps = self.get_earnings_per_share(include_dividends, diluted)

        years = eps.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        earnings_yield = valuation.get_earnings_yield(eps, share_prices)

        if growth:
            return calculate_growth(
                earnings_yield,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return earnings_yield.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_payout_ratio(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculates the (dividend) payout ratio, which measures the proportion of earnings
        paid out as dividends to shareholders.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_payout_ratio()
        ```
        """
        payout_ratio = valuation.get_payout_ratio(
            self._cash_flow_statement.loc[:, "Dividends Paid", :],
            self._income_statement.loc[:, "Net Income", :],
        )

        if growth:
            return calculate_growth(
                payout_ratio, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return payout_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_tangible_asset_value(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the tangible asset value, which represents the total value of a company's
        assets that can be used to generate revenue.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_tangible_asset_value()
        ```
        """
        tangible_asset_value = valuation.get_tangible_asset_value(
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Total Liabilities", :],
            self._balance_sheet_statement.loc[:, "Goodwill", :],
        )

        if growth:
            return calculate_growth(
                tangible_asset_value,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return tangible_asset_value.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_net_current_asset_value(
        self, rounding: int | None = 4, growth: bool = False, lag: int | list[int] = 1
    ):
        """
        Calculate the net current asset value, which is the total value of a company's
        current assets minus its current liabilities.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_net_current_asset_value()
        ```
        """
        net_current_asset_value = valuation.get_net_current_asset_value(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if growth:
            return calculate_growth(
                net_current_asset_value,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return net_current_asset_value.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_ev_to_ebit(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the net current asset value, which is the total value of a company's
        current assets minus its current liabilities.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.ratios.get_ev_to_ebit()
        ```
        """
        enterprise_value = self.get_enterprise_value(diluted)

        ev_to_ebit = valuation.get_ev_to_ebit(
            enterprise_value,
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :]
            + +self._income_statement.loc[:, "Interest Expense", :],
        )

        if growth:
            return calculate_growth(
                ev_to_ebit, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return ev_to_ebit.round(rounding if rounding else self._rounding)
