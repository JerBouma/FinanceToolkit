"""Ratios Module"""

__docformat__ = "google"


import warnings

import numpy as np
import pandas as pd

from financetoolkit.helpers import calculate_growth, handle_portfolio
from financetoolkit.ratios import (
    efficiency_model,
    liquidity_model,
    profitability_model,
    solvency_model,
    valuation_model,
)
from financetoolkit.ratios.helpers import map_period_data_to_daily_data
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.error_model import handle_errors

logger = logger_model.get_logger()

# Runtime errors are ignored on purpose given the nature of the calculations
# sometimes leading to division by zero or other mathematical errors. This is however
# for financial analysis purposes not an issue and should not be considered as a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-public-methods,too-many-locals,eval-used


class Ratios:
    """
    The Ratios Module contains over 50+ ratios that can be used to analyse companies.
    These ratios are divided into 5 categories which are efficiency, liquidity,
    profitability, solvency and valuation. Each ratio is calculated using
    the data from the Toolkit module.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical: dict[str, pd.DataFrame],
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cash: pd.DataFrame,
        quarterly: bool = False,
        rounding: int | None = 4,
    ):
        """
        Initializes the Ratios Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the calculations.
            historical (dict[str, pd.DataFrame]): The historical data to use for the calculations.
                Typically includes a "period" and "daily" key to access the respective data.
            balance (pd.DataFrame): The balance sheet data to use for the calculations.
            income (pd.DataFrame): The income statement data to use for the calculations.
            cash (pd.DataFrame): The cash flow statement data to use for the calculations.
            an optional parameter given that you can also define the custom ratios through the Toolkit initialization.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        profitability_ratios = toolkit.ratios.collect_profitability_ratios()

        profitability_ratios.loc['AAPL']
        ```

        Which returns:

        |                                             |     2018 |     2019 |     2020 |     2021 |     2022 |
        |:--------------------------------------------|---------:|---------:|---------:|---------:|---------:|
        | Gross Margin                                | 0.383437 | 0.378178 | 0.382332 | 0.417794 | 0.433096 |
        | Operating Margin                            | 0.26694  | 0.24572  | 0.241473 | 0.297824 | 0.302887 |
        | Net Profit Margin                           | 0.224142 | 0.212381 | 0.209136 | 0.258818 | 0.253096 |
        | Interest Burden Ratio                       | 1.02828  | 1.02827  | 1.01211  | 1.00237  | 0.997204 |
        | Income Before Tax Profit Margin             | 0.274489 | 0.252666 | 0.244398 | 0.298529 | 0.30204  |
        | Effective Tax Rate                          | 0.183422 | 0.159438 | 0.144282 | 0.133023 | 0.162045 |
        | Return on Assets (ROA)                      | 0.162775 | 0.16323  | 0.177256 | 0.269742 | 0.282924 |
        | Return on Equity (ROE)                      | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |
        | Return on Invested Capital (ROIC)           | 0.269858 | 0.293721 | 0.344126 | 0.503852 | 0.562645 |
        | Return on Capital Employed (ROCE)           | 0.305968 | 0.297739 | 0.320207 | 0.495972 | 0.613937 |
        | Return on Tangible Assets                   | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |
        | Income Quality Ratio                        | 1.30073  | 1.25581  | 1.4052   | 1.09884  | 1.22392  |
        | Net Income per EBT                          | 0.816578 | 0.840562 | 0.855718 | 0.866977 | 0.837955 |
        | Free Cash Flow to Operating Cash Flow Ratio | 0.828073 | 0.848756 | 0.909401 | 0.893452 | 0.912338 |
        | EBT to EBIT Ratio                           | 0.957448 | 0.948408 | 0.958936 | 0.976353 | 0.975982 |
        | EBIT to Revenue                             | 0.286688 | 0.26641  | 0.254864 | 0.305759 | 0.309473 |
        """
        self._tickers = tickers
        self._tickers_without_portfolio = [
            ticker for ticker in tickers if ticker != "Portfolio"
        ]
        self._balance_sheet_statement: pd.DataFrame = balance
        self._income_statement: pd.DataFrame = income
        self._cash_flow_statement: pd.DataFrame = cash
        self._available_custom_ratios_options: list[str] = []
        self._custom_ratios: pd.DataFrame = pd.DataFrame()
        self._custom_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._rounding: int | None = rounding
        self._quarterly: bool = quarterly
        self._portfolio_weights: dict | None = None

        # Initialization of Historical Data
        self._historical_data: pd.DataFrame = historical["period"]
        self._daily_historical_data: pd.DataFrame = historical["daily"]

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
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects all ratios based on the provided data.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculations.
                Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for the calculation.
                Defaults to True.
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.Series or pd.DataFrame: Ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.collect_all_ratios()
        ```
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        if self._efficiency_ratios.empty:
            self.collect_efficiency_ratios(days=days, trailing=trailing)
        if self._liquidity_ratios.empty:
            self.collect_liquidity_ratios(trailing=trailing)
        if self._profitability_ratios.empty:
            self.collect_profitability_ratios(trailing=trailing)
        if self._solvency_ratios.empty:
            self.collect_solvency_ratios(diluted=diluted, trailing=trailing)
        if self._valuation_ratios.empty:
            self.collect_valuation_ratios(
                include_dividends=include_dividends, diluted=diluted, trailing=trailing
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

        self._all_ratios = self._all_ratios.round(
            rounding if rounding else self._rounding
        )

        # In case sorting accidentally fails, the index is sorted again
        # to follow the same order as the financial statements
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._all_ratios
        ]
        self._all_ratios = self._all_ratios.reindex(available_columns, axis=1)

        if growth:
            self._all_ratios_growth = calculate_growth(
                self._all_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._all_ratios_growth.loc[self._tickers[0]]
                if growth
                else self._all_ratios.loc[self._tickers[0]]
            )

        return self._all_ratios_growth if growth else self._all_ratios

    def collect_custom_ratios(
        self,
        custom_ratios_dict: dict | None = None,
        options: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculates all Custom Ratios based on the data provided.

        Note that any of the following characters are considered as operators:
            +, -, *, /, **, %, //, <, >, ==, !=, >=, <=, (, )
        using any of the above characters as part of the column naming will result into an error.

        Args:
            custom_ratios (dict): A dictionary containing the custom ratios to calculate.
            options (bool): Whether to return the available names to use in the custom ratios.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Custom ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various custom ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

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
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-10-01",
            quarterly=True
        )

        custom_ratios = companies.ratios.collect_custom_ratios(
            custom_ratios_dict=custom_ratios
        )

        custom_ratios.loc['AMZN']
        ```

        Which returns:

        |                        |         2022Q4 |         2023Q1 |         2023Q2 |         2023Q3 |
        |:-----------------------|---------------:|---------------:|---------------:|---------------:|
        | WC / Net Income as %   |  463.349       |  427.335       |  398.924       |  371.423       |
        | Large Revenues         |    1           |    1           |    1           |    1           |
        | Quick Assets           |    1.35341e+11 |    1.41847e+11 |    1.5995e+11  |    1.80898e+11 |
        | Cash Op Expenses       |    2.1056e+10  |    1.9972e+10  |    2.2854e+10  |    1.9042e+10  |
        | Daily Cash Op Expenses |    5.76877e+07 |    5.47178e+07 |    6.26137e+07 |    5.21699e+07 |
        | Defensive Interval     | 2346.1         | 2592.34        | 2554.55        | 3467.48        |
        """
        if self._all_ratios.empty:
            self.collect_all_ratios()

        if not custom_ratios_dict and not options:
            logger.error(
                "Please define custom ratios dictionary to the custom_ratios_dict parameter. See "
                "https://www.jeroenbouma.com/projects/financetoolkit/custom-ratios how to do this."
            )
            return None

        if options:
            logger.info(
                "The following names are available to be used in the Custom Ratios calculations."
            )

            self._available_custom_ratios_options = list(
                set(
                    list(self._balance_sheet_statement.index.get_level_values(level=1))
                    + list(self._income_statement.index.get_level_values(level=1))
                    + list(self._cash_flow_statement.index.get_level_values(level=1))
                    + list(self._all_ratios.index.get_level_values(level=1))
                )
            )

            self._available_custom_ratios_options.sort()

            return self._available_custom_ratios_options

        custom_ratios = pd.DataFrame(
            0,
            index=pd.MultiIndex.from_product(
                [self._tickers, custom_ratios_dict.keys()]  # type: ignore
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
        for name, formula in custom_ratios_dict.items():  # type: ignore
            # Rearrange the formula dict in case a formula is dependent on another formula
            # and the order would result into errors
            for sub_name, sub_formula in custom_ratios_dict.items():  # type: ignore
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
                        logger.error(
                            "Column %s not found in total_financials and is not a number. "
                            "Therefore the formula %s is invalid. Use collect_custom_ratios(options=True) "
                            "to see the available columns.",
                            formula_section_stripped,
                            formula,
                        )
                        break

            if formula_adjusted:
                calculation = eval(formula_adjusted)  # noqa

                total_financials.loc[:, name, :] = calculation.astype(
                    np.float64
                ).to_numpy()

                self._custom_ratios = total_financials.loc[
                    :, list(custom_ratios_dict.keys()), :  # type: ignore
                ]
                self._custom_ratios = self._custom_ratios.sort_index(
                    axis=0, level=0, sort_remaining=False
                )

                self._custom_ratios = self._custom_ratios.round(
                    rounding if rounding else self._rounding
                )

        if growth:
            self._custom_ratios_growth = calculate_growth(
                self._custom_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._custom_ratios_growth[self._tickers[0]]
                if growth
                else self._custom_ratios.loc[self._tickers[0]]
            )

        return self._custom_ratios_growth if growth else self._custom_ratios

    @handle_errors
    def collect_efficiency_ratios(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects all Efficiency Ratios based on the provided data.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.Series or pd.DataFrame: Efficiency ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various efficiency ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.collect_efficiency_ratios()
        ```
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        efficiency_ratios: dict = {}

        efficiency_ratios["Days of Inventory Outstanding"] = (
            self.get_days_of_inventory_outstanding(days=days, trailing=trailing)
        )
        efficiency_ratios["Days of Sales Outstanding"] = (
            self.get_days_of_sales_outstanding(days=days, trailing=trailing)
        )
        efficiency_ratios["Operating Cycle"] = self.get_operating_cycle(
            trailing=trailing
        )
        efficiency_ratios["Days of Accounts Payable Outstanding"] = (
            self.get_days_of_accounts_payable_outstanding(days=days, trailing=trailing)
        )
        efficiency_ratios["Cash Conversion Cycle"] = self.get_cash_conversion_cycle(
            days=days, trailing=trailing
        )
        efficiency_ratios["Cash Conversion Efficiency"] = (
            self.get_cash_conversion_efficiency(trailing=trailing)
        )
        efficiency_ratios["Receivables Turnover"] = self.get_receivables_turnover(
            trailing=trailing
        )
        efficiency_ratios["Inventory Turnover Ratio"] = (
            self.get_inventory_turnover_ratio()
        )
        efficiency_ratios["Accounts Payable Turnover Ratio"] = (
            self.get_accounts_payables_turnover_ratio()
        )
        efficiency_ratios["SGA-to-Revenue Ratio"] = self.get_sga_to_revenue_ratio(
            trailing=trailing
        )
        efficiency_ratios["Fixed Asset Turnover"] = self.get_fixed_asset_turnover(
            trailing=trailing
        )
        efficiency_ratios["Asset Turnover Ratio"] = self.get_asset_turnover_ratio(
            trailing=trailing
        )
        efficiency_ratios["Operating Ratio"] = self.get_operating_ratio(
            trailing=trailing
        )

        self._efficiency_ratios = (
            pd.concat(efficiency_ratios)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        # Ensure the ticker order remains the same as in self._tickers
        self._efficiency_ratios = self._efficiency_ratios.loc[self._tickers]

        self._efficiency_ratios = self._efficiency_ratios.round(
            rounding if rounding else self._rounding
        )

        # In case sorting accidentally fails, the index is sorted again
        # to follow the same order as the financial statements
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._efficiency_ratios
        ]
        self._efficiency_ratios = self._efficiency_ratios.reindex(
            available_columns, axis=1
        )

        if growth:
            self._efficiency_ratios_growth = calculate_growth(
                self._efficiency_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._efficiency_ratios_growth[self._tickers[0]]
                if growth
                else self._efficiency_ratios.loc[self._tickers[0]]
            )

        return self._efficiency_ratios_growth if growth else self._efficiency_ratios

    @handle_portfolio
    @handle_errors
    def get_asset_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        The asset turnover ratio is calculated by dividing the company's net sales
        (revenue) by its average total assets. It measures how well a company utilizes
        its assets to generate revenue. A higher asset turnover ratio indicates that the
        company is generating more revenue per unit of assets, which is generally seen
        as a positive sign of operational efficiency.

        The formula is as follows:

        - Asset Turnover Ratio = Net Sales / Average Total Assets

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.Series: Asset turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the asset turnover ratio
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        asset_turnover_ratios = toolkit.ratios.get_asset_turnover_ratio()
        ```
        """
        if trailing:
            asset_turnover_ratio = efficiency_model.get_asset_turnover_ratio(
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            asset_turnover_ratio = efficiency_model.get_asset_turnover_ratio(
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        if growth:
            return calculate_growth(
                asset_turnover_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return asset_turnover_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_inventory_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the inventory turnover ratio, an efficiency ratio that measures
        how quickly a company sells its inventory.

        The inventory turnover ratio is calculated by dividing the cost of goods sold
        (COGS) by the average inventory value. It indicates how many times a company's
        inventory is sold and replaced over a period. A higher inventory turnover ratio
        suggests that a company is effectively managing its inventory by quickly
        converting it into sales.

        The formula is as follows:

        - Inventory Turnover Ratio = Cost of Goods Sold / Average Inventory

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.Series: Inventory turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the inventory turnover ratio
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        inventory_turnover_ratios = toolkit.ratios.get_inventory_turnover_ratio()
        ```
        """
        if trailing:
            inventory_turnover_ratio = efficiency_model.get_inventory_turnover_ratio(
                self._income_statement.loc[:, "Cost of Goods Sold", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            inventory_turnover_ratio = efficiency_model.get_inventory_turnover_ratio(
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        if growth:
            return calculate_growth(
                inventory_turnover_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return inventory_turnover_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_days_of_inventory_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the days sales in inventory ratio, an efficiency ratio that measures
        how long it takes a company to sell its inventory.

        The days sales in inventory ratio (DSI) is calculated by dividing the average
        inventory by the cost of goods sold (COGS) and then multiplying by the number
        of days in the period. It represents the average number of days it takes for
        a company to sell its inventory. A lower DSI indicates that the company is
        selling its inventory more quickly.

        The formula is as follows:

        - Days Sales in Inventory Ratio = (Average Inventory / Cost of Goods Sold) * Days

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Days sales in inventory ratio values.

        Notes:
        - The method retrieves historical data and calculates the DSI ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.get_days_of_inventory_outstanding()
        ```
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        if trailing:
            days_of_inventory_outstanding = (
                efficiency_model.get_days_of_inventory_outstanding(
                    self._balance_sheet_statement.loc[:, "Inventory", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                    self._income_statement.loc[:, "Cost of Goods Sold", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            days_of_inventory_outstanding = (
                efficiency_model.get_days_of_inventory_outstanding(
                    self._balance_sheet_statement.loc[:, "Inventory", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    self._income_statement.loc[:, "Cost of Goods Sold", :],
                    days,
                )
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

    @handle_portfolio
    @handle_errors
    def get_days_of_sales_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the days of sales outstanding ratio, an efficiency ratio that measures
        the average number of days it takes a company to collect payment on its
        credit sales.

        The days of sales outstanding (DSO) ratio is calculated by dividing the accounts
        receivable by the total credit sales and then multiplying by the number of days
        in the period. It represents the average number of days it takes for a company
        to collect payment on its credit sales. A lower DSO indicates that the company
        is collecting payments more quickly.

        The formula is as follows:

        - Days of Sales Outstanding Ratio = (Accounts Receivable / Total Credit Sales) * Days

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Days of sales outstanding ratio values.

        Notes:
        - The method retrieves historical data and calculates the DSO ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        dso_ratios = toolkit.ratios.get_days_of_sales_outstanding()
        ```
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        if trailing:
            days_of_sales_outstanding = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            days_of_sales_outstanding = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(2)
                .mean()
                .T,
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

    @handle_portfolio
    @handle_errors
    def get_operating_cycle(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating cycle ratio, an efficiency ratio that measures the average
        number of days it takes a company to turn its inventory into cash.

        The operating cycle represents the total time required to purchase inventory,
        convert it into finished goods, sell the goods to customers, and collect the
        accounts receivable. It is calculated by adding the days sales in inventory (DSI)
        and the days of sales outstanding (DSO).

        The formula is as follows:

        - Operating Cycle Ratio = Days of Sales in Inventory + Days of Sales Outstanding

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Operating cycle ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating cycle ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        operating_cycle_ratios = toolkit.ratios.get_operating_cycle()
        ```
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        if trailing:
            days_of_inventory = efficiency_model.get_days_of_inventory_outstanding(
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._income_statement.loc[:, "Cost of Goods Sold", :]
                .T.rolling(trailing)
                .sum()
                .T,
                days,
            )

            days_of_sales = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .shift(axis=1)
                .T.rolling(trailing)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
                days,
            )
        else:
            days_of_inventory = efficiency_model.get_days_of_inventory_outstanding(
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                days,
            )
            days_of_sales = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :],
                days,
            )

        operating_cycle = efficiency_model.get_operating_cycle(
            days_of_inventory, days_of_sales
        )

        if growth:
            return calculate_growth(
                operating_cycle,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return operating_cycle.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_accounts_payables_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the accounts payable turnover ratio, an efficiency ratio that measures how
        quickly a company pays its suppliers.

        The accounts payable turnover ratio indicates how many times, on average, a company
        pays off its accounts payable during a specific period. A higher turnover ratio is
        generally favorable, as it suggests that the company is efficiently managing its
        payments to suppliers.

        The formula is as follows:

        - Accounts Payable Turnover Ratio = Cost of Goods Sold / Average Accounts Payable

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Accounts payable turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the accounts payable turnover ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ap_turnover_ratios = toolkit.ratios.get_accounts_payables_turnover_ratio()
        ```
        """
        if trailing:
            accounts_payables_turnover_ratio = (
                efficiency_model.get_accounts_payables_turnover_ratio(
                    self._income_statement.loc[:, "Cost of Goods Sold", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                )
            )
        else:
            accounts_payables_turnover_ratio = (
                efficiency_model.get_accounts_payables_turnover_ratio(
                    self._income_statement.loc[:, "Cost of Goods Sold", :],
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(2)
                    .mean()
                    .T,
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

    @handle_portfolio
    @handle_errors
    def get_days_of_accounts_payable_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the days payables outstanding, an efficiency ratio that measures the
        number of days it takes a company to pay its suppliers.

        The days payables outstanding (DPO) ratio is used to assess how efficiently a company
        manages its accounts payable. It calculates the average number of days it takes for
        a company to pay its suppliers after receiving an invoice. A higher DPO ratio indicates
        that the company is taking longer to pay its suppliers, which may have implications for
        its relationships with suppliers.

        The formula is as follows:

        - Days Payables Outstanding = (Average Accounts Payable / Cost of Goods Sold) * Days

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Days payables outstanding (DPO) ratio values.

        Notes:
        - The method retrieves historical data and calculates the DPO ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        dpo_ratios = toolkit.ratios.get_days_of_accounts_payable_outstanding()
        ```
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        if trailing:
            days_of_accounts_payable_outstanding = (
                efficiency_model.get_days_of_accounts_payable_outstanding(
                    self._income_statement.loc[:, "Cost of Goods Sold", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                )
            )
        else:
            days_of_accounts_payable_outstanding = (
                efficiency_model.get_days_of_accounts_payable_outstanding(
                    self._income_statement.loc[:, "Cost of Goods Sold", :],
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(2)
                    .mean()
                    .T,
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

    @handle_portfolio
    @handle_errors
    def get_cash_conversion_cycle(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Cash Conversion Cycle, which measures the amount of time it takes for a company to convert
        its investments in inventory and accounts receivable into cash, while considering the time it takes to pay
        its accounts payable. This ratio is also known as Cash-to-Cash Cycle (C2C) or Net Operating Cycle.

        The Cash Conversion Cycle (CCC) is an important measure of a company's liquidity management and efficiency
        in managing its working capital. It takes into account the time it takes to sell inventory, collect payments
        from customers, and pay suppliers. A shorter CCC indicates that a company is able to quickly convert its
        investments into cash, which can be a positive sign of efficient operations.

        The formula is as follows:

        - Cash Conversion Cycle = Days of Sales in Inventory + Days of Sales Outstanding - Days of Accounts Payable
            Outstanding

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Cash Conversion Cycle (CCC) values.

        Notes:
        - The method retrieves historical data and calculates the CCC for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the CCC values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ccc_values = toolkit.ratios.get_cash_conversion_cycle()
        ```
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        if trailing:
            days_of_inventory = efficiency_model.get_days_of_inventory_outstanding(
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._income_statement.loc[:, "Cost of Goods Sold", :]
                .T.rolling(trailing)
                .sum()
                .T,
                days,
            )

            days_of_sales = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
                days,
            )

            days_of_payables = (
                efficiency_model.get_days_of_accounts_payable_outstanding(
                    self._income_statement.loc[:, "Cost of Goods Sold", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                    days,
                )
            )
        else:
            days_of_inventory = efficiency_model.get_days_of_inventory_outstanding(
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                days,
            )
            days_of_sales = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :],
                days,
            )

            days_of_payables = (
                efficiency_model.get_days_of_accounts_payable_outstanding(
                    self._income_statement.loc[:, "Cost of Goods Sold", :],
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    days,
                )
            )

        cash_conversion_cycle = efficiency_model.get_cash_conversion_cycle(
            days_of_inventory, days_of_sales, days_of_payables
        )

        if growth:
            return calculate_growth(
                cash_conversion_cycle,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return cash_conversion_cycle.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_cash_conversion_efficiency(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash conversion efficiency, an efficiency ratio that measures how
        efficiently a company converts its sales into cash. It is also known as the cash
        conversion ratio.

        The cash conversion efficiency ratio is calculated by dividing the operating cash
        flow by the revenue. It indicates how much of a company's sales are converted into
        cash. A higher cash conversion efficiency ratio is generally favorable, as it
        suggests that the company is able to convert its sales into cash more efficiently.

        The formula is as follows:

        - Cash Conversion Efficiency Ratio = Operating Cash Flow / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Cash conversion efficiency ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.get_cash_conversion_efficiency()
        ```
        """
        if trailing:
            cash_conversion_efficiency = (
                efficiency_model.get_cash_conversion_efficiency(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._income_statement.loc[:, "Revenue", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            cash_conversion_efficiency = (
                efficiency_model.get_cash_conversion_efficiency(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    self._income_statement.loc[:, "Revenue", :],
                )
            )

        if growth:
            return calculate_growth(
                cash_conversion_efficiency,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return cash_conversion_efficiency.round(
            rounding if rounding else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_receivables_turnover(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the receivables turnover, a ratio that measures how efficiently a
        company uses its assets by comparing the amount of credit extended to customers to
        the amount of sales generated.

        The receivables turnover ratio is an important measure of how well a company manages its
        accounts receivable. It indicates how quickly a company collects payments from its customers.
        A higher turnover ratio is generally favorable as it suggests that the company is collecting
        payments more quickly, which improves its cash flow and working capital management.

        The formula is as follows:

        - Receivables Turnover Ratio = Net Credit Sales / Average Accounts Receivable

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.
        Returns:
            pd.DataFrame: Receivables turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the receivables turnover ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        receivables_turnover = toolkit.ratios.get_receivables_turnover()
        ```
        """
        if trailing:
            receivables_turnover = efficiency_model.get_receivables_turnover(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            receivables_turnover = efficiency_model.get_receivables_turnover(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :],
            )

        if growth:
            return calculate_growth(
                receivables_turnover,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return receivables_turnover.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_sga_to_revenue_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the sales, general, and administrative (SG&A) expenses to revenue ratio,
        which measures the SG&A expenses relative to the revenue of the company.

        The SG&A to revenue ratio is calculated by dividing the total SG&A expenses by the
        company's revenue and then multiplying by 100 to express it as a percentage. It
        provides insight into the efficiency of a company's cost management and its ability
        to control its overhead costs.

        The formula is as follows:

        - SG&A to Revenue Ratio = SG&A Expenses / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: SG&A to revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the SG&A to revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        sga_to_revenue_ratios = toolkit.ratios.get_sga_to_revenue_ratio()
        ```
        """
        if trailing:
            sga_to_revenue_ratio = efficiency_model.get_sga_to_revenue_ratio(
                self._income_statement.loc[
                    :, "Selling, General and Administrative Expenses", :
                ]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            sga_to_revenue_ratio = efficiency_model.get_sga_to_revenue_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_fixed_asset_turnover(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Fixed Asset Turnover ratio, an efficiency ratio that
        measures how efficiently a company uses its fixed assets to generate sales.

        The Fixed Asset Turnover ratio is calculated by dividing the company's net sales
        by the average fixed assets. It indicates how well a company is utilizing its fixed
        assets to generate revenue. A higher ratio suggests more efficient utilization of
        fixed assets.

        The formula is as follows:

        - Fixed Asset Turnover Ratio = Net Sales / Average Fixed Assets

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Fixed Asset Turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the Fixed Asset Turnover ratio
        for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        fixed_asset_turnover_ratios = toolkit.ratios.get_fixed_asset_turnover()
        ```
        """
        if trailing:
            fixed_asset_turnover = efficiency_model.get_fixed_asset_turnover(
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
                self._balance_sheet_statement.loc[:, "Fixed Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            fixed_asset_turnover = efficiency_model.get_fixed_asset_turnover(
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Fixed Assets", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        if growth:
            return calculate_growth(
                fixed_asset_turnover,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return fixed_asset_turnover.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_operating_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating ratio, a financial metric that measures the efficiency
        of a company's operations by comparing its operating expenses to its revenue.

        The operating ratio is calculated by dividing the company's operating expenses by
        its net sales and multiplying by 100 to express it as a percentage. It provides
        insight into how efficiently a company is managing its operations.

        The formula is as follows:

        - Operating Ratio = (Operating Expenses + Cost of Goods Sold) / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Operating ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        operating_ratios = toolkit.ratios.get_operating_ratio()
        ```
        """
        if trailing:
            operating_ratio = efficiency_model.get_operating_ratio(
                self._income_statement.loc[:, "Operating Expenses", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Cost of Goods Sold", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            operating_ratio = efficiency_model.get_operating_ratio(
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

    def collect_liquidity_ratios(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Liquidity Ratios based on the provided data.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Liquidity ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various liquidity ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        liquidity_ratios = toolkit.ratios.collect_liquidity_ratios()
        ```
        """
        liquidity_ratios: dict = {}

        liquidity_ratios["Current Ratio"] = self.get_current_ratio(trailing=trailing)
        liquidity_ratios["Quick Ratio"] = self.get_quick_ratio(trailing=trailing)
        liquidity_ratios["Cash Ratio"] = self.get_cash_ratio(trailing=trailing)
        liquidity_ratios["Working Capital"] = self.get_working_capital(
            trailing=trailing
        )
        liquidity_ratios["Operating Cash Flow Ratio"] = (
            self.get_operating_cash_flow_ratio(trailing=trailing)
        )
        liquidity_ratios["Operating Cash Flow to Sales Ratio"] = (
            self.get_operating_cash_flow_sales_ratio(trailing=trailing)
        )
        liquidity_ratios["Short Term Coverage Ratio"] = (
            self.get_short_term_coverage_ratio(trailing=trailing)
        )

        self._liquidity_ratios = (
            pd.concat(liquidity_ratios)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._liquidity_ratios = self._liquidity_ratios.loc[self._tickers]

        self._liquidity_ratios = self._liquidity_ratios.round(
            rounding if rounding else self._rounding
        )

        # In case sorting accidentally fails, the index is sorted again
        # to follow the same order as the financial statements
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._liquidity_ratios
        ]
        self._liquidity_ratios = self._liquidity_ratios.reindex(
            available_columns, axis=1
        )

        if growth:
            self._liquidity_ratios_growth = calculate_growth(
                self._liquidity_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._liquidity_ratios_growth[self._tickers[0]]
                if growth
                else self._liquidity_ratios.loc[self._tickers[0]]
            )

        return self._liquidity_ratios_growth if growth else self._liquidity_ratios

    @handle_portfolio
    @handle_errors
    def get_current_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the current ratio, a liquidity ratio that measures a company's ability
        to pay off its short-term liabilities with its current assets.

        The current ratio is calculated by dividing a company's current assets by its
        current liabilities. It indicates whether a company can meet its short-term
        obligations using its short-term assets.

        The formula is as follows:

        - Current Ratio = Current Assets / Current Liabilities

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Current ratio values.

        Notes:
        - The method retrieves historical data and calculates the current ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        current_ratios = toolkit.ratios.get_current_ratio()
        ```
        """
        if trailing:
            current_ratio = liquidity_model.get_current_ratio(
                self._balance_sheet_statement.loc[:, "Total Current Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            current_ratio = liquidity_model.get_current_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_quick_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the quick ratio (also known as the acid-test ratio), a more stringent
        measure of liquidity that excludes inventory from current assets.

        This ratio is also referred to as the Acid Test Ratio.

        The quick ratio is calculated by subtracting inventory from current assets and
        then dividing the result by current liabilities. It provides insight into a
        company's ability to cover its short-term liabilities using its most liquid
        assets without relying on inventory.

        The formula is as follows:

        - Quick Ratio = (Cash and Cash Equivalents + Short Term Investments + Accounts Receivable) / Current Liabilities

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Quick ratio values.

        Notes:
        - The method retrieves historical data and calculates the quick ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit
        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        quick_ratios = toolkit.ratios.get_quick_ratio()
        ```
        """
        if trailing:
            quick_ratio = liquidity_model.get_quick_ratio(
                self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Short Term Investments", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            quick_ratio = liquidity_model.get_quick_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_cash_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash ratio, a liquidity ratio that measures a company's ability
        to pay off its short-term liabilities with its cash and cash equivalents.

        The cash ratio is calculated by dividing the sum of cash and cash equivalents
        by current liabilities. It provides insight into a company's immediate ability
        to cover its short-term obligations using its most liquid assets.

        The formula is as follows:

        - Cash Ratio = (Cash and Cash Equivalents + Short Term Investments) / Current Liabilities

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Cash ratio values.

        Notes:
        - The method retrieves historical data and calculates the cash ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        cash_ratios = toolkit.ratios.get_cash_ratio()
        ```
        """
        if trailing:
            cash_ratio = liquidity_model.get_cash_ratio(
                self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Short Term Investments", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            cash_ratio = liquidity_model.get_cash_ratio(
                self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
                self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
            )

        if growth:
            return calculate_growth(
                cash_ratio, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return cash_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_working_capital(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the working capital, which is the difference between a company's current assets
        and current liabilities.

        The working capital is calculated by subtracting total current liabilities from total
        current assets. It represents the company's short-term financial health and its ability
        to cover its current obligations using its liquid assets.

        The formula is as follows:

        - Working Capital = Current Assets — Current Liabilities

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Working capital values.

        Notes:
        - The method retrieves historical data and calculates the working capital for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the working capital
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        working_capitals = toolkit.ratios.get_working_capital()
        ```
        """
        if trailing:
            working_capital = liquidity_model.get_working_capital(
                self._balance_sheet_statement.loc[:, "Total Current Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            working_capital = liquidity_model.get_working_capital(
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

    @handle_portfolio
    @handle_errors
    def get_operating_cash_flow_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating cash flow ratio, a liquidity ratio that measures a company's
        ability to pay off its current liabilities with its operating cash flow.

        The operating cash flow ratio is calculated by dividing operating cash flow by
        current liabilities. It indicates whether a company's operating cash flow is
        sufficient to cover its short-term obligations.

        The formula is as follows:

        - Operating Cash Flow Ratio = Cash Flow from Operations / Current Liabilities

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Operating cash flow ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating cash flow ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        operating_cash_flow_ratios = toolkit.ratios.get_operating_cash_flow_ratio()
        ```
        """
        if trailing:
            operating_cash_flow_ratio = liquidity_model.get_operating_cash_flow_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            operating_cash_flow_ratio = liquidity_model.get_operating_cash_flow_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_operating_cash_flow_sales_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating cash flow to sales ratio, a liquidity ratio that
        measures the ability of a company to generate cash from its sales.

        The operating cash flow to sales ratio is calculated by dividing operating cash flow by
        sales revenue. It indicates the proportion of sales revenue that is converted into cash
        from operating activities.

        The formula is as follows:

        - Operating Cash Flow to Sales Ratio = Cash Flow from Operations / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Operating cash flow to sales ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating cash flow to sales ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        operating_cash_flow_sales_ratios = toolkit.ratios.get_operating_cash_flow_sales_ratio()
        ```
        """
        if trailing:
            operating_cash_flow_sales_ratio = (
                liquidity_model.get_operating_cash_flow_sales_ratio(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._income_statement.loc[:, "Revenue", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            operating_cash_flow_sales_ratio = (
                liquidity_model.get_operating_cash_flow_sales_ratio(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    self._income_statement.loc[:, "Revenue", :],
                )
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

    @handle_portfolio
    @handle_errors
    def get_short_term_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the short-term coverage ratio, a liquidity ratio that measures a company's
        ability to pay off its short-term obligations with its operating cash flow.

        The short-term coverage ratio is calculated by dividing operating cash flow by short-term debt.
        It assesses the company's ability to meet its short-term obligations using its operating cash flow.

        The formula is as follows:

        - Short Term Coverage Ratio = Cash Flow from Operations / (Accounts Receivable + Inventory — Accounts Payable)

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.get_short_term_coverage_ratio()
        ```
        """
        if trailing:
            short_term_coverage_ratio = liquidity_model.get_short_term_coverage_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            short_term_coverage_ratio = liquidity_model.get_short_term_coverage_ratio(
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

    def collect_profitability_ratios(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Profitability Ratios based on the provided data.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Profitability ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various profitability ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        profitability_ratios = toolkit.ratios.collect_profitability_ratios()
        ```
        """
        profitability_ratios: dict = {}

        profitability_ratios["Gross Margin"] = self.get_gross_margin(trailing=trailing)
        profitability_ratios["Operating Margin"] = self.get_operating_margin(
            trailing=trailing
        )
        profitability_ratios["Net Profit Margin"] = self.get_net_profit_margin(
            trailing=trailing
        )
        profitability_ratios["Interest Coverage Ratio"] = (
            self.get_interest_coverage_ratio(trailing=trailing)
        )
        profitability_ratios["Income Before Tax Profit Margin"] = (
            self.get_income_before_tax_profit_margin(trailing=trailing)
        )
        profitability_ratios["Effective Tax Rate"] = self.get_effective_tax_rate(
            trailing=trailing
        )
        profitability_ratios["Return on Assets"] = self.get_return_on_assets(
            trailing=trailing
        )
        profitability_ratios["Return on Equity"] = self.get_return_on_equity(
            trailing=trailing
        )
        profitability_ratios["Return on Invested Capital"] = (
            self.get_return_on_invested_capital(trailing=trailing)
        )
        profitability_ratios["Return on Capital Employed"] = (
            self.get_return_on_capital_employed(trailing=trailing)
        )
        profitability_ratios["Return on Tangible Assets"] = (
            self.get_return_on_tangible_assets(trailing=trailing)
        )
        profitability_ratios["Income Quality Ratio"] = self.get_income_quality_ratio(
            trailing=trailing
        )
        profitability_ratios["Net Income per EBT"] = self.get_net_income_per_ebt(
            trailing=trailing
        )
        profitability_ratios["Free Cash Flow to Operating Cash Flow Ratio"] = (
            self.get_free_cash_flow_operating_cash_flow_ratio()
        )
        profitability_ratios["EBT to EBIT Ratio"] = self.get_EBT_to_EBIT(
            trailing=trailing
        )
        profitability_ratios["EBIT to Revenue"] = self.get_EBIT_to_revenue(
            trailing=trailing
        )

        self._profitability_ratios = (
            pd.concat(profitability_ratios)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._profitability_ratios = self._profitability_ratios.loc[self._tickers]

        self._profitability_ratios = self._profitability_ratios.round(
            rounding if rounding else self._rounding
        )

        # In case sorting accidentally fails, the index is sorted again
        # to follow the same order as the financial statements
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._profitability_ratios
        ]
        self._profitability_ratios = self._profitability_ratios.reindex(
            available_columns, axis=1
        )

        if growth:
            self._profitability_ratios_growth = calculate_growth(
                self._profitability_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
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

    @handle_portfolio
    @handle_errors
    def get_gross_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the gross margin, a profitability ratio that measures the percentage of
        revenue that exceeds the cost of goods sold.

        The gross margin ratio is calculated by subtracting the cost of goods sold (COGS) from
        the total revenue and then dividing the result by the total revenue. It represents the
        portion of revenue that contributes to covering other expenses and generating profit.

        The formula is as follows:

        - Gross Margin Ratio = (Revenue — Cost of Goods Sold) / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Gross margin ratio values.

        Notes:
        - The method retrieves historical data and calculates the gross margin ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        gross_margin_ratios = toolkit.ratios.get_gross_margin()
        ```
        """
        if trailing:
            gross_margin = profitability_model.get_gross_margin(
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
                self._income_statement.loc[:, "Cost of Goods Sold", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            gross_margin = profitability_model.get_gross_margin(
                self._income_statement.loc[:, "Revenue", :],
                self._income_statement.loc[:, "Cost of Goods Sold", :],
            )

        if growth:
            return calculate_growth(
                gross_margin, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return gross_margin.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_operating_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating margin, a profitability ratio that measures the percentage of
        revenue that remains after deducting operating expenses.

        The operating margin ratio is calculated by subtracting the operating expenses from the
        total revenue and then dividing the result by the total revenue. It indicates how efficiently
        a company is managing its operating expenses in relation to its revenue.

        The formula is as follows:

        - Operating Margin Ratio = Operating Income / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Operating margin ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating margin ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        operating_margin_ratios = toolkit.ratios.get_operating_margin()
        ```
        """
        if trailing:
            operating_margin = profitability_model.get_operating_margin(
                self._income_statement.loc[:, "Operating Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            operating_margin = profitability_model.get_operating_margin(
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

    @handle_portfolio
    @handle_errors
    def get_net_profit_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the net profit margin, a profitability ratio that measures the percentage
        of profit a company earns per dollar of revenue.

        The net profit margin ratio is calculated by dividing the net income by the total revenue.
        It indicates the portion of each dollar of revenue that represents profit after all expenses
        have been deducted. A higher net profit margin is generally considered favorable.

        The formula is as follows:

        - Net Profit Margin Ratio = Net Income / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Net profit margin ratio values.

        Notes:
        - The method retrieves historical data and calculates the net profit margin ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        net_profit_margin_ratios = toolkit.ratios.get_net_profit_margin()
        ```
        """
        if trailing:
            net_profit_margin = profitability_model.get_net_profit_margin(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            net_profit_margin = profitability_model.get_net_profit_margin(
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

    @handle_portfolio
    @handle_errors
    def get_interest_burden_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute the Interest Coverage Ratio, a metric that reveals a company's
        ability to cover its interest expenses with its pre-tax profits.
        This ratio measures the proportion of pre-tax profits required to
        pay for interest payments and is crucial in determining a
        company's financial health.

        The Interest Coverage Ratio is calculated by dividing the earnings before
        interest and taxes (EBIT) by the interest expenses. A higher ratio indicates
        that the company has more earnings to cover its interest expenses, which is
        generally considered favorable.

        The formula is as follows:

        - Interest Coverage Ratio = EBIT (or Operating Income) / Interest Expenses

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Interest Coverage Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Interest Coverage Ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        interest_coverage_ratios = toolkit.ratios.get_interest_burden_ratio()
        ```
        """
        if trailing:
            interest_burden_ratio = profitability_model.get_interest_coverage_ratio(
                self._income_statement.loc[:, "Operating Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Interest Expense", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            interest_burden_ratio = profitability_model.get_interest_burden_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_income_before_tax_profit_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Pretax Profit Margin, which is the ratio of a company's pre-tax profit to
        its revenue, indicating how much profit a company makes before paying taxes on its earnings.

        The Pretax Profit Margin is calculated by dividing the pre-tax profit by the revenue.
        It provides insight into how efficiently a company is able to generate profits from its revenue.

        The formula is as follows:

        - Pretax Profit Margin = Income Before Tax / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Pretax Profit Margin values.

        Notes:
        - The method retrieves historical data and calculates the Pretax Profit Margin for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        pretax_profit_margin = toolkit.ratios.get_income_before_tax_profit_margin()
        ```
        """
        if trailing:
            income_before_tax_profit_margin = (
                profitability_model.get_income_before_tax_profit_margin(
                    self._income_statement.loc[:, "Income Before Tax", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._income_statement.loc[:, "Revenue", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            income_before_tax_profit_margin = (
                profitability_model.get_income_before_tax_profit_margin(
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

    @handle_portfolio
    @handle_errors
    def get_effective_tax_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the effective tax rate, a financial ratio that measures the
        percentage of pretax income that is paid as taxes.

        The effective tax rate is calculated by dividing the income tax expense by the
        pre-tax income.

        The formula is as follows:

        - Effective Tax Rate = Income Tax Expense / Income Before Tax

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Effective tax rate values.

        Notes:
        - The method retrieves historical data and calculates the effective tax rate for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        effective_tax_rate = toolkit.ratios.get_effective_tax_rate()
        ```
        """
        if trailing:
            effective_tax_rate = profitability_model.get_effective_tax_rate(
                self._income_statement.loc[:, "Income Tax Expense", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Income Before Tax", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            effective_tax_rate = profitability_model.get_effective_tax_rate(
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

    @handle_portfolio
    @handle_errors
    def get_return_on_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the return on assets (ROA), a profitability ratio that measures how
        efficiently a company uses its assets to generate profits.

        The return on assets is calculated by dividing the net income by the average total assets. Note
        that it is false to take the total assets at the end of the period given that income statements
        report over the period whereas a balance sheet reports on the period.

        The formula is as follows:

        - Return on Assets = Net Income / Average Total Assets

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Return on assets (ROA) values.

        Notes:
        - The method retrieves historical data and calculates the ROA for each asset in the
        Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        roa_ratios = toolkit.ratios.get_return_on_assets()
        ```
        """
        if trailing:
            return_on_assets = profitability_model.get_return_on_assets(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            return_on_assets = profitability_model.get_return_on_assets(
                self._income_statement.loc[:, "Net Income", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        if growth:
            return calculate_growth(
                return_on_assets,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return return_on_assets.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_return_on_equity(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the return on equity (ROE), a profitability ratio that measures how
        efficiently a company generates profits using its shareholders' equity.

        The return on equity is calculated by dividing the net income by the average shareholders' equity.
        Shareholders' equity represents the residual interest in the assets of a company after deducting liabilities.
        Note that it is false to take the total assets at the end of the period given that income statements report
        over the period whereas a balance sheet reports on the period.

        ROE provides insight into the company's ability to generate profits from the investments made by
        its shareholders. A higher ROE indicates that the company is using its equity effectively to generate
        higher returns for its shareholders.

        The formula is as follows:

        - Return on Equity = Net Income / Average Shareholders' Equity (or Total Equity)

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Return on equity (ROE) values.

        Notes:
        - The method retrieves historical data and calculates the ROE for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        roe_ratios = toolkit.ratios.get_return_on_equity()
        ```
        """
        if trailing:
            return_on_equity = profitability_model.get_return_on_equity(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(window=trailing)
                .mean()
                .T,
            )

        else:
            return_on_equity = profitability_model.get_return_on_equity(
                self._income_statement.loc[:, "Net Income", :],
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(window=2)
                .mean()
                .T,
            )

        if growth:
            return calculate_growth(
                return_on_equity,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return return_on_equity.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_return_on_invested_capital(
        self,
        dividend_adjusted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the return on invested capital (ROIC), a financial ratio that measures
        the company's return on the capital invested in it, including both equity and debt.

        The return on invested capital is calculated by dividing the net operating profit after taxes (NOPAT)
        by the average invested capital. Invested capital includes both equity and debt, making this ratio
        a valuable measure of how efficiently a company generates returns for all of its investors.

        The formula is as follows:

        - Return on Invested Capital = (Net Income — Paid Dividends) / Average Invested Capital

        Args:
            dividend_adjusted (bool, optional): Whether to adjust the net operating profit after taxes
            with the dividends paid. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Return on invested capital (ROIC) values.

        Notes:
        - The method retrieves historical data and calculates the ROIC for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        roic_ratios = toolkit.ratios.get_return_on_invested_capital()
        ```
        """
        if trailing:
            return_on_invested_capital = (
                profitability_model.get_return_on_invested_capital(
                    self._income_statement.loc[:, "Net Income", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    (
                        self._cash_flow_statement.loc[:, "Dividends Paid", :]
                        .T.rolling(trailing)
                        .sum()
                        .T
                        if dividend_adjusted
                        else 0
                    ),
                    self._balance_sheet_statement.loc[:, "Total Equity", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Debt", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                )
            )
        else:
            return_on_invested_capital = (
                profitability_model.get_return_on_invested_capital(
                    self._income_statement.loc[:, "Net Income", :],
                    (
                        self._cash_flow_statement.loc[:, "Dividends Paid", :]
                        if dividend_adjusted
                        else 0
                    ),
                    self._balance_sheet_statement.loc[:, "Total Equity", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Debt", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                )
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

    @handle_portfolio
    @handle_errors
    def get_income_quality_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the income quality ratio, a financial metric that measures the cash flow from
        operating activities relative to the net income of the company.

        The income quality ratio provides insights into the quality of a company's reported earnings.
        By comparing the cash flow from operating activities to the net income, this ratio helps assess
        whether a company's reported profits are backed by actual cash flow. A higher income quality
        ratio suggests higher earnings quality and a better ability to convert profits into cash flow.

        The formula is as follows:

        - Income Quality Ratio = Cash Flow from Operations / Net Income

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Income quality ratio values.

        Notes:
        - The method retrieves historical data and calculates the income quality ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        income_quality_ratios = toolkit.ratios.get_income_quality_ratio()
        ```
        """
        if trailing:
            income_quality_ratio = profitability_model.get_income_quality_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            income_quality_ratio = profitability_model.get_income_quality_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._income_statement.loc[:, "Net Income", :],
            )

        if growth:
            return calculate_growth(
                income_quality_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return income_quality_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_return_on_tangible_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the return on tangible assets, a financial ratio that measures the amount of profit
        generated by a company's tangible assets.

        The return on tangible assets (ROTA) provides insights into the efficiency with which a company
        utilizes its tangible assets to generate profits. Tangible assets include physical assets such as
        buildings, machinery, and equipment. ROTA indicates how well a company can generate profits from
        its core operational assets.

        The formula is as follows:

        - Return on Tangible Assets = Net Income / Average Tangible Assets

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Return on tangible assets (ROTA) values.

        Notes:
        - The method retrieves historical data and calculates the ROTA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        rota_ratios = toolkit.ratios.get_return_on_tangible_assets()
        ```
        """
        if trailing:
            return_on_tangible_assets = (
                profitability_model.get_return_on_tangible_assets(
                    self._income_statement.loc[:, "Net Income", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Assets", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                    self._balance_sheet_statement.loc[:, "Intangible Assets", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Liabilities", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                )
            )
        else:
            return_on_tangible_assets = (
                profitability_model.get_return_on_tangible_assets(
                    self._income_statement.loc[:, "Net Income", :],
                    self._balance_sheet_statement.loc[:, "Total Assets", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    self._balance_sheet_statement.loc[:, "Intangible Assets", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Liabilities", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                )
            )

        if growth:
            return calculate_growth(
                return_on_tangible_assets,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return return_on_tangible_assets.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_return_on_capital_employed(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the return on capital employed (ROCE), a profitability ratio that measures the amount of return
        a company generates from the capital it has invested in the business.

        Return on capital employed (ROCE) is a crucial financial metric that evaluates the efficiency and profitability
        of a company's utilization of both equity and debt capital to generate profits. It assesses how well the company
        generates earnings relative to the total capital invested in the business.

        The formula is as follows:

        - Return on Capital Employed = EBIT / (Total Assets — Current Liabilities)

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Return on capital employed (ROCE) values.

        Notes:
        - The method retrieves historical data and calculates the ROCE for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        roce_ratios = toolkit.ratios.get_return_on_capital_employed()
        ```
        """
        if trailing:
            return_on_capital_employed = (
                profitability_model.get_return_on_capital_employed(
                    self._income_statement.loc[:, "Net Income", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._income_statement.loc[:, "Interest Expense", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._income_statement.loc[:, "Income Tax Expense", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Assets", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                )
            )
        else:
            return_on_capital_employed = (
                profitability_model.get_return_on_capital_employed(
                    self._income_statement.loc[:, "Net Income", :],
                    self._income_statement.loc[:, "Interest Expense", :],
                    self._income_statement.loc[:, "Income Tax Expense", :],
                    self._balance_sheet_statement.loc[:, "Total Assets", :],
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                )
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

    @handle_portfolio
    @handle_errors
    def get_net_income_per_ebt(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the net income per earnings before taxes (EBT), a profitability ratio that measures
        the net income generated for each dollar of EBT.

        The net income per earnings before taxes (EBT) ratio helps evaluate the extent to which a company's net
        income is generated from its operating activities before considering the impact of income taxes. It gives
        insights into how effectively a company generates profit relative to its taxable income.

        The formula is as follows:

        - Net Income per EBT = Net Income / Income Before Tax

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Net income per earnings before taxes (EBT) values.

        Notes:
        - The method retrieves historical data and calculates the net income per EBT for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        net_income_per_ebt_ratios = toolkit.ratios.get_net_income_per_ebt()
        ```
        """
        if trailing:
            net_income_per_ebt = profitability_model.get_net_income_per_ebt(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Income Tax Expense", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            net_income_per_ebt = profitability_model.get_net_income_per_ebt(
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

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_operating_cash_flow_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the free cash flow to operating cash flow ratio, a profitability ratio that measures
        the amount of free cash flow a company generates for every dollar of operating cash flow.

        The free cash flow to operating cash flow ratio helps assess how well a company's operating activities
        translate into free cash flow, which is the cash available after all expenses and investments. A higher
        ratio indicates that the company is generating strong free cash flow relative to its operating cash flow,
        which could signify efficient capital management.

        The formula is as follows:

        - Free Cash Flow to Operating Cash Flow Ratio = Free Cash Flow / Cash Flow from Operations

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Free cash flow to operating cash flow ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        fcf_to_ocf_ratios = toolkit.ratios.get_free_cash_flow_operating_cash_flow_ratio()
        ```
        """
        if trailing:
            free_cash_flow_operating_cash_flow_ratio = (
                profitability_model.get_free_cash_flow_operating_cash_flow_ratio(
                    self._cash_flow_statement.loc[:, "Free Cash Flow", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            free_cash_flow_operating_cash_flow_ratio = (
                profitability_model.get_free_cash_flow_operating_cash_flow_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_tax_burden_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the tax burden ratio, which is the ratio of a company's
        net income to its income before tax, indicating how much of a
        company's income is retained after taxes.

        The tax burden ratio measures the portion of a company's earnings that
        is paid as taxes. A higher ratio indicates that a larger portion of the
        income is being retained by the company after taxes. This ratio provides
        insights into the tax efficiency of the company and its ability to manage
        its tax liabilities.

        The formula is as follows:

        - Tax Burden Ratio = Net Income / Income Before Tax

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Tax burden ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        tax_burden_ratios = toolkit.ratios.get_tax_burden_ratio()
        ```
        """
        if trailing:
            tax_burden_ratio = profitability_model.get_tax_burden_ratio(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Income Before Tax", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            tax_burden_ratio = profitability_model.get_tax_burden_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_EBT_to_EBIT(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the EBT to EBIT, which is the ratio of a company's earnings before tax to its
        earnings before interest and taxes, indicating how much of a company's earnings are
        generated before paying interest on debt.

        The EBT to EBIT ratio measures the proportion of a company's earnings that is generated
        before paying interest and taxes. It provides insights into how a company's operating
        performance is impacted by interest expenses and tax obligations. A higher ratio indicates
        that a larger portion of the company's earnings is generated from its core operations
        before considering interest payments and taxes.

        The formula is as follows:

        - EBT to EBIT = (Net Income + Income Tax Expense) / (Net Income + Income Tax Expense + Interest Expense)

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: EBT to EBIT ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ebt_to_ebit_ratios = toolkit.ratios.get_EBT_to_EBIT()
        ```
        """
        if trailing:
            EBT_to_EBIT = profitability_model.get_EBT_to_EBIT(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T
                + self._income_statement.loc[:, "Income Tax Expense", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T
                + self._income_statement.loc[:, "Income Tax Expense", :]
                .T.rolling(trailing)
                .sum()
                .T
                + self._income_statement.loc[:, "Interest Expense", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            EBT_to_EBIT = profitability_model.get_EBT_to_EBIT(
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

    @handle_portfolio
    @handle_errors
    def get_EBIT_to_revenue(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the EBIT per Revenue, which is the ratio of a company's earnings
        before interest and taxes to its revenue, indicating how much profit a
        company generates from its operations before paying interest on debt
        and taxes on its earnings.

        The EBIT to Revenue ratio measures the company's ability to generate profit
        from its core operations relative to its revenue. It provides insights into
        the operational efficiency and profitability of the company, as it excludes
        the impact of interest expenses and taxes on its earnings. A higher ratio
        indicates that a larger portion of the company's revenue is converted into
        operating profit.

        The formula is as follows:

        - EBIT to Revenue = EBIT / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: EBIT to Revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ebit_to_revenue_ratios = toolkit.ratios.get_EBIT_to_revenue()
        ```
        """
        if trailing:
            EBIT_to_revenue = profitability_model.get_EBIT_to_revenue(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T
                + self._income_statement.loc[:, "Income Tax Expense", :]
                .T.rolling(trailing)
                .sum()
                .T
                + self._income_statement.loc[:, "Interest Expense", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        EBIT_to_revenue = profitability_model.get_EBIT_to_revenue(
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

    def collect_solvency_ratios(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Solvency Ratios based on the provided data.

        Args:
            diluted (bool, optional): Whether to use diluted shares for the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Solvency ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various solvency ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        solvency_ratios = toolkit.ratios.collect_solvency_ratios()
        ```
        """
        solvency_ratios: dict = {}

        solvency_ratios["Debt-to-Assets Ratio"] = self.get_debt_to_assets_ratio(
            trailing=trailing
        )
        solvency_ratios["Debt-to-Equity Ratio"] = self.get_debt_to_equity_ratio(
            trailing=trailing
        )
        solvency_ratios["Debt Service Coverage Ratio"] = (
            self.get_debt_service_coverage_ratio(trailing=trailing)
        )
        solvency_ratios["Equity Multiplier"] = self.get_equity_multiplier(
            trailing=trailing
        )
        solvency_ratios["Free Cash Flow Yield"] = self.get_free_cash_flow_yield(
            diluted=diluted, trailing=trailing
        )
        solvency_ratios["Net-Debt to EBITDA Ratio"] = self.get_net_debt_to_ebitda_ratio(
            trailing=trailing
        )
        solvency_ratios["Cash Flow Coverage Ratio"] = self.get_cash_flow_coverage_ratio(
            trailing=trailing
        )
        solvency_ratios["CAPEX Coverage Ratio"] = self.get_capex_coverage_ratio(
            trailing=trailing
        )
        solvency_ratios["Dividend CAPEX Coverage Ratio"] = (
            self.get_capex_dividend_coverage_ratio(trailing=trailing)
        )

        self._solvency_ratios = (
            pd.concat(solvency_ratios)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._solvency_ratios = self._solvency_ratios.loc[self._tickers]

        self._solvency_ratios = self._solvency_ratios.round(
            rounding if rounding else self._rounding
        )

        # In case sorting accidentally fails, the index is sorted again
        # to follow the same order as the financial statements
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._solvency_ratios
        ]
        self._solvency_ratios = self._solvency_ratios.reindex(available_columns, axis=1)

        if growth:
            self._solvency_ratios_growth = calculate_growth(
                self._solvency_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._solvency_ratios_growth[self._tickers[0]]
                if growth
                else self._solvency_ratios.loc[self._tickers[0]]
            )

        return self._solvency_ratios_growth if growth else self._solvency_ratios

    @handle_portfolio
    @handle_errors
    def get_debt_to_assets_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the debt to assets ratio, a solvency ratio that measures the proportion
        of a company's assets that are financed by debt.

        This ratio, also known as the Debt Ratio, indicates the percentage of a company's
        total assets that are funded by debt. It is a measure of a company's financial
        leverage and indicates the extent to which a company relies on borrowed funds to
        finance its operations. A higher ratio implies a higher level of debt in the company's
        capital structure, which could increase financial risk.

        The formula is as follows:

        - Debt to Assets Ratio = Total Debt / Total Assets

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Debt to assets ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        debt_to_assets_ratios = toolkit.ratios.get_debt_to_assets_ratio()
        ```
        """
        if trailing:
            debt_to_assets_ratio = solvency_model.get_debt_to_assets_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            debt_to_assets_ratio = solvency_model.get_debt_to_assets_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_debt_to_equity_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the debt to equity ratio, a solvency ratio that measures the
        proportion of a company's equity that is financed by debt. This ratio is also known
        as the Gearing Ratio.

        The debt to equity ratio, for short the D/E ratio, indicates the relative
        contribution of debt and equity to a company's capital structure. It helps assess
        the level of financial risk a company carries due to its debt obligations. A higher
        ratio implies a higher reliance on debt to finance the business, which could increase
        risk but also potentially lead to higher returns for shareholders.

        The formula is as follows:

        - Debt to Equity Ratio = Total Debt / Total Equity

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Debt to equity ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        debt_to_equity_ratios = toolkit.ratios.get_debt_to_equity_ratio()
        ```
        """
        if trailing:
            debt_to_equity_ratio = solvency_model.get_debt_to_equity_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            debt_to_equity_ratio = solvency_model.get_debt_to_equity_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_interest_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the interest coverage ratio, a solvency ratio that measures a company's
        ability to pay its interest expenses on outstanding debt.

        The interest coverage ratio evaluates a company's ability to meet its interest
        obligations from its operating income. A higher ratio indicates a company's stronger
        ability to cover its interest payments using its earnings, implying lower financial risk.
        Conversely, a lower ratio suggests a company may have difficulty meeting its interest
        obligations and could be at higher risk of default.

        The formula is as follows:

        - Interest Coverage Ratio = Operating Income / (Interest Expense + Depreciation and Amortization)

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Interest coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        interest_coverage_ratios = toolkit.ratios.get_interest_coverage_ratio()
        ```
        """
        if trailing:
            interest_coverage_ratio = solvency_model.get_interest_coverage_ratio(
                self._income_statement.loc[:, "Operating Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Interest Expense", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            interest_coverage_ratio = solvency_model.get_interest_coverage_ratio(
                self._income_statement.loc[:, "Operating Income", :],
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
                self._income_statement.loc[:, "Interest Expense", :],
            )

        if growth:
            return calculate_growth(
                interest_coverage_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return interest_coverage_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_equity_multiplier(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the equity multiplier, a solvency ratio that measures the degree to which
        a company uses borrowed money (debt) to finance its operations and growth.

        The equity multiplier helps assess the financial leverage of a company by indicating
        how much of its assets are financed by equity versus debt. A higher equity multiplier
        suggests that the company relies more on debt financing, which can amplify returns but
        also increases financial risk. Conversely, a lower equity multiplier indicates a
        larger portion of assets is financed by equity, potentially lowering financial risk.

        The formula is as follows:

        - Equity Multiplier = Average Total Assets / Average Total Equity

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Equity multiplier values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        equity_multipliers = toolkit.ratios.get_equity_multiplier()
        ```
        """
        if trailing:
            equity_multiplier = solvency_model.get_equity_multiplier(
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            equity_multiplier = solvency_model.get_equity_multiplier(
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        if growth:
            return calculate_growth(
                equity_multiplier,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return equity_multiplier.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_debt_service_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the debt service coverage ratio, a solvency ratio that measures a company's
        ability to service its debt with its net operating income.

        The debt service coverage ratio provides insights into a company's ability to meet its
        debt obligations from its operating income. It is especially important for companies
        with significant debt obligations, as a lower ratio indicates higher financial risk and
        potential difficulties in servicing debt payments.

        The formula is as follows:

        - Debt Service Coverage Ratio = Operating Income / Total Current Liabilities

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Debt service coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        debt_service_coverage_ratios = toolkit.ratios.get_debt_service_coverage_ratio()
        ```
        """
        if trailing:
            debt_service_coverage_ratio = (
                solvency_model.get_debt_service_coverage_ratio(
                    self._income_statement.loc[:, "Operating Income", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                )
            )
        else:
            debt_service_coverage_ratio = (
                solvency_model.get_debt_service_coverage_ratio(
                    self._income_statement.loc[:, "Operating Income", :],
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                )
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

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_yield(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculates the free cash flow yield ratio, which measures the free cash flow
        relative to the market capitalization of the company.

        The free cash flow yield ratio is a measure of how efficiently a company generates
        free cash flow relative to its market value. It provides insights into whether the
        company's valuation is reasonable compared to the amount of cash it generates.

        The formula is as follows:

        - Free Cash Flow Yield Ratio = Free Cash Flow / Market Capitalization

        Args:
            show_daily (bool, optional): Whether to use daily data for the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for market capitalization. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Free cash flow yield ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        free_cash_flow_yield_ratios = toolkit.ratios.get_free_cash_flow_yield()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        free_cash_flow = self._cash_flow_statement.loc[:, "Free Cash Flow", :]

        years = self._balance_sheet_statement.columns
        begin, end = str(years[0]), str(years[-1])

        if show_daily:
            share_prices = self._daily_historical_data.loc[begin:, "Adj Close"][
                self._tickers_without_portfolio
            ]

            average_shares = map_period_data_to_daily_data(
                period_data=average_shares,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )

            free_cash_flow = map_period_data_to_daily_data(
                period_data=free_cash_flow,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )
        else:
            share_prices = self._historical_data.loc[begin:end, "Adj Close"][
                self._tickers_without_portfolio
            ].T

        if trailing:
            market_cap = valuation_model.get_market_cap(
                share_prices.T.rolling(trailing).sum().T,
                average_shares,
            )

            free_cash_flow_yield = solvency_model.get_free_cash_flow_yield(
                free_cash_flow.T.rolling(trailing).sum().T,
                market_cap,
            )
        else:
            market_cap = valuation_model.get_market_cap(share_prices, average_shares)

            free_cash_flow_yield = solvency_model.get_free_cash_flow_yield(
                free_cash_flow,
                market_cap,
            )

        if growth:
            return calculate_growth(
                free_cash_flow_yield,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return free_cash_flow_yield.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_net_debt_to_ebitda_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculates the net debt to EBITDA ratio, which measures the net debt of the company
        relative to its EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization).

        The net debt to EBITDA ratio is a measure of a company's ability to manage its debt
        obligations in relation to its earnings and cash flow. A lower ratio indicates better
        financial health and a stronger ability to manage debt.

        The formula is as follows:

        - Net Debt to EBITDA Ratio = Net Debt / EBITDA

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Net debt to EBITDA ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        net_debt_to_ebitda_ratios = toolkit.ratios.get_net_debt_to_ebitda_ratio()
        ```
        """
        if trailing:
            net_debt_to_ebitda_ratio = solvency_model.get_net_debt_to_ebitda_ratio(
                self._income_statement.loc[:, "Operating Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Net Debt", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            net_debt_to_ebitda_ratio = solvency_model.get_net_debt_to_ebitda_ratio(
                self._income_statement.loc[:, "Operating Income", :],
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
                self._balance_sheet_statement.loc[:, "Net Debt", :],
            )

        if growth:
            return calculate_growth(
                net_debt_to_ebitda_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return net_debt_to_ebitda_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_cash_flow_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the cash flow coverage ratio, a solvency ratio that measures a company's
        ability to pay off its debt with its operating cash flow.

        The cash flow coverage ratio assesses a company's ability to meet its debt obligations
        by comparing its operating cash flow to its total debt. A higher ratio indicates a
        stronger ability to cover its debt with cash generated from operations.

        The formula is as follows:

        - Cash Flow Coverage Ratio = Cash Flow from Operations / Total Debt

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Cash flow coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        cash_flow_coverage_ratios = toolkit.ratios.get_cash_flow_coverage_ratio()
        ```
        """
        if trailing:
            cash_flow_coverage_ratio = solvency_model.get_cash_flow_coverage_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Total Debt", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            cash_flow_coverage_ratio = solvency_model.get_cash_flow_coverage_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_capex_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the capital expenditure coverage ratio, a solvency ratio that
        measures a company's ability to cover its capital expenditures with its
        cash flow from operations.

        The capex coverage ratio evaluates a company's ability to fund its capital
        expenditures, which are essential for maintaining and growing its business,
        using the cash generated from its operations. A higher ratio indicates a
        stronger ability to fund capital investments from operating cash flow.

        The formula is as follows:

        - Capital Expenditure Coverage Ratio = Cash Flow from Operations / Capital Expenditure

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Capital expenditure coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        capex_coverage_ratios = toolkit.ratios.get_capex_coverage_ratio()
        ```
        """
        if trailing:
            capex_coverage_ratio = solvency_model.get_capex_coverage_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._cash_flow_statement.loc[:, "Capital Expenditure", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            capex_coverage_ratio = solvency_model.get_capex_coverage_ratio(
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

    @handle_portfolio
    @handle_errors
    def get_capex_dividend_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the dividend paid and capital expenditure coverage ratio, a solvency ratio
        that measures a company's ability to cover both its capital expenditures and dividend
        payments with its cash flow from operations.

        The capex dividend coverage ratio assesses whether a company's cash flow from
        operations is sufficient to cover both its capital expenditures (which are essential
        for maintaining and growing its business) and its dividend payments to shareholders.
        A higher ratio indicates a stronger ability to fund both capex and dividends from
        operating cash flow.

        The formula is as follows:

        - Dividend Paid and Capital Expenditure Coverage Ratio = Cash Flow from Operations /
            (Capital Expenditure + Dividends Paid)

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Dividend paid and capex coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        capex_dividend_coverage_ratios = toolkit.ratios.get_capex_dividend_coverage_ratio()
        ```
        """
        if trailing:
            dividend_capex_coverage_ratio = (
                solvency_model.get_dividend_capex_coverage_ratio(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._cash_flow_statement.loc[:, "Capital Expenditure", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._cash_flow_statement.loc[:, "Dividends Paid", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            dividend_capex_coverage_ratio = (
                solvency_model.get_dividend_capex_coverage_ratio(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    self._cash_flow_statement.loc[:, "Capital Expenditure", :],
                    self._cash_flow_statement.loc[:, "Dividends Paid", :],
                )
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

    def collect_valuation_ratios(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Valuation Ratios based on the provided data.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculations. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Valuation ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various valuation ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        valuation_ratios = toolkit.ratios.collect_valuation_ratios()
        ```
        """
        valuation_ratios: dict = {}

        valuation_ratios["Earnings per Share"] = self.get_earnings_per_share(
            include_dividends=include_dividends, diluted=diluted, trailing=trailing
        )
        valuation_ratios["Revenue per Share"] = self.get_revenue_per_share(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Price-to-Earnings"] = self.get_price_to_earnings_ratio(
            include_dividends=include_dividends, diluted=diluted
        )
        valuation_ratios["Price-to-Earnings-Growth"] = (
            self.get_price_to_earnings_growth_ratio(
                include_dividends=include_dividends, diluted=diluted
            )
        )
        valuation_ratios["Book Value per Share"] = self.get_book_value_per_share(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Price-to-Book"] = self.get_price_to_book_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Interest Debt per Share"] = self.get_interest_debt_per_share(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["CAPEX per Share"] = self.get_capex_per_share(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Earnings Yield"] = self.get_earnings_yield(
            include_dividends=include_dividends, diluted=diluted, trailing=trailing
        )
        valuation_ratios["Dividend Payout Ratio"] = self.get_dividend_payout_ratio(
            trailing=trailing
        )

        dividend_yield = self.get_dividend_yield(trailing=trailing)
        dividend_yield_columns = [
            column
            for column in dividend_yield.columns
            if column in valuation_ratios["Dividend Payout Ratio"].columns
        ]
        valuation_ratios["Dividend Yield"] = dividend_yield.loc[
            :, dividend_yield_columns
        ]

        valuation_ratios["Weighted Dividend Yield"] = self.get_weighted_dividend_yield(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Price-to-Cash-Flow"] = self.get_price_to_cash_flow_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Price-to-Free-Cash-Flow"] = (
            self.get_price_to_free_cash_flow_ratio(diluted=diluted, trailing=trailing)
        )
        valuation_ratios["Market Cap"] = self.get_market_cap(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Enterprise Value"] = self.get_enterprise_value(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["EV-to-Sales"] = self.get_ev_to_sales_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["EV-to-EBIT"] = self.get_ev_to_ebit(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["EV-to-EBITDA"] = self.get_ev_to_ebitda_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["EV-to-Operating-Cash-Flow"] = (
            self.get_ev_to_operating_cashflow_ratio(diluted=diluted, trailing=trailing)
        )
        valuation_ratios["Tangible Asset Value"] = self.get_tangible_asset_value(
            trailing=trailing
        )
        valuation_ratios["Net Current Asset Value"] = self.get_net_current_asset_value(
            trailing=trailing
        )

        self._valuation_ratios = (
            pd.concat(valuation_ratios)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._valuation_ratios = self._valuation_ratios.loc[self._tickers]

        self._valuation_ratios = self._valuation_ratios.round(
            rounding if rounding else self._rounding
        )

        # In case sorting accidentally fails, the index is sorted again
        # to follow the same order as the financial statements
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._valuation_ratios
        ]
        self._valuation_ratios = self._valuation_ratios.reindex(
            available_columns, axis=1
        )

        if growth:
            self._valuation_ratios_growth = calculate_growth(
                self._valuation_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._valuation_ratios_growth[self._tickers[0]]
                if growth
                else self._valuation_ratios.loc[self._tickers[0]]
            )

        return self._valuation_ratios_growth if growth else self._valuation_ratios

    @handle_portfolio
    @handle_errors
    def get_earnings_per_share(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the earnings per share (EPS), a valuation ratio that measures the amount
        of net income earned per share of outstanding common stock.

        The earnings per share (EPS) is a widely used financial metric that helps investors
        understand the profitability of a company on a per-share basis. It provides insight
        into the portion of a company's earnings that is allocated to each outstanding share
        of its common stock. EPS is an important measure for investors and analysts when
        assessing a company's financial performance and comparing it to other companies.

        The formula is as follows:

        - Earnings per Share (EPS) = (Net Income - Preferred Dividends Paid) / Weighted Average Shares

        Args:
            include_dividends (bool, optional): Whether to include dividends in the EPS calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted earnings per share. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Earnings per share (EPS) values.

        Notes:
        - The method retrieves historical data and calculates the EPS for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the EPS values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        eps_ratios = toolkit.ratios.get_earnings_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            dividends = (
                self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :]
                .T.rolling(trailing)
                .sum()
                .T
                if include_dividends
                else 0
            )

            earnings_per_share = valuation_model.get_earnings_per_share(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                dividends,
                average_shares,
            )
        else:
            dividends = (
                self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :]
                if include_dividends
                else 0
            )

            earnings_per_share = valuation_model.get_earnings_per_share(
                self._income_statement.loc[:, "Net Income", :],
                dividends,
                average_shares,
            )

        if growth:
            return calculate_growth(
                earnings_per_share,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return earnings_per_share.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_revenue_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the revenue per share, a valuation ratio that measures the amount
        of revenue generated per outstanding share of a company's stock.

        The revenue per share is an important metric that provides insight into a
        company's ability to generate revenue on a per-share basis. It can help investors
        understand the company's revenue-generation efficiency and its overall financial health.

        The formula is as follows:

        - Revenue per Share = Revenue / Weighted Average (Diluted) Shares

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Revenue per share values.

        Notes:
        - The method retrieves historical data and calculates the revenue per share for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the revenue per share values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        revenue_per_share = toolkit.ratios.get_revenue_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            revenue_per_share = valuation_model.get_revenue_per_share(
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
                average_shares,
            )
        else:
            revenue_per_share = valuation_model.get_revenue_per_share(
                self._income_statement.loc[:, "Revenue", :], average_shares
            )

        if growth:
            return calculate_growth(
                revenue_per_share,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return revenue_per_share.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_price_to_earnings_ratio(
        self,
        show_daily: bool = False,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the price earnings ratio (P/E), a valuation ratio that compares a
        company's stock price to its earnings per share.

        The price earnings ratio is a widely used valuation metric that helps investors
        assess the relative value of a company's stock. A higher P/E ratio may indicate
        that the market has high expectations for the company's future growth, while a
        lower P/E ratio may suggest that the company is undervalued.

        The formula is as follows:

        - Price to Earnings Ratio (P/E) = Share Price / Earnings per Share (EPS)

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: Price earnings ratio (P/E) values.

        Notes:
        - The method retrieves historical data and calculates the P/E ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the P/E ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        pe_ratio = toolkit.ratios.get_price_to_earnings_ratio()
        ```
        """
        eps = self.get_earnings_per_share(
            include_dividends, diluted, trailing=4 if self._quarterly else None
        )

        years = eps.columns
        begin, end = str(years[0]), str(years[-1])

        if show_daily:
            share_prices = self._daily_historical_data.loc[begin:, "Adj Close"][
                self._tickers_without_portfolio
            ]

            eps = map_period_data_to_daily_data(
                period_data=eps,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )
        else:
            share_prices = self._historical_data.loc[begin:end, "Adj Close"][
                self._tickers_without_portfolio
            ].T

        price_to_earnings_ratio = valuation_model.get_price_to_earnings_ratio(
            share_prices, eps
        )

        if growth:
            return calculate_growth(
                price_to_earnings_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return price_to_earnings_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_price_to_earnings_growth_ratio(
        self,
        use_ebitda_growth_rate: bool = False,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Calculate the price earnings to growth (PEG) ratio, a valuation metric that
        measures the ratio of the price-to-earnings ratio to earnings growth rate.

        The price-to-earnings growth (PEG) ratio provides a more comprehensive valuation
        measure compared to the P/E ratio alone. It takes into account a company's earnings
        growth rate, allowing investors to assess whether a stock is overvalued or undervalued
        relative to its growth prospects.

        The formula is as follows:

        - Price Earnings to Growth Ratio (PEG) = Price Earnings Ratio (P/E) / Growth Rate

        Args:
            use_ebitda_growth_rate (bool, optional): Whether to use EBITDA growth rate for the calculation.
                Defaults to False.
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Price earnings to growth (PEG) ratio values.

        Notes:
        - The method retrieves historical data and calculates the PEG ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the PEG ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        peg_ratio = toolkit.ratios.get_price_to_earnings_growth_ratio()
        ```
        """
        trailing_metric = 5 * 4 if self._quarterly else 5

        if use_ebitda_growth_rate:
            growth_rate = (
                self._income_statement.loc[:, "EBITDA", :]
                .T.rolling(trailing_metric)
                .sum()
                .T
            )

            growth_rate = calculate_growth(growth_rate)
        else:
            growth_rate = self.get_earnings_per_share(
                include_dividends,
                diluted=diluted,
                growth=True,
                trailing=trailing_metric,
            )

        price_earnings = self.get_price_to_earnings_ratio(
            include_dividends, diluted=diluted
        )

        price_to_earnings_growth_ratio = (
            valuation_model.get_price_to_earnings_growth_ratio(
                price_earnings, growth_rate * 100
            )
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

    @handle_portfolio
    @handle_errors
    def get_book_value_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the book value per share, a valuation ratio that measures the
        amount of common equity value per share outstanding.

        The book value per share is a fundamental valuation metric that reflects
        the net worth of a company attributed to each outstanding share of common stock.

        The formula is as follows:

        - Book Value per Share = (Total Shareholder Equity — Preferred Stock) / Weighted Average (Diluted) Shares

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Book value per share values.

        Notes:
        - The method retrieves historical data and calculates the book value per share for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the book value per share values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        book_value_per_share = toolkit.ratios.get_book_value_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            book_value_per_share = valuation_model.get_book_value_per_share(
                self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Preferred Stock", :]
                .T.rolling(trailing)
                .mean()
                .T,
                average_shares.T.rolling(trailing).mean().T,
            )
        else:
            book_value_per_share = valuation_model.get_book_value_per_share(
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

    @handle_portfolio
    @handle_errors
    def get_price_to_book_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the price to book ratio, a valuation ratio that compares a
        company's market price to its book value per share.

        The price to book ratio is a key valuation metric that helps investors
        assess whether a company's stock is overvalued or undervalued relative to its
        underlying net asset value.

        The formula is as follows:

        - Price to Book Ratio = Share Price / Book Value per Share

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Price to book ratio values.

        Notes:
        - The method retrieves historical data and calculates the price to book ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the price to book ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        price_to_book_ratio = toolkit.ratios.get_price_to_book_ratio()
        ```
        """
        book_value_per_share = self.get_book_value_per_share(
            diluted, trailing=trailing if trailing else None
        )

        years = book_value_per_share.columns
        begin, end = str(years[0]), str(years[-1])

        if show_daily:
            share_prices = self._daily_historical_data.loc[begin:, "Adj Close"][
                self._tickers_without_portfolio
            ]

            book_value_per_share = map_period_data_to_daily_data(
                period_data=book_value_per_share,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )
        else:
            share_prices = self._historical_data.loc[begin:end, "Adj Close"][
                self._tickers_without_portfolio
            ].T

        price_to_book_ratio = valuation_model.get_price_to_book_ratio(
            share_prices, book_value_per_share
        )

        if growth:
            return calculate_growth(
                price_to_book_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return price_to_book_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_interest_debt_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the interest debt per share, a valuation ratio that measures the
        amount of interest expense incurred per outstanding share of a company's stock.

        The interest debt per share ratio provides insight into how much interest a company
        pays on its debt relative to its shareholder base. It can help investors assess
        the financial burden of interest expenses on the company's profitability.

        The formula is as follows:

        - Interest Debt per Share = (Interest Expense / Total Debt) / Weighted Average (Diluted) Shares

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Interest debt per share values.

        Notes:
        - The method retrieves historical data and calculates the interest debt per share ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the interest debt per share values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        interest_debt_per_share = toolkit.ratios.get_interest_debt_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            interest_debt_per_share = valuation_model.get_interest_debt_per_share(
                self._income_statement.loc[:, "Interest Expense", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Total Debt", :]
                .T.rolling(trailing)
                .mean()
                .T,
                average_shares.T.rolling(trailing).mean().T,
            )
        else:
            interest_debt_per_share = valuation_model.get_interest_debt_per_share(
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

    @handle_portfolio
    @handle_errors
    def get_capex_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the capex per share, a valuation ratio that measures the amount of
        capital expenditures made per outstanding share of a company's stock.

        The capex per share ratio provides insight into how much capital a company invests
        in its operations and growth initiatives relative to its shareholder base. It can
        help investors assess the level of reinvestment into the business.

        The formula is as follows:

        - CAPEX per Share = Capital Expenditure / Weighted Average (Diluted) Shares

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Capex per share values.

        Notes:
        - The method retrieves historical data and calculates the capex per share ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the capex per share values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        capex_per_share = toolkit.ratios.get_capex_per_share()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            capex_per_share = valuation_model.get_capex_per_share(
                self._cash_flow_statement.loc[:, "Capital Expenditure", :]
                .T.rolling(trailing)
                .sum()
                .T,
                average_shares,
            )
        else:
            capex_per_share = valuation_model.get_capex_per_share(
                self._cash_flow_statement.loc[:, "Capital Expenditure", :],
                average_shares,
            )

        if growth:
            return calculate_growth(
                capex_per_share,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return capex_per_share.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_dividend_yield(
        self,
        show_daily: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the dividend yield ratio, a valuation ratio that measures the
        amount of dividends distributed per share of stock relative to the stock's price.

        The dividend yield ratio is used by investors to assess the income potential
        of an investment in a company's stock based on the dividends it pays out. A higher
        dividend yield can be attractive to income-seeking investors.

        The formula is as follows:

        - Dividend Yield = Dividends per Share / Share Price

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Dividend yield values.

        Notes:
        - The method retrieves historical data and calculates the dividend yield ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the dividend yield values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        dividend_yield = toolkit.ratios.get_dividend_yield()
        ```
        """
        if show_daily:
            share_prices = self._daily_historical_data.loc[:, "Adj Close"][
                self._tickers_without_portfolio
            ]
            dividends = self._daily_historical_data.loc[:, "Dividends"][
                self._tickers_without_portfolio
            ]
        else:
            share_prices = self._historical_data.loc[:, "Adj Close"][
                self._tickers_without_portfolio
            ].T
            dividends = self._historical_data.loc[:, "Dividends"][
                self._tickers_without_portfolio
            ].T

        dividend_yield = valuation_model.get_dividend_yield(
            dividends.T.rolling(trailing).sum().T if trailing else dividends,
            share_prices,
        )

        if growth:
            return calculate_growth(
                dividend_yield,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return dividend_yield.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_weighted_dividend_yield(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the weighted dividend yield ratio, a valuation ratio that measures the
        amount of dividends distributed per share of stock relative to the stock's price.

        This dividend yield ratio takes into account the (diluted) weighted average shares and actual
        dividends paid as found in the cash flow statement. It provides a more accurate reflection
        of the dividends paid out per share, considering any changes in the number of shares.

        The formula is as follows:

        - Weighted Dividend Yield = Dividends Paid / Weighted Average (Diluted) Shares * Share Price

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Weighted dividend yield values.

        Notes:
        - The method retrieves historical data and calculates the weighted dividend yield ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the weighted dividend yield values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        weighted_dividend_yield = toolkit.ratios.get_weighted_dividend_yield()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        dividends_paid = abs(self._cash_flow_statement.loc[:, "Dividends Paid", :])

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        if show_daily:
            share_prices = self._daily_historical_data.loc[begin:, "Adj Close"][
                self._tickers_without_portfolio
            ]

            average_shares = map_period_data_to_daily_data(
                period_data=average_shares,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )

            dividends_paid = map_period_data_to_daily_data(
                period_data=dividends_paid,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )
        else:
            share_prices = self._historical_data.loc[begin:end, "Adj Close"][
                self._tickers_without_portfolio
            ].T

        if trailing:
            weighted_dividend_yield = valuation_model.get_weighted_dividend_yield(
                dividends_paid.T.rolling(trailing).sum().T,
                average_shares,
                share_prices,
            )
        else:
            weighted_dividend_yield = valuation_model.get_weighted_dividend_yield(
                dividends_paid,
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

    @handle_portfolio
    @handle_errors
    def get_price_to_cash_flow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the price to cash flow ratio, a valuation ratio that compares a
        company's market price to its operating cash flow per share.

        The price to cash flow ratio is a key valuation metric that helps investors
        assess the relative value of a company's stock. It is similar to the price to
        earnings ratio, but uses cash flow instead of earnings in the denominator.

        The formula is as follows:

        - Price to Cash Flow Ratio = Share Price / Cash Flow from Operations per Share

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Price to cash flow ratio values.

        Notes:
        - The method retrieves historical data and calculates the price to cash flow ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the price to cash flow ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        price_to_cash_flow_ratio = toolkit.ratios.get_price_to_cash_flow_ratio()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        cash_flow_from_operations = self._cash_flow_statement.loc[
            :, "Cash Flow from Operations", :
        ]

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        if show_daily:
            share_prices = self._daily_historical_data.loc[begin:, "Adj Close"][
                self._tickers_without_portfolio
            ]

            average_shares = map_period_data_to_daily_data(
                period_data=average_shares,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )

            cash_flow_from_operations = map_period_data_to_daily_data(
                period_data=cash_flow_from_operations,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )
        else:
            share_prices = self._historical_data.loc[begin:end, "Adj Close"][
                self._tickers_without_portfolio
            ].T

        market_cap = valuation_model.get_market_cap(share_prices, average_shares)

        if trailing:
            price_to_cash_flow_ratio = valuation_model.get_price_to_cash_flow_ratio(
                market_cap,
                cash_flow_from_operations.T.rolling(trailing).sum().T,
            )
        else:
            price_to_cash_flow_ratio = valuation_model.get_price_to_cash_flow_ratio(
                market_cap, cash_flow_from_operations
            )

        if growth:
            return calculate_growth(
                price_to_cash_flow_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return price_to_cash_flow_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_price_to_free_cash_flow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the price to free cash flow ratio, a valuation ratio that compares a
        company's market price to its free cash flow per share.

        This ratio provides insight into how the market values a company's ability to generate free cash flow.

        The formula is as follows:

        - Price to Free Cash Flow Ratio = Market Cap / Free Cash Flow

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Price to free cash flow ratio values.

        Notes:
        - The method retrieves historical data and calculates the price to free cash flow ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the price to free cash flow ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        price_to_free_cash_flow_ratio = toolkit.ratios.get_price_to_free_cash_flow_ratio()
        ```
        """
        market_cap = self.get_market_cap(
            diluted=diluted,
            trailing=trailing if trailing else None,
            show_daily=show_daily,
        )

        free_cash_flow = self._cash_flow_statement.loc[:, "Free Cash Flow", :]

        if show_daily:
            free_cash_flow = map_period_data_to_daily_data(
                period_data=free_cash_flow,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
            )

        if trailing:
            price_to_free_cash_flow_ratio = (
                valuation_model.get_price_to_free_cash_flow_ratio(
                    market_cap,
                    free_cash_flow.T.rolling(trailing).sum().T,
                )
            )
        else:
            price_to_free_cash_flow_ratio = (
                valuation_model.get_price_to_free_cash_flow_ratio(
                    market_cap, free_cash_flow
                )
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

    @handle_portfolio
    @handle_errors
    def get_market_cap(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculates the market capitalization of the company.

        Market capitalization, often referred to as "market cap," is the total value of a company's
        outstanding shares of stock in the stock market. It is calculated by multiplying the current
        market price per share by the total number of outstanding shares.

        The formula is as follows:

        - Market Capitalization = Share Price * Weighted Average (Diluted) Shares

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Market capitalization values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        market_cap = toolkit.ratios.get_market_cap()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        if show_daily:
            share_prices = self._daily_historical_data.loc[begin:, "Adj Close"][
                self._tickers_without_portfolio
            ]

            average_shares = map_period_data_to_daily_data(
                period_data=average_shares,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )
        else:
            share_prices = self._historical_data.loc[begin:end, "Adj Close"][
                self._tickers_without_portfolio
            ].T

        if trailing:
            market_cap = valuation_model.get_market_cap(
                share_prices,
                average_shares,
            )
        else:
            market_cap = valuation_model.get_market_cap(share_prices, average_shares)

        if growth:
            return calculate_growth(
                market_cap, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return market_cap.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_enterprise_value(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculates the Enterprise Value (EV) of a company. The Enterprise Value (EV)
        is a measure of a company's total value, often used as a more comprehensive
        alternative to market capitalization. It is calculated as the sum of a company's
        market capitalization, outstanding debt, minority interest, and
        preferred equity, minus the cash and cash equivalents.

        The formula is as follows:

        - Enterprise Value = Market Capitalization + Total Debt + Minority Interest + Preferred Equity
            — Cash and Cash Equivalents

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Enterprise Value values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        enterprise_value = toolkit.ratios.get_enterprise_value()
        ```
        """
        total_debt = self._balance_sheet_statement.loc[:, "Total Debt", :]
        minority_interest = self._balance_sheet_statement.loc[:, "Minority Interest", :]
        preferred_stock = self._balance_sheet_statement.loc[:, "Preferred Stock", :]
        cash_and_cash_equivalents = self._balance_sheet_statement.loc[
            :, "Cash and Cash Equivalents", :
        ]

        market_cap = self.get_market_cap(
            diluted=diluted,
            trailing=trailing if trailing else None,
            show_daily=show_daily,
        )

        if show_daily:
            total_debt = map_period_data_to_daily_data(
                period_data=total_debt,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
            )

            minority_interest = map_period_data_to_daily_data(
                period_data=minority_interest,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
            )

            preferred_stock = map_period_data_to_daily_data(
                period_data=preferred_stock,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
            )

            cash_and_cash_equivalents = map_period_data_to_daily_data(
                period_data=cash_and_cash_equivalents,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
            )

        if trailing:
            enterprise_value = valuation_model.get_enterprise_value(
                market_cap,
                total_debt.T.rolling(trailing).mean().T,
                minority_interest.T.rolling(trailing).mean().T,
                preferred_stock.T.rolling(trailing).mean().T,
                cash_and_cash_equivalents.T.rolling(trailing).mean().T,
            )
        else:
            enterprise_value = valuation_model.get_enterprise_value(
                market_cap,
                total_debt,
                minority_interest,
                preferred_stock,
                cash_and_cash_equivalents,
            )

        if growth:
            return calculate_growth(
                enterprise_value,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return enterprise_value.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_ev_to_sales_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the EV to sales ratio, a valuation ratio that compares a company's
        enterprise value (EV) to its total revenue.

        This ratio compares the enterprise value (EV) to the total revenue generated
        by the company. It can provide insights into how efficiently a company is using
        its revenue to generate value for its investors.

        The formula is as follows:

        - Enterprise Value to Sales Ratio = Enterprise Value / Total Revenue

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: EV to Sales Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ev_to_sales_ratio = toolkit.ratios.get_ev_to_sales_ratio()
        ```
        """
        enterprise_value = self.get_enterprise_value(
            diluted=diluted,
            trailing=trailing if trailing else None,
            show_daily=show_daily,
        )

        revenue = self._income_statement.loc[:, "Revenue", :]

        if show_daily:
            revenue = map_period_data_to_daily_data(
                period_data=revenue,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
            )

        if trailing:
            ev_to_sales_ratio = valuation_model.get_ev_to_sales_ratio(
                enterprise_value,
                revenue.T.rolling(trailing).sum().T,
            )
        else:
            ev_to_sales_ratio = valuation_model.get_ev_to_sales_ratio(
                enterprise_value, revenue
            )

        if growth:
            return calculate_growth(
                ev_to_sales_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return ev_to_sales_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_ev_to_ebitda_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the enterprise value over EBITDA ratio, a valuation ratio that
        measures a company's total value (including debt and equity) relative to its EBITDA.

        This ratio helps investors understand how many times the enterprise value exceeds the
        company's EBITDA, providing insights into the company's debt load and operating performance.

        The formula is as follows:

        - Enterprise Value to EBITDA Ratio = Enterprise Value / EBITDA

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: EV to EBITDA Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ev_to_ebitda_ratio = toolkit.ratios.get_ev_to_ebitda_ratio()
        ```
        """
        enterprise_value = self.get_enterprise_value(
            diluted=diluted,
            trailing=trailing if trailing else None,
            show_daily=show_daily,
        )

        operating_income = self._income_statement.loc[:, "Operating Income", :]
        depreciation_and_amortization = self._cash_flow_statement.loc[
            :, "Depreciation and Amortization", :
        ]

        if show_daily:
            operating_income = map_period_data_to_daily_data(
                period_data=operating_income,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
            )

            depreciation_and_amortization = map_period_data_to_daily_data(
                period_data=depreciation_and_amortization,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
            )

        if trailing:
            ev_to_ebitda_ratio = valuation_model.get_ev_to_ebitda_ratio(
                enterprise_value,
                operating_income.T.rolling(trailing).sum().T,
                depreciation_and_amortization.T.rolling(trailing).sum().T,
            )
        else:
            ev_to_ebitda_ratio = valuation_model.get_ev_to_ebitda_ratio(
                enterprise_value, operating_income, depreciation_and_amortization
            )

        if growth:
            return calculate_growth(
                ev_to_ebitda_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return ev_to_ebitda_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_ev_to_operating_cashflow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the enterprise value over operating cash flow ratio, a valuation ratio that
        measures a company's total value (including debt and equity) relative to its operating cash flow.

        The ratio is a valuation metric that helps investors assess the company's valuation
        relative to its operating cash flow. This ratio provides insights into how many times
        the enterprise value exceeds the company's operating cash flow, indicating the company's
        ability to generate cash from its operations.

        The formula is as follows:

        - Enterprise Value to Operating Cash Flow Ratio = Enterprise Value / Operating Cash Flow

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: EV to Operating Cash Flow Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ev_to_operating_cashflow_ratio = toolkit.ratios.get_ev_to_operating_cashflow_ratio()
        ```
        """
        enterprise_value = self.get_enterprise_value(
            diluted=diluted,
            trailing=trailing if trailing else None,
            show_daily=show_daily,
        )

        cash_flow_from_operations = self._cash_flow_statement.loc[
            :, "Cash Flow from Operations", :
        ]

        if show_daily:
            cash_flow_from_operations = map_period_data_to_daily_data(
                period_data=cash_flow_from_operations,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
            )

        if trailing:
            ev_to_operating_cashflow_ratio = (
                valuation_model.get_ev_to_operating_cashflow_ratio(
                    enterprise_value,
                    cash_flow_from_operations.T.rolling(trailing).sum().T,
                )
            )
        else:
            ev_to_operating_cashflow_ratio = (
                valuation_model.get_ev_to_operating_cashflow_ratio(
                    enterprise_value, cash_flow_from_operations
                )
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

    @handle_portfolio
    @handle_errors
    def get_earnings_yield(
        self,
        show_daily: bool = False,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the earnings yield ratio, a valuation ratio that measures the earnings per share
        relative to the market price per share.

        The earnings yield ratio is a valuation metric that provides insights into how much a
        company's earnings contribute to its stock price. It compares the earnings per share
        to the market price per share, helping investors understand the earnings potential of
        the company relative to its current market value

        The formula is as follows:

        - Earnings Yield Ratio = Earnings per Share / Share Price

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Earnings Yield Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        earnings_yield_ratio = toolkit.ratios.get_earnings_yield()
        ```
        """
        eps = self.get_earnings_per_share(
            include_dividends, diluted=diluted, trailing=trailing if trailing else None
        )

        years = eps.columns
        begin, end = str(years[0]), str(years[-1])

        if show_daily:
            share_prices = self._daily_historical_data.loc[begin:, "Adj Close"][
                self._tickers_without_portfolio
            ]

            eps = map_period_data_to_daily_data(
                period_data=eps,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
            )
        else:
            share_prices = self._historical_data.loc[begin:end, "Adj Close"][
                self._tickers_without_portfolio
            ].T

        earnings_yield = valuation_model.get_earnings_yield(eps, share_prices)

        if growth:
            return calculate_growth(
                earnings_yield,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return earnings_yield.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_dividend_payout_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the Dividend payout ratio, a financial metric that measures the proportion
        of earnings paid out as dividends to shareholders.

        The payout ratio is a financial metric that helps investors assess the
        portion of a company's earnings that is being distributed to shareholders
        in the form of dividends. It's a valuable indicator for dividend investors as
        it indicates the sustainability of dividend payments and the company's
        approach to distributing profits.

        The formula is as follows:

        - Dividend Payout Ratio = Dividends Paid / Net Income

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Payout Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.get_dividend_payout_ratio()
        ```
        """
        if trailing:
            payout_ratio = valuation_model.get_dividend_payout_ratio(
                self._cash_flow_statement.loc[:, "Dividends Paid", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            payout_ratio = valuation_model.get_dividend_payout_ratio(
                self._cash_flow_statement.loc[:, "Dividends Paid", :],
                self._income_statement.loc[:, "Net Income", :],
            )

        if growth:
            return calculate_growth(
                payout_ratio, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return payout_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_reinvestment_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the Reinvestment rate, a financial metric that measures the proportion
        of earnings retained by the company.

        The reinvestment rate is a financial metric that helps investors assess the
        portion of a company's earnings that is being retained by the company
        for future growth. It's a valuable indicator for dividend investors as
        it indicates the sustainability of dividend payments and the company's
        approach to distributing profits.

        The formula is as follows:

        - Reinvestment Rate = 1 — Dividend Payout Ratio

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Reinvestment Rate values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.get_reinvestment_rate()
        ```
        """
        if trailing:
            dividend_payout_ratio = self.get_dividend_payout_ratio(trailing=trailing)
            reinvestment_ratio = valuation_model.get_reinvestment_ratio(
                dividend_payout_ratio=dividend_payout_ratio,
            )
        else:
            dividend_payout_ratio = self.get_dividend_payout_ratio()
            reinvestment_ratio = valuation_model.get_reinvestment_ratio(
                dividend_payout_ratio=dividend_payout_ratio,
            )

        if growth:
            return calculate_growth(
                reinvestment_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        return reinvestment_ratio.round(rounding if rounding else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_tangible_asset_value(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the tangible asset value, a financial metric that represents the total value
        of a company's assets that can be used to generate revenue. Tangible assets are those
        physical assets that have a finite monetary value and can be sold, used, or consumed.

        The formula is as follows:

        - Tangible Asset Value = Total Assets — Total Liabilities — Goodwill

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Tangible Asset Value values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        tangible_asset_value = toolkit.ratios.get_tangible_asset_value()
        ```
        """
        if trailing:
            tangible_asset_value = valuation_model.get_tangible_asset_value(
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Liabilities", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Goodwill", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            tangible_asset_value = valuation_model.get_tangible_asset_value(
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

    @handle_portfolio
    @handle_errors
    def get_net_current_asset_value(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the net current asset value, a financial metric that represents the total value
        of a company's current assets minus its current liabilities. It indicates the extent to
        which a company's short-term assets exceed its short-term liabilities.

        The formula is as follows:

        - Net Current Asset Value = Total Current Assets — Total Current Liabilities

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Net Current Asset Value values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        net_current_asset_value = toolkit.ratios.get_net_current_asset_value()
        ```
        """
        if trailing:
            net_current_asset_value = valuation_model.get_net_current_asset_value(
                self._balance_sheet_statement.loc[:, "Total Current Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            net_current_asset_value = valuation_model.get_net_current_asset_value(
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

    @handle_portfolio
    @handle_errors
    def get_ev_to_ebit(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Calculate the enterprise value over earnings before interest and taxes (EBIT) ratio,
        which is a valuation metric that compares a company's total value (including debt and equity)
        relative to its earnings before interest and taxes.

        The formula is as follows:

        - Enterprise Value to EBIT Ratio = Enterprise Value / EBIT

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Enterprise Value over EBIT values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ev_to_ebit_ratio = toolkit.ratios.get_ev_to_ebit()
        ```
        """
        enterprise_value = self.get_enterprise_value(
            diluted=diluted,
            trailing=trailing if trailing else None,
            show_daily=show_daily,
        )

        ebit = (
            self._income_statement.loc[:, "Net Income", :]
            + self._income_statement.loc[:, "Income Tax Expense", :]
            + self._income_statement.loc[:, "Interest Expense", :]
        )

        if show_daily:
            ebit = map_period_data_to_daily_data(
                period_data=ebit,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
            )

        if trailing:
            ev_to_ebit = valuation_model.get_ev_to_ebit(
                enterprise_value,
                ebit.T.rolling(trailing).sum().T,
            )
        else:
            ev_to_ebit = valuation_model.get_ev_to_ebit(enterprise_value, ebit)

        if growth:
            return calculate_growth(
                ev_to_ebit, lag=lag, rounding=rounding if rounding else self._rounding
            )

        return ev_to_ebit.round(rounding if rounding else self._rounding)
