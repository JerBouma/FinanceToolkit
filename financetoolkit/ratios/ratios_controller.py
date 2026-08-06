"""Ratios Module"""

__docformat__ = "google"


import warnings

import numpy as np
import pandas as pd

from financetoolkit.cache import policy_model, ticker_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.fmp_model import get_analyst_estimates as _get_analyst_estimates
from financetoolkit.helpers import handle_portfolio
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
from financetoolkit.utilities.statistics_model import (
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

logger = logger_model.get_logger()

# Division by zero is normal in these calculations, not a bug.
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
        start_date: str | None = None,
        end_date: str | None = None,
        api_key: str = "",
        sleep_timer: bool = False,
        user_subscription: str = "Free",
        analyst_estimates_cache: dict | None = None,
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
            api_key (str, optional): FinancialModelingPrep API key, used only by
                `get_forward_price_earnings_ratio` and `get_forward_price_earnings_growth_ratio` to
                lazily fetch (and cache) analyst estimates the first time either method is called —
                not on initialization or on any other ratio. Defaults to "", which means those two
                methods will report that the data is unavailable (requires a Premium FMP
                subscription).
            sleep_timer (bool, optional): Whether to set a sleep timer when the rate limit is
                reached while fetching analyst estimates. Only works with a Premium subscription.
                Defaults to False.
            user_subscription (str, optional): The FMP subscription plan, used only for the analyst
                estimates fetch. Defaults to "Free".
            analyst_estimates_cache (dict, optional): A mutable dict used to cache the fetched
                analyst estimates. The Toolkit passes the same dict instance in every time this
                class is (re)constructed (each access to `toolkit.ratios` builds a fresh Ratios
                instance), so this is what makes the fetch-once behaviour survive across separate
                `toolkit.ratios.<method>()` calls rather than resetting every time. Defaults to
                None, which uses a private dict scoped to this instance only.

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
        self._api_key: str = api_key
        self._sleep_timer: bool = sleep_timer
        self._user_subscription: str = user_subscription
        self._analyst_estimates_cache: dict = (
            analyst_estimates_cache if analyst_estimates_cache is not None else {}
        )
        self._available_custom_ratios_options: list[str] = []
        self._custom_ratios: pd.DataFrame = pd.DataFrame()
        self._custom_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._rounding: int | None = rounding
        self._quarterly: bool = quarterly
        self._portfolio_weights: dict | None = None
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date

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
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        toolkit.ratios.collect_all_ratios().loc['AAPL']
        ```

        Which returns:

        |                           |         2021 |         2022 |         2023 |         2024 |         2025 |
        |:--------------------------|-------------:|-------------:|-------------:|-------------:|-------------:|
        | Price-to-Cash-Flow        | 28.7847      | 17.3655      | 27.5403      | 32.6289      | 36.5905      |
        | Price-to-Free-Cash-Flow   | 32.2174      | 19.0341      | 30.5711      | 35.4618      | 41.301       |
        | Market Cap                |  2.9947e+12  |  2.12121e+12 |  3.04439e+12 |  3.8585e+12  |  4.07918e+12 |
        | Enterprise Value          |  3.09629e+12 |  2.23005e+12 |  3.13835e+12 |  3.94761e+12 |  4.15562e+12 |
        | EV-to-Sales               |  8.464       |  5.6553      |  8.188       | 10.0953      |  9.9856      |
        | EV-to-EBIT                | 27.682       | 18.274       | 26.671       | 31.9683      | 31.3091      |
        | EV-to-EBITDA              | 25.7524      | 17.0831      | 24.9432      | 29.3152      | 28.7093      |
        | EV-to-Operating-Cash-Flow | 29.7611      | 18.2565      | 28.3904      | 33.3825      | 37.2762      |
        | Tangible Asset Value      |  6.309e+10   |  5.0672e+10  |  6.2146e+10  |  5.695e+10   |  7.3733e+10  |
        | Net Current Asset Value   |  9.355e+09   | -1.8577e+10  | -1.742e+09   | -2.3405e+10  | -1.7674e+10  |
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

        # Sorted again so the index follows the financial statements' order.
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._all_ratios
        ]
        self._all_ratios = self._all_ratios.reindex(available_columns, axis=1)

        if growth:
            self._all_ratios_growth = calculate_growth(
                dataset=self._all_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        all_ratios = self._all_ratios
        all_ratios_growth = self._all_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                all_ratios_growth = calculate_standardization(
                    dataset=all_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                all_ratios = calculate_standardization(
                    dataset=all_ratios, rounding=standardize_rounding, axis="columns"
                )

        if len(self._tickers) == 1:
            return (
                all_ratios_growth.loc[self._tickers[0]]
                if growth
                else all_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (all_ratios_growth if growth else all_ratios).loc[
            :, self._start_date : self._end_date
        ]

    def collect_custom_ratios(
        self,
        custom_ratios_dict: dict | None = None,
        options: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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
            # Reordered so a formula depending on another is evaluated after it.
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
                dataset=self._custom_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        custom_ratios = self._custom_ratios
        custom_ratios_growth = self._custom_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                custom_ratios_growth = calculate_standardization(
                    dataset=custom_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                custom_ratios = calculate_standardization(
                    dataset=custom_ratios, rounding=standardize_rounding, axis="columns"
                )

        if len(self._tickers) == 1:
            return (
                custom_ratios_growth[self._tickers[0]]
                if growth
                else custom_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (custom_ratios_growth if growth else custom_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_errors
    def collect_efficiency_ratios(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects all Efficiency Ratios based on the provided data.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        toolkit.ratios.collect_efficiency_ratios().loc['AAPL']
        ```

        Which returns:

        |                                 |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:--------------------------------|-------:|-------:|-------:|-------:|-------:|
        | Accounts Payable Turnover Ratio | 4.3887 | 3.7609 | 3.3795 | 3.1975 | 3.1834 |
        | SGA-to-Revenue Ratio            | 0.0601 | 0.0636 | 0.065  | 0.0667 | 0.0663 |
        | Fixed Asset Turnover            | 1.846  | 1.8192 | 1.7979 | 1.8576 | 1.9664 |
        | Asset Turnover Ratio            | 1.0841 | 1.1206 | 1.0868 | 1.0899 | 1.1493 |
        | Operating Ratio                 | 0.7022 | 0.6971 | 0.7018 | 0.6849 | 0.6803 |
        | R&D Intensity Ratio             | 0.0599 | 0.0666 | 0.078  | 0.0802 | 0.083  |
        | S&M to Revenue Ratio            | 0      | 0      | 0      | 0.0477 | 0      |
        | G&A to Revenue Ratio            | 0      | 0      | 0      | 0.0191 | 0.0663 |
        | SBC to Revenue Ratio            | 0.0216 | 0.0229 | 0.0283 | 0.0299 | 0.0309 |
        | Deferred Revenue Ratio          | 0.0208 | 0.0201 | 0.021  | 0.0211 | 0.0218 |
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
        efficiency_ratios["R&D Intensity Ratio"] = (
            self.get_research_and_development_ratio(trailing=trailing)
        )
        efficiency_ratios["S&M to Revenue Ratio"] = (
            self.get_selling_and_marketing_ratio(trailing=trailing)
        )
        efficiency_ratios["G&A to Revenue Ratio"] = (
            self.get_general_and_administrative_ratio(trailing=trailing)
        )
        efficiency_ratios["SBC to Revenue Ratio"] = (
            self.get_stock_based_compensation_ratio(trailing=trailing)
        )
        efficiency_ratios["Deferred Revenue Ratio"] = self.get_deferred_revenue_ratio(
            trailing=trailing
        )
        efficiency_ratios["Working Capital Turnover Ratio"] = (
            self.get_working_capital_turnover_ratio(trailing=trailing)
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

        # Sorted again so the index follows the financial statements' order.
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
                dataset=self._efficiency_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        efficiency_ratios = self._efficiency_ratios
        efficiency_ratios_growth = self._efficiency_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                efficiency_ratios_growth = calculate_standardization(
                    dataset=efficiency_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                efficiency_ratios = calculate_standardization(
                    dataset=efficiency_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                efficiency_ratios_growth[self._tickers[0]]
                if growth
                else efficiency_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (efficiency_ratios_growth if growth else efficiency_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_asset_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: asset efficiency, revenue per asset.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 1.0841 | 1.1206 | 1.0868 | 1.0899 | 1.1493 |
        | TSLA | 0.942  | 1.1277 | 1.0243 | 0.8544 | 0.7298 |
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

        return finalize_dataset(
            dataset=asset_turnover_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_inventory_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: stock turnover ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | 40.0303 | 38.7899 | 37.9777 | 30.8955 | 33.9834 |
        | TSLA |  8.1593 |  6.5185 |  5.9787 |  6.2582 |  6.3692 |
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

        return finalize_dataset(
            dataset=inventory_turnover_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_days_of_inventory_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: DIO, days inventory outstanding.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL |  9.1181 |  9.4097 |  9.6109 | 11.814  | 10.7405 |
        | TSLA | 44.7344 | 55.9945 | 61.0502 | 58.3231 | 57.307  |
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

        return finalize_dataset(
            dataset=days_of_inventory_outstanding,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_days_of_sales_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: DSO, days sales outstanding, receivable days.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | 21.1517 | 25.2057 | 27.4699 | 29.3645 | 32.0949 |
        | TSLA | 12.8814 | 10.8991 | 12.1826 | 14.807  | 17.3095 |
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

        return finalize_dataset(
            dataset=days_of_sales_outstanding,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_cycle(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: business cycle, cash-to-cash cycle.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | 30.2698 | 34.6154 | 37.0808 | 41.1785 | 42.8354 |
        | TSLA | 57.6159 | 66.8936 | 73.2328 | 73.1301 | 74.6164 |
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

        return finalize_dataset(
            dataset=operating_cycle,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_accounts_payables_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: payables efficiency, AP turnover.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 4.3887 | 3.7609 | 3.3795 | 3.1975 | 3.1834 |
        | TSLA | 5.0034 | 4.795  | 5.33   | 5.9647 | 6.0153 |
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

        return finalize_dataset(
            dataset=accounts_payables_turnover_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_days_of_accounts_payable_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: DPO, days payable outstanding.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |     2023 |     2024 |     2025 |
        |:-----|--------:|--------:|---------:|---------:|---------:|
        | AAPL | 83.1683 | 97.0504 | 108.003  | 114.15   | 114.657  |
        | TSLA | 72.951  | 76.1207 |  68.4805 |  61.1935 |  60.6784 |
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

        return finalize_dataset(
            dataset=days_of_accounts_payable_outstanding,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_conversion_cycle(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        - Cash Conversion Cycle = Days of Sales in Inventory + Days of Sales Outstanding — Days of Accounts Payable
            Outstanding

        Also known as: CCC.

        Args:
            days (int, optional): The number of days to use for the calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |     2021 |     2022 |     2023 |     2024 |     2025 |
        |:-----|---------:|---------:|---------:|---------:|---------:|
        | AAPL | -52.8985 | -62.435  | -70.9225 | -72.9716 | -71.8218 |
        | TSLA | -15.3351 |  -9.2271 |   4.7523 |  11.9367 |  13.9381 |
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

        return finalize_dataset(
            dataset=cash_conversion_cycle,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_conversion_efficiency(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: CCE, cash efficiency ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.2844 | 0.3098 | 0.2884 | 0.3024 | 0.2679 |
        | TSLA | 0.2136 | 0.1807 | 0.137  | 0.1528 | 0.1555 |
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

        return finalize_dataset(
            dataset=cash_conversion_efficiency,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_receivables_turnover(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: receivables efficiency, accounts receivable turnover.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0579 | 0.0691 | 0.0753 | 0.0805 | 0.0879 |
        | TSLA | 0.0353 | 0.0299 | 0.0334 | 0.0406 | 0.0474 |
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

        return finalize_dataset(
            dataset=receivables_turnover,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_sga_to_revenue_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: SG&A ratio, selling general administrative ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0601 | 0.0636 | 0.065  | 0.0667 | 0.0663 |
        | TSLA | 0.0839 | 0.0484 | 0.0496 | 0.0527 | 0.0615 |
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

        return finalize_dataset(
            dataset=sga_to_revenue_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_fixed_asset_turnover(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: fixed asset efficiency, PP&E turnover.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 1.846  | 1.8192 | 1.7979 | 1.8576 | 1.9664 |
        | TSLA | 1.7804 | 2.1312 | 1.9665 | 1.6185 | 1.4273 |
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

        return finalize_dataset(
            dataset=fixed_asset_turnover,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_working_capital_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the working capital turnover ratio, an efficiency ratio that
        measures how effectively a company uses its working capital to generate
        revenue.

        A high working capital turnover ratio indicates that a company is generating a
        large amount of revenue relative to the working capital it employs, which can
        signal an efficient (or, if extreme, undercapitalized) operation. A low ratio
        can indicate excess inventory, slow receivables collection, or otherwise
        underutilized working capital.

        The formula is as follows:

        - Working Capital Turnover Ratio = Revenue / Average Working Capital

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Working capital turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the working capital
        turnover ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        working_capital_turnover_ratios = toolkit.ratios.get_working_capital_turnover_ratio()
        ```
        """
        working_capital = liquidity_model.get_working_capital(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if trailing:
            working_capital_turnover_ratio = (
                efficiency_model.get_working_capital_turnover_ratio(
                    self._income_statement.loc[:, "Revenue", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    working_capital.T.rolling(trailing).mean().T,
                )
            )
        else:
            working_capital_turnover_ratio = (
                efficiency_model.get_working_capital_turnover_ratio(
                    self._income_statement.loc[:, "Revenue", :],
                    working_capital.T.rolling(2).mean().T,
                )
            )

        return finalize_dataset(
            dataset=working_capital_turnover_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: operating efficiency ratio, operating expense ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.7022 | 0.6971 | 0.7018 | 0.6849 | 0.6803 |
        | TSLA | 0.8793 | 0.8324 | 0.9081 | 0.9276 | 0.9541 |
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

        return finalize_dataset(
            dataset=operating_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_research_and_development_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the research and development (R&D) intensity ratio, an efficiency
        ratio that measures how much a company reinvests in research and development
        relative to its revenue.

        This ratio is particularly relevant for comparing companies in technology,
        pharmaceutical and other innovation-driven industries, where R&D spending is
        a key driver of future growth.

        The formula is as follows:

        - R&D Intensity Ratio = Research and Development Expenses / Revenue

        Also known as: R&D intensity, R&D to sales ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: R&D intensity ratio values.

        Notes:
        - The method retrieves historical data and calculates the R&D intensity ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        rd_ratios = toolkit.ratios.get_research_and_development_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0599 | 0.0666 |  0.078 | 0.0802 | 0.083  |
        | TSLA | 0.0482 | 0.0377 |  0.041 | 0.0465 | 0.0676 |
        """
        if trailing:
            rd_ratio = efficiency_model.get_research_and_development_ratio(
                self._income_statement.loc[:, "Research and Development Expenses", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            rd_ratio = efficiency_model.get_research_and_development_ratio(
                self._income_statement.loc[:, "Research and Development Expenses", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=rd_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_selling_and_marketing_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the selling and marketing (S&M) expenses to revenue ratio, an
        efficiency ratio that measures the proportion of revenue spent on selling and
        marketing activities.

        This ratio isolates the selling and marketing component of the combined SG&A
        expense line (see `get_sga_to_revenue_ratio`), which is useful for comparing
        customer-acquisition efficiency independently of administrative overhead.

        The formula is as follows:

        - S&M to Revenue Ratio = Selling and Marketing Expenses / Revenue

        Also known as: S&M ratio, sales and marketing intensity.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: S&M to revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the S&M to revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.
        - Not every company reports Selling and Marketing Expenses separately from General
        and Administrative Expenses, in which case this ratio will be unavailable.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        sm_ratios = toolkit.ratios.get_selling_and_marketing_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL |      0 |      0 |      0 | 0.0477 |      0 |
        | TSLA |      0 |      0 |      0 | 0      |      0 |
        """
        if trailing:
            sm_ratio = efficiency_model.get_selling_and_marketing_ratio(
                self._income_statement.loc[:, "Selling and Marketing Expenses", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            sm_ratio = efficiency_model.get_selling_and_marketing_ratio(
                self._income_statement.loc[:, "Selling and Marketing Expenses", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=sm_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_general_and_administrative_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the general and administrative (G&A) expenses to revenue ratio, an
        efficiency ratio that measures the proportion of revenue spent on general and
        administrative overhead.

        This ratio isolates the administrative component of the combined SG&A expense
        line (see `get_sga_to_revenue_ratio`), which is useful for assessing overhead
        efficiency independently of selling and marketing spend.

        The formula is as follows:

        - G&A to Revenue Ratio = General and Administrative Expenses / Revenue

        Also known as: G&A ratio, overhead ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: G&A to revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the G&A to revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.
        - Not every company reports General and Administrative Expenses separately from
        Selling and Marketing Expenses, in which case this ratio will be unavailable.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ga_ratios = toolkit.ratios.get_general_and_administrative_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0      | 0      | 0      | 0.0191 | 0.0663 |
        | TSLA | 0.0839 | 0.0484 | 0.0496 | 0.0527 | 0.0615 |
        """
        if trailing:
            ga_ratio = efficiency_model.get_general_and_administrative_ratio(
                self._income_statement.loc[:, "General and Administrative Expenses", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            ga_ratio = efficiency_model.get_general_and_administrative_ratio(
                self._income_statement.loc[:, "General and Administrative Expenses", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=ga_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_stock_based_compensation_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the stock-based compensation (SBC) to revenue ratio, an efficiency
        ratio that measures how much of a company's revenue is being used to
        compensate employees through non-cash equity awards.

        A high or rising SBC-to-revenue ratio is a common quality-of-earnings flag,
        particularly for technology companies, since SBC is added back in cash flow
        from operations but represents real economic dilution for shareholders.

        The formula is as follows:

        - SBC to Revenue Ratio = Stock Based Compensation / Revenue

        Also known as: SBC intensity, equity compensation ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: SBC to revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the SBC to revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        sbc_ratios = toolkit.ratios.get_stock_based_compensation_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0216 | 0.0229 | 0.0283 | 0.0299 | 0.0309 |
        | TSLA | 0.0394 | 0.0192 | 0.0187 | 0.0205 | 0.0298 |
        """
        if trailing:
            sbc_ratio = efficiency_model.get_stock_based_compensation_ratio(
                self._cash_flow_statement.loc[:, "Stock Based Compensation", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            sbc_ratio = efficiency_model.get_stock_based_compensation_ratio(
                self._cash_flow_statement.loc[:, "Stock Based Compensation", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=sbc_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_deferred_revenue_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the deferred revenue ratio, an efficiency ratio that measures the
        size of a company's deferred revenue (payments collected for goods or
        services not yet delivered) relative to its revenue.

        This ratio is a common leading indicator for subscription and SaaS
        businesses, where a growing deferred revenue balance relative to revenue can
        signal accelerating future revenue recognition.

        The formula is as follows:

        - Deferred Revenue Ratio = Deferred Revenue / Revenue

        Also known as: deferred revenue intensity, unearned revenue ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Deferred revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the deferred revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.
        - Not every company reports Deferred Revenue on its Balance Sheet, in which case this
        ratio will be unavailable.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        deferred_revenue_ratios = toolkit.ratios.get_deferred_revenue_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0208 | 0.0201 | 0.021  | 0.0211 | 0.0218 |
        | TSLA | 0.0441 | 0.0345 | 0.0386 | 0.0426 | 0.0361 |
        """
        if trailing:
            deferred_revenue_ratio = efficiency_model.get_deferred_revenue_ratio(
                self._balance_sheet_statement.loc[:, "Deferred Revenue", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            deferred_revenue_ratio = efficiency_model.get_deferred_revenue_ratio(
                self._balance_sheet_statement.loc[:, "Deferred Revenue", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=deferred_revenue_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    def collect_liquidity_ratios(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Liquidity Ratios based on the provided data.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        liquidity_ratios.loc['AAPL']
        ```

        Which returns:

        |                                    |       2021 |        2022 |       2023 |        2024 |        2025 |
        |:-----------------------------------|-----------:|------------:|-----------:|------------:|------------:|
        | Current Ratio                      |  1.0746    |  0.8794     |  0.988     |  0.8673     |  0.8933     |
        | Quick Ratio                        |  0.7086    |  0.4967     |  0.6267    |  0.5589     |  0.5704     |
        | Cash Ratio                         |  0.4992    |  0.3137     |  0.4236    |  0.3695     |  0.3302     |
        | Working Capital                    |  9.355e+09 | -1.8577e+10 | -1.742e+09 | -2.3405e+10 | -1.7674e+10 |
        | Operating Cash Flow Ratio          |  0.8291    |  0.7933     |  0.7607    |  0.6704     |  0.6731     |
        | Operating Cash Flow to Sales Ratio |  0.2844    |  0.3098     |  0.2884    |  0.3024     |  0.2679     |
        | Short Term Coverage Ratio          | -4.7495    | -3.9423     | -4.1291    | -4.1839     | -4.5755     |
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
        liquidity_ratios["Defensive Interval Ratio"] = (
            self.get_defensive_interval_ratio(trailing=trailing)
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

        # Sorted again so the index follows the financial statements' order.
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
                dataset=self._liquidity_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        liquidity_ratios = self._liquidity_ratios
        liquidity_ratios_growth = self._liquidity_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                liquidity_ratios_growth = calculate_standardization(
                    dataset=liquidity_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                liquidity_ratios = calculate_standardization(
                    dataset=liquidity_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                liquidity_ratios_growth[self._tickers[0]]
                if growth
                else liquidity_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (liquidity_ratios_growth if growth else liquidity_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_current_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: short-term liquidity, working capital ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 1.0746 | 0.8794 | 0.988  | 0.8673 | 0.8933 |
        | TSLA | 1.3753 | 1.532  | 1.7259 | 2.0249 | 2.1644 |
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

        return finalize_dataset(
            dataset=current_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_quick_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: acid-test ratio, liquid ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.7086 | 0.4967 | 0.6267 | 0.5589 | 0.5704 |
        | TSLA | 0.9957 | 0.9411 | 1.1341 | 1.4219 | 1.5335 |
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

        return finalize_dataset(
            dataset=quick_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: most conservative liquidity ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.4992 | 0.3137 | 0.4236 | 0.3695 | 0.3302 |
        | TSLA | 0.8986 | 0.8306 | 1.012  | 1.2686 | 1.3893 |
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

        return finalize_dataset(
            dataset=cash_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_working_capital(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: net current assets, operating liquidity.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |      2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|----------:|------------:|------------:|------------:|------------:|
        | AAPL | 9.355e+09 | -1.8577e+10 | -1.742e+09  | -2.3405e+10 | -1.7674e+10 |
        | TSLA | 7.395e+09 |  1.4208e+10 |  2.0868e+10 |  2.9539e+10 |  3.6928e+10 |
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

        return finalize_dataset(
            dataset=working_capital,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_cash_flow_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: OCF ratio, cash liquidity.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.8291 | 0.7933 | 0.7607 | 0.6704 | 0.6731 |
        | TSLA | 0.5835 | 0.5513 | 0.4611 | 0.5178 | 0.465  |
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

        return finalize_dataset(
            dataset=operating_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_cash_flow_sales_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: cash flow to sales ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.2844 | 0.3098 | 0.2884 | 0.3024 | 0.2679 |
        | TSLA | 0.2136 | 0.1807 | 0.137  | 0.1528 | 0.1555 |
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

        return finalize_dataset(
            dataset=operating_cash_flow_sales_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_short_term_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the short-term coverage ratio, a liquidity ratio that measures a company's
        ability to pay off its short-term obligations with its operating cash flow.

        The short-term coverage ratio is calculated by dividing operating cash flow by short-term debt.
        It assesses the company's ability to meet its short-term obligations using its operating cash flow.

        The formula is as follows:

        - Short Term Coverage Ratio = Cash Flow from Operations / Short Term Debt

        Also known as: short-term debt coverage.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.get_short_term_coverage_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | -4.7495 | -3.9423 | -4.1291 | -4.1839 | -4.5755 |
        | TSLA | -4.882  | 27.4701 |  4.9042 |  3.7675 |  4.0998 |
        """
        if trailing:
            short_term_coverage_ratio = liquidity_model.get_short_term_coverage_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Short Term Debt", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            short_term_coverage_ratio = liquidity_model.get_short_term_coverage_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._balance_sheet_statement.loc[:, "Short Term Debt", :],
            )

        return finalize_dataset(
            dataset=short_term_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_defensive_interval_ratio(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the defensive interval ratio (DIR), a liquidity ratio that measures
        how many days a company could continue to cover its operating expenses using
        only its existing defensive (most liquid) assets, without relying on
        additional revenue.

        Unlike the current, quick, and cash ratios, which express liquidity relative
        to current liabilities, the defensive interval ratio expresses liquidity
        relative to the company's actual daily cash burn rate, making it a more direct
        measure of how long a company could survive a sudden stop in incoming cash
        flow.

        The formula is as follows:

        - Defensive Interval Ratio = (Cash and Cash Equivalents + Short Term Investments +
          Accounts Receivable) / Daily Operating Expenses

        Where Daily Operating Expenses = (Operating Expenses - Depreciation and
        Amortization) / Days, i.e. the average cash operating expenses incurred per
        day, net of the largest non-cash charge (depreciation and amortization).

        Also known as: defensive interval period, basic defense interval.

        Args:
            days (int, optional): The number of days to use for the daily operating
                expenses calculation. Defaults to 365.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Defensive interval ratio values, expressed in days.

        Notes:
        - The method retrieves historical data and calculates the defensive interval
        ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        defensive_interval_ratios = toolkit.ratios.get_defensive_interval_ratio()
        ```
        """
        if not days:
            days = 365 / 4 if self._quarterly else 365

        if trailing:
            operating_expenses = (
                self._income_statement.loc[:, "Operating Expenses", :]
                .T.rolling(trailing)
                .sum()
                .T
            )
            depreciation_and_amortization = (
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :]
                .T.rolling(trailing)
                .sum()
                .T
            )
            daily_operating_expenses = (
                operating_expenses - depreciation_and_amortization
            ) / days

            defensive_interval_ratio = liquidity_model.get_defensive_interval_ratio(
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
                daily_operating_expenses,
            )
        else:
            operating_expenses = self._income_statement.loc[:, "Operating Expenses", :]
            depreciation_and_amortization = self._cash_flow_statement.loc[
                :, "Depreciation and Amortization", :
            ]
            daily_operating_expenses = (
                operating_expenses - depreciation_and_amortization
            ) / days

            defensive_interval_ratio = liquidity_model.get_defensive_interval_ratio(
                self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
                self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                daily_operating_expenses,
            )

        return finalize_dataset(
            dataset=defensive_interval_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    def collect_profitability_ratios(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Profitability Ratios based on the provided data.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        profitability_ratios.loc['AAPL']
        ```

        Which returns:

        |                                             |   2021 |   2022 |   2023 |    2024 |   2025 |
        |:--------------------------------------------|-------:|-------:|-------:|--------:|-------:|
        | Return on Invested Capital                  | 0.5637 | 0.599  | 0.6068 |  0.6019 | 0.7038 |
        | Return on Capital Employed                  | 0.496  | 0.6139 | 0.5677 |  0.6548 | 0.6855 |
        | Return on Tangible Assets                   | 0.155  | 0.1543 | 0.1495 |  0.1425 | 0.17   |
        | Income Quality Ratio                        | 1.0988 | 1.2239 | 1.1397 |  1.2616 | 0.9953 |
        | Net Income per EBT                          | 0.867  | 0.838  | 0.8528 |  0.7591 | 0.8439 |
        | Free Cash Flow to Operating Cash Flow Ratio | 0.8935 | 0.9123 | 0.9009 |  0.9201 | 0.8859 |
        | EBT to EBIT Ratio                           | 0.9764 | 0.976  | 0.9666 |  1      | 1      |
        | EBIT to Revenue                             | 0.3058 | 0.3095 | 0.307  |  0.3158 | 0.3189 |
        | Cash Tax Rate                               | 0.2324 | 0.1643 | 0.1642 |  0.2114 | 0.3267 |
        | Tax Rate Divergence                         | 0.0994 | 0.0023 | 0.017  | -0.0295 | 0.1706 |
        """
        profitability_ratios: dict = {}

        profitability_ratios["Gross Margin"] = self.get_gross_margin(trailing=trailing)
        profitability_ratios["Operating Margin"] = self.get_operating_margin(
            trailing=trailing
        )
        profitability_ratios["Net Profit Margin"] = self.get_net_profit_margin(
            trailing=trailing
        )
        profitability_ratios["EBITDA Margin"] = self.get_ebitda_margin(
            trailing=trailing
        )
        profitability_ratios["Free Cash Flow Margin"] = self.get_free_cash_flow_margin(
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
        profitability_ratios["Cash Return on Assets"] = self.get_cash_return_on_assets(
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
        profitability_ratios["Cash Tax Rate"] = self.get_cash_tax_rate(
            trailing=trailing
        )
        profitability_ratios["Tax Rate Divergence"] = self.get_tax_rate_divergence(
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

        # Sorted again so the index follows the financial statements' order.
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
                dataset=self._profitability_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        profitability_ratios = self._profitability_ratios
        profitability_ratios_growth = self._profitability_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                profitability_ratios_growth = calculate_standardization(
                    dataset=profitability_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                profitability_ratios = calculate_standardization(
                    dataset=profitability_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                profitability_ratios_growth[self._tickers[0]]
                if growth
                else profitability_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (profitability_ratios_growth if growth else profitability_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_gross_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: gross profit margin, gross profit percentage.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.4178 | 0.4331 | 0.4413 | 0.4621 | 0.4691 |
        | TSLA | 0.2528 | 0.256  | 0.1825 | 0.1786 | 0.1803 |
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

        return finalize_dataset(
            dataset=gross_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: EBIT margin, operating income margin.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.2978 | 0.3029 | 0.2982 | 0.3151 | 0.3197 |
        | TSLA | 0.1212 | 0.1676 | 0.0919 | 0.0724 | 0.0459 |
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

        return finalize_dataset(
            dataset=operating_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_net_profit_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: net margin, bottom line margin, net income margin.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.2588 | 0.2531 | 0.2531 | 0.2397 | 0.2692 |
        | TSLA | 0.1028 | 0.1545 | 0.155  | 0.073  | 0.04   |
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

        return finalize_dataset(
            dataset=net_profit_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_ebitda_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the EBITDA margin, a profitability ratio that measures the
        percentage of revenue that remains as earnings before interest, taxes,
        depreciation and amortization (EBITDA).

        EBITDA margin approximates a company's core operating profitability before the
        effects of financing decisions (interest), tax jurisdictions, and non-cash
        accounting choices around fixed and intangible assets (depreciation and
        amortization). This makes it a commonly used metric to compare operating
        performance across companies with different capital structures, tax regimes,
        and depreciation policies.

        The formula is as follows:

        - EBITDA Margin = (Operating Income + Depreciation and Amortization) / Revenue

        Also known as: EBITDA-to-revenue ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: EBITDA margin values.

        Notes:
        - The method retrieves historical data and calculates the EBITDA margin for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ebitda_margins = toolkit.ratios.get_ebitda_margin()
        ```
        """
        if trailing:
            ebitda_margin = profitability_model.get_ebitda_margin(
                self._income_statement.loc[:, "Operating Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            ebitda_margin = profitability_model.get_ebitda_margin(
                self._income_statement.loc[:, "Operating Income", :],
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=ebitda_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_interest_burden_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: interest burden, EBIT to EBT ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        toolkit = Toolkit(["TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        interest_coverage_ratios = toolkit.ratios.get_interest_burden_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |     2024 |     2025 |
        |:-----|--------:|--------:|--------:|---------:|---------:|
        | TSLA | 17.5822 | 71.4974 | 56.9936 |  20.2171 |  12.8846 |
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

        return finalize_dataset(
            dataset=interest_burden_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_income_before_tax_profit_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Pretax Profit Margin, which is the ratio of a company's pre-tax profit to
        its revenue, indicating how much profit a company makes before paying taxes on its earnings.

        The Pretax Profit Margin is calculated by dividing the pre-tax profit by the revenue.
        It provides insight into how efficiently a company is able to generate profits from its revenue.

        The formula is as follows:

        - Pretax Profit Margin = Income Before Tax / Revenue

        Also known as: pretax margin, EBT margin.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.2985 | 0.302  | 0.2967 | 0.3158 | 0.3189 |
        | TSLA | 0.1178 | 0.1684 | 0.1031 | 0.092  | 0.0557 |
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

        return finalize_dataset(
            dataset=income_before_tax_profit_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_effective_tax_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the effective tax rate, a financial ratio that measures the
        percentage of pretax income that is paid as taxes.

        The effective tax rate is calculated by dividing the income tax expense by the
        pre-tax income.

        The formula is as follows:

        - Effective Tax Rate = Income Tax Expense / Income Before Tax

        Also known as: tax burden, actual tax rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |    2023 |   2024 |   2025 |
        |:-----|-------:|-------:|--------:|-------:|-------:|
        | AAPL | 0.133  | 0.162  |  0.1472 | 0.2409 | 0.1561 |
        | TSLA | 0.1102 | 0.0825 | -0.5015 | 0.2043 | 0.2696 |
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

        return finalize_dataset(
            dataset=effective_tax_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: ROA, asset profitability.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.2806 | 0.2836 | 0.275  | 0.2613 | 0.3093 |
        | TSLA | 0.0968 | 0.1742 | 0.1588 | 0.0624 | 0.0292 |
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

        return finalize_dataset(
            dataset=return_on_assets,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_return_on_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash return on assets (Cash ROA), a profitability ratio that
        measures how efficiently a company uses its assets to generate operating cash
        flow.

        Unlike the return on assets, which uses accrual-based net income, the cash
        return on assets uses operating cash flow, making it less sensitive to
        non-cash accounting choices (e.g. depreciation method, revenue recognition
        timing, working capital accruals). Comparing cash ROA to ROA is a useful
        earnings-quality cross-check: a cash ROA that persistently trails ROA can
        indicate that reported profits are not being converted into cash.

        The formula is as follows:

        - Cash Return on Assets = Cash Flow from Operations / Average Total Assets

        Also known as: cash ROA.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Cash return on assets (Cash ROA) values.

        Notes:
        - The method retrieves historical data and calculates the cash ROA for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        cash_roa_ratios = toolkit.ratios.get_cash_return_on_assets()
        ```
        """
        if trailing:
            cash_return_on_assets = profitability_model.get_cash_return_on_assets(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            cash_return_on_assets = profitability_model.get_cash_return_on_assets(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        return finalize_dataset(
            dataset=cash_return_on_assets,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_equity(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: ROE, shareholders equity return, equity profitability.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 1.4744 | 1.7546 | 1.7195 | 1.5741 | 1.7142 |
        | TSLA | 0.2002 | 0.3248 | 0.2739 | 0.1039 | 0.0485 |
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

        return finalize_dataset(
            dataset=return_on_equity,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_invested_capital(
        self,
        dividend_adjusted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: ROIC.

        Args:
            dividend_adjusted (bool, optional): Whether to adjust the net operating profit after taxes
            with the dividends paid. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.5637 | 0.599  | 0.6068 | 0.6019 | 0.7038 |
        | TSLA | 0.1429 | 0.2733 | 0.2403 | 0.0889 | 0.0425 |
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

        return finalize_dataset(
            dataset=return_on_invested_capital,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_income_quality_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: earnings quality, accrual ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 1.0988 | 1.2239 | 1.1397 | 1.2616 | 0.9953 |
        | TSLA | 2.0779 | 1.1701 | 0.8838 | 2.093  | 3.8869 |
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

        return finalize_dataset(
            dataset=income_quality_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_tangible_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: ROTA, tangible asset return.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.155  | 0.1543 | 0.1495 | 0.1425 | 0.17   |
        | TSLA | 0.0632 | 0.118  | 0.1114 | 0.0443 | 0.0208 |
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

        return finalize_dataset(
            dataset=return_on_tangible_assets,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_capital_employed(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: ROCE.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.496  | 0.6139 | 0.5677 | 0.6548 | 0.6855 |
        | TSLA | 0.1556 | 0.25   | 0.1304 | 0.0999 | 0.0524 |
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

        return finalize_dataset(
            dataset=return_on_capital_employed,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_net_income_per_ebt(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: net income to pre-tax income.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.867  | 0.838  | 0.8528 | 0.7591 | 0.8439 |
        | TSLA | 0.8878 | 0.9175 | 1.5002 | 0.7951 | 0.7272 |
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

        return finalize_dataset(
            dataset=net_income_per_ebt,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_operating_cash_flow_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: FCF to OCF ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.8935 | 0.9123 | 0.9009 | 0.9201 | 0.8859 |
        | TSLA | 0.3029 | 0.5129 | 0.3287 | 0.24   | 0.4218 |
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

        return finalize_dataset(
            dataset=free_cash_flow_operating_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the free cash flow margin, a profitability ratio that measures the
        percentage of revenue that is converted into free cash flow.

        Unlike the net profit margin, which can be distorted by non-cash accounting
        items (e.g. depreciation, stock-based compensation, deferred taxes) and by
        working capital timing, the free cash flow margin reflects the cash a company
        actually generates, after capital expenditures, for every dollar of revenue. A
        persistently low or declining free cash flow margin relative to the net profit
        margin can be a quality-of-earnings warning sign.

        The formula is as follows:

        - Free Cash Flow Margin = Free Cash Flow / Revenue

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Free cash flow margin values.

        Notes:
        - The method retrieves historical data and calculates the free cash flow
        margin for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        free_cash_flow_margins = toolkit.ratios.get_free_cash_flow_margin()
        ```
        """
        if trailing:
            free_cash_flow_margin = profitability_model.get_free_cash_flow_margin(
                self._cash_flow_statement.loc[:, "Free Cash Flow", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
            )
        else:
            free_cash_flow_margin = profitability_model.get_free_cash_flow_margin(
                self._cash_flow_statement.loc[:, "Free Cash Flow", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=free_cash_flow_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_tax_burden_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: tax efficiency ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.867  | 0.838  | 0.8528 | 0.7591 | 0.8439 |
        | TSLA | 0.8723 | 0.9173 | 1.504  | 0.7931 | 0.7188 |
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

        return finalize_dataset(
            dataset=tax_burden_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_EBT_to_EBIT(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: EBT to EBIT, interest burden.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.9764 | 0.976  | 0.9666 | 1      | 1      |
        | TSLA | 0.9438 | 0.9863 | 0.9846 | 0.9624 | 0.9392 |
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

        return finalize_dataset(
            dataset=EBT_to_EBIT,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_EBIT_to_revenue(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: EBIT margin, operating income to revenue.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.3058 | 0.3095 | 0.307  | 0.3158 | 0.3189 |
        | TSLA | 0.1227 | 0.1707 | 0.1049 | 0.0954 | 0.0586 |
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

        return finalize_dataset(
            dataset=EBIT_to_revenue,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_tax_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash tax rate, which measures the percentage of pretax income
        that is actually paid out in cash taxes, as opposed to the accrual-based
        effective tax rate.

        The formula is as follows:

        - Cash Tax Rate = Income Taxes Paid / Income Before Tax

        Also known as: cash effective tax rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Cash tax rate values.

        Notes:
        - The method retrieves historical data and calculates the cash tax rate for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        cash_tax_rates = toolkit.ratios.get_cash_tax_rate()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.2324 | 0.1643 | 0.1642 | 0.2114 | 0.3267 |
        | TSLA | 0.0884 | 0.0877 | 0.1122 | 0.1481 | 0      |
        """
        if trailing:
            cash_tax_rate = profitability_model.get_cash_tax_rate(
                self._cash_flow_statement.loc[:, "Income Taxes Paid", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Income Before Tax", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            cash_tax_rate = profitability_model.get_cash_tax_rate(
                self._cash_flow_statement.loc[:, "Income Taxes Paid", :],
                self._income_statement.loc[:, "Income Before Tax", :],
            )

        return finalize_dataset(
            dataset=cash_tax_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_tax_rate_divergence(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the tax rate divergence, which measures the difference between the
        cash tax rate and the accrual-based effective tax rate.

        A persistently positive divergence indicates the company is paying more in
        cash taxes than it is recognizing as tax expense (e.g. due to the reversal of
        deferred tax liabilities), while a persistently negative divergence indicates
        the opposite and can be a quality-of-earnings red flag if it stems from
        aggressive tax deferral rather than timing differences.

        The formula is as follows:

        - Tax Rate Divergence = Cash Tax Rate - Effective Tax Rate

        Also known as: cash-accrual tax gap.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Tax rate divergence values.

        Notes:
        - The method retrieves historical data and calculates the tax rate divergence for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        tax_rate_divergences = toolkit.ratios.get_tax_rate_divergence()
        ```

        Which returns:

        |      |    2021 |   2022 |   2023 |    2024 |    2025 |
        |:-----|--------:|-------:|-------:|--------:|--------:|
        | AAPL |  0.0994 | 0.0023 | 0.017  | -0.0295 |  0.1706 |
        | TSLA | -0.0218 | 0.0052 | 0.6137 | -0.0562 | -0.2696 |
        """
        tax_rate_divergence = profitability_model.get_tax_rate_divergence(
            self.get_cash_tax_rate(trailing=trailing),
            self.get_effective_tax_rate(trailing=trailing),
        )

        return finalize_dataset(
            dataset=tax_rate_divergence,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    def collect_solvency_ratios(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Solvency Ratios based on the provided data.

        Args:
            diluted (bool, optional): Whether to use diluted shares for the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        solvency_ratios.loc['AAPL']
        ```

        Which returns:

        |                                   |     2021 |     2022 |     2023 |     2024 |     2025 |
        |:----------------------------------|---------:|---------:|---------:|---------:|---------:|
        | Debt Service Coverage Ratio       |   0.8683 |   0.7757 |   0.7866 |   0.6985 |   0.8033 |
        | Equity Multiplier                 |   5.255  |   6.1862 |   6.252  |   6.0251 |   5.5418 |
        | Free Cash Flow Yield              |   0.031  |   0.0525 |   0.0327 |   0.0282 |   0.0242 |
        | Net-Debt to EBITDA Ratio          |   0.8449 |   0.8337 |   0.7468 |   0.6618 |   0.5281 |
        | Cash Flow Coverage Ratio          |   0.7621 |   0.922  |   0.892  |   0.9932 |   0.992  |
        | CAPEX Coverage Ratio              |  -9.3855 | -11.4075 | -10.087  | -12.5176 |  -8.7678 |
        | Dividend CAPEX Coverage Ratio     |  -4.0716 |  -4.781  |  -4.2543 |  -4.7913 |  -3.9623 |
        | Debt-to-Capital Ratio             |   0.6839 |   0.7233 |   0.666  |   0.6764 |   0.6038 |
        """
        solvency_ratios: dict = {}

        solvency_ratios["Debt-to-Assets Ratio"] = self.get_debt_to_assets_ratio(
            trailing=trailing
        )
        solvency_ratios["Asset Coverage Ratio"] = self.get_asset_coverage_ratio(
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
        solvency_ratios["Gross Debt to EBITDA Ratio"] = (
            self.get_gross_debt_to_ebitda_ratio(trailing=trailing)
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
        solvency_ratios["Debt-to-Capital Ratio"] = self.get_debt_to_capital_ratio(
            trailing=trailing
        )
        solvency_ratios["Preferred Dividend Coverage Ratio"] = (
            self.get_preferred_dividend_coverage_ratio(trailing=trailing)
        )
        solvency_ratios["Interest Paid to Expense Ratio"] = (
            self.get_interest_paid_to_expense_ratio(trailing=trailing)
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

        # Sorted again so the index follows the financial statements' order.
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._solvency_ratios
        ]
        self._solvency_ratios = self._solvency_ratios.reindex(available_columns, axis=1)

        if growth:
            self._solvency_ratios_growth = calculate_growth(
                dataset=self._solvency_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        solvency_ratios = self._solvency_ratios
        solvency_ratios_growth = self._solvency_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                solvency_ratios_growth = calculate_standardization(
                    dataset=solvency_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                solvency_ratios = calculate_standardization(
                    dataset=solvency_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                solvency_ratios_growth[self._tickers[0]]
                if growth
                else solvency_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (solvency_ratios_growth if growth else solvency_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_debt_to_assets_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: asset leverage, debt ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.3889 | 0.3756 | 0.3515 | 0.3262 | 0.3128 |
        | TSLA | 0.1428 | 0.0698 | 0.0898 | 0.1116 | 0.0608 |
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

        return finalize_dataset(
            dataset=debt_to_assets_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_asset_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the asset coverage ratio, a solvency ratio that measures how well a
        company's tangible assets, after settling current liabilities, can cover its
        total debt.

        This ratio is commonly used by lenders and bondholders to assess the extent to
        which a company's hard (tangible) assets would be available to repay debt
        obligations in a liquidation scenario, since intangible assets (e.g. goodwill)
        typically have little to no recovery value and current liabilities are assumed
        to be settled first out of current assets.

        The formula is as follows:

        - Asset Coverage Ratio = (Total Assets - Intangible Assets - Total Current Liabilities) / Total Debt

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Asset coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        asset_coverage_ratios = toolkit.ratios.get_asset_coverage_ratio()
        ```
        """
        if trailing:
            asset_coverage_ratio = solvency_model.get_asset_coverage_ratio(
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Intangible Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Debt", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            asset_coverage_ratio = solvency_model.get_asset_coverage_ratio(
                self._balance_sheet_statement.loc[:, "Total Assets", :],
                self._balance_sheet_statement.loc[:, "Intangible Assets", :],
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
                self._balance_sheet_statement.loc[:, "Total Debt", :],
            )

        return finalize_dataset(
            dataset=asset_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_debt_to_equity_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: D/E ratio, leverage ratio, gearing ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 2.1639 | 2.6145 | 1.9942 | 2.0906 | 1.5241 |
        | TSLA | 0.2809 | 0.1252 | 0.1505 | 0.1849 | 0.1011 |
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

        return finalize_dataset(
            dataset=debt_to_equity_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_interest_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: TIE, times interest earned.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |     2024 |    2025 |
        |:-----|--------:|--------:|--------:|---------:|--------:|
        | AAPL | 45.4567 | 44.538  | 31.9908 | -        | -       |
        | TSLA | 25.4286 | 90.0471 | 86.9103 |  35.5543 |  31.074 |
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

        return finalize_dataset(
            dataset=interest_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_equity_multiplier(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: financial leverage multiplier.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 5.255  | 6.1862 | 6.252  | 6.0251 | 5.5418 |
        | TSLA | 2.0679 | 1.8646 | 1.7255 | 1.6657 | 1.6601 |
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

        return finalize_dataset(
            dataset=equity_multiplier,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_debt_service_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: DSCR, debt service ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.8683 | 0.7757 | 0.7866 | 0.6985 | 0.8033 |
        | TSLA | 0.331  | 0.5113 | 0.3093 | 0.2455 | 0.1373 |
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

        return finalize_dataset(
            dataset=debt_service_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
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
        standardize: bool = False,
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

        Also known as: FCF yield.

        Args:
            show_daily (bool, optional): Whether to use daily data for the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for market capitalization. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.031  | 0.0525 | 0.0327 | 0.0282 | 0.0242 |
        | TSLA | 0.0029 | 0.0176 | 0.005  | 0.0025 | 0.0039 |
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
                (
                    share_prices.rolling(trailing).sum()
                    if show_daily
                    else share_prices.T.rolling(trailing).sum().T
                ),
                average_shares,
            )

            free_cash_flow_yield = solvency_model.get_free_cash_flow_yield(
                (
                    free_cash_flow.rolling(trailing).sum()
                    if show_daily
                    else free_cash_flow.T.rolling(trailing).sum().T
                ),
                market_cap,
            )
        else:
            market_cap = valuation_model.get_market_cap(share_prices, average_shares)

            free_cash_flow_yield = solvency_model.get_free_cash_flow_yield(
                free_cash_flow,
                market_cap,
            )

        return finalize_dataset(
            dataset=free_cash_flow_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_net_debt_to_ebitda_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL |  0.8449 |  0.8337 |  0.7468 |  0.6618 |  0.5281 |
        | TSLA | -0.9225 | -0.6108 | -0.5034 | -0.2022 | -0.7747 |
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

        return finalize_dataset(
            dataset=net_debt_to_ebitda_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_gross_debt_to_ebitda_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculates the gross debt to EBITDA ratio, which measures the total (gross)
        debt of the company relative to its EBITDA (Earnings Before Interest, Taxes,
        Depreciation, and Amortization).

        This differs from the Net-Debt to EBITDA Ratio in that it uses total (gross)
        debt rather than net debt (total debt minus cash and cash equivalents). Gross
        debt to EBITDA is a more conservative leverage measure since it does not
        assume that a company's cash balance would actually be used to pay down debt,
        which matters when comparing companies with restricted cash, cash earmarked
        for other purposes, or when assessing gross refinancing risk rather than net
        economic leverage.

        The formula is as follows:

        - Gross Debt to EBITDA Ratio = Total Debt / EBITDA

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Gross debt to EBITDA ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        gross_debt_to_ebitda_ratios = toolkit.ratios.get_gross_debt_to_ebitda_ratio()
        ```
        """
        if trailing:
            gross_debt_to_ebitda_ratio = solvency_model.get_gross_debt_to_ebitda_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._income_statement.loc[:, "Operating Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :]
                .T.rolling(trailing)
                .sum()
                .T,
            )
        else:
            gross_debt_to_ebitda_ratio = solvency_model.get_gross_debt_to_ebitda_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :],
                self._income_statement.loc[:, "Operating Income", :],
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
            )

        return finalize_dataset(
            dataset=gross_debt_to_ebitda_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_flow_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.7621 | 0.922  | 0.892  | 0.9932 | 0.992  |
        | TSLA | 1.2957 | 2.5616 | 1.3847 | 1.0954 | 1.7606 |
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

        return finalize_dataset(
            dataset=cash_flow_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_capex_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: capital expenditure coverage.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |     2022 |     2023 |     2024 |    2025 |
        |:-----|--------:|---------:|---------:|---------:|--------:|
        | AAPL | -9.3855 | -11.4075 | -10.087  | -12.5176 | -8.7678 |
        | TSLA | -1.4346 |  -2.053  |  -1.4896 |  -1.3157 | -1.7294 |
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

        return finalize_dataset(
            dataset=capex_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_capex_dividend_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: capex dividend coverage.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |   2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|-------:|--------:|--------:|--------:|
        | AAPL | -4.0716 | -4.781 | -4.2543 | -4.7913 | -3.9623 |
        | TSLA | -1.4346 | -2.053 | -1.4896 | -1.3157 | -1.7294 |
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

        return finalize_dataset(
            dataset=dividend_capex_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_debt_to_capital_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the debt to capital ratio, a solvency ratio that measures the
        proportion of a company's total capital (debt plus equity) that is financed
        by debt.

        Unlike the debt to equity ratio, which can theoretically exceed one or become
        negative with low or negative equity, the debt to capital ratio is bounded
        between 0 and 1 under normal circumstances, making it easier to compare
        across companies with very different capital structures.

        The formula is as follows:

        - Debt to Capital Ratio = Total Debt / (Total Debt + Total Equity)

        Also known as: capitalization ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Debt to capital ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        debt_to_capital_ratios = toolkit.ratios.get_debt_to_capital_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.6839 | 0.7233 | 0.666  | 0.6764 | 0.6038 |
        | TSLA | 0.2193 | 0.1113 | 0.1308 | 0.156  | 0.0918 |
        """
        if trailing:
            debt_to_capital_ratio = solvency_model.get_debt_to_capital_ratio(
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
            debt_to_capital_ratio = solvency_model.get_debt_to_capital_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :],
                self._balance_sheet_statement.loc[:, "Total Equity", :],
            )

        return finalize_dataset(
            dataset=debt_to_capital_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_preferred_dividend_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the preferred dividend coverage ratio, a solvency ratio that
        measures a company's ability to pay dividends owed to preferred shareholders
        out of its net income.

        The formula is as follows:

        - Preferred Dividend Coverage Ratio = Net Income / |Preferred Dividends Paid|

        Also known as: preferred dividend cover.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Preferred dividend coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.
        - This ratio is only meaningful for companies that have preferred stock outstanding;
        it will be zero-division/NaN for companies without preferred dividends.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["WFC"], api_key="FINANCIAL_MODELING_PREP_KEY")

        preferred_dividend_coverage_ratios = toolkit.ratios.get_preferred_dividend_coverage_ratio()
        ```

        Which returns:

        |     |     2021 |     2022 |     2023 |     2024 |     2025 |
        |:----|---------:|---------:|---------:|---------:|---------:|
        | WFC |  17.2763 |  11.2664 |  15.7599 |  16.9299 |  19.5543 |
        """
        if trailing:
            preferred_dividend_coverage_ratio = (
                solvency_model.get_preferred_dividend_coverage_ratio(
                    self._income_statement.loc[:, "Net Income", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            preferred_dividend_coverage_ratio = (
                solvency_model.get_preferred_dividend_coverage_ratio(
                    self._income_statement.loc[:, "Net Income", :],
                    self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :],
                )
            )

        return finalize_dataset(
            dataset=preferred_dividend_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_interest_paid_to_expense_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the interest paid to interest expense ratio, which measures how
        much of the accrual-based interest expense reported on the income statement
        was actually paid out in cash during the period.

        A ratio consistently below one can indicate that interest is being accrued
        (e.g. on payment-in-kind debt) rather than paid, while a ratio well above one
        can indicate the payment of previously accrued interest or a mismatch between
        the cash and accrual reporting periods, both of which are relevant
        quality-of-earnings signals.

        The formula is as follows:

        - Interest Paid to Expense Ratio = Interest Paid / Interest Expense

        Also known as: cash interest coverage, interest cash conversion.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Interest paid to interest expense ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        interest_paid_to_expense_ratios = toolkit.ratios.get_interest_paid_to_expense_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |     2024 |   2025 |
        |:-----|-------:|-------:|-------:|---------:|-------:|
        | AAPL | 1.0159 | 0.9775 | 0.9669 | -        | -      |
        | TSLA | 0.717  | 0.7958 | 0.8077 |   0.7914 |      0 |
        """
        if trailing:
            interest_paid_to_expense_ratio = (
                solvency_model.get_interest_paid_to_expense_ratio(
                    self._cash_flow_statement.loc[:, "Interest Paid", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._income_statement.loc[:, "Interest Expense", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            interest_paid_to_expense_ratio = (
                solvency_model.get_interest_paid_to_expense_ratio(
                    self._cash_flow_statement.loc[:, "Interest Paid", :],
                    self._income_statement.loc[:, "Interest Expense", :],
                )
            )

        return finalize_dataset(
            dataset=interest_paid_to_expense_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    def collect_valuation_ratios(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        valuation_ratios.loc['AAPL']
        ```

        Which returns:

        |                             |        2021 |         2022 |        2023 |        2024 |        2025 |
        |:----------------------------|------------:|-------------:|------------:|------------:|------------:|
        | EV-to-EBIT                  | 27.682      | 18.274       | 26.671      | 31.9683     | 31.3091     |
        | EV-to-EBITDA                | 25.7524     | 17.0831      | 24.9432     | 29.3152     | 28.7093     |
        | EV-to-Operating-Cash-Flow   | 29.7611     | 18.2565      | 28.3904     | 33.3825     | 37.2762     |
        | Tangible Asset Value        |  6.309e+10  |  5.0672e+10  |  6.2146e+10 |  5.695e+10  |  7.3733e+10 |
        | Net Current Asset Value     |  9.355e+09  | -1.8577e+10  | -1.742e+09  | -2.3405e+10 | -1.7674e+10 |
        | EV-to-Free-Cash-Flow        | 33.3102     | 20.0107      | 31.5146     | 36.2809     | 42.075      |
        | Graham Number               | 21.7378     | 20.662       | 23.2902     | 22.4928     | 28.7292     |
        | Buyback Yield               |  0.0283     |  0.0421      |  0.0255     |  0.0246     |  0.0222     |
        | Shareholder Yield           |  0.0283     |  0.0421      |  0.0255     |  0.0246     |  0.0251     |
        | SBC-Adjusted Free Cash Flow |  8.5047e+10 |  1.02405e+11 |  8.8751e+10 |  9.7119e+10 |  8.5904e+10 |
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
        valuation_ratios["Price-to-Sales"] = self.get_price_to_sales_ratio(
            diluted=diluted, trailing=trailing
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
        valuation_ratios["EV-to-Free-Cash-Flow"] = self.get_ev_to_free_cash_flow_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Buyback Yield"] = self.get_buyback_yield(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["Shareholder Yield"] = self.get_shareholder_yield(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios["SBC-Adjusted Free Cash Flow"] = (
            self.get_sbc_adjusted_free_cash_flow(trailing=trailing)
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

        # Sorted again so the index follows the financial statements' order.
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
                dataset=self._valuation_ratios,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        valuation_ratios = self._valuation_ratios
        valuation_ratios_growth = self._valuation_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                valuation_ratios_growth = calculate_standardization(
                    dataset=valuation_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                valuation_ratios = calculate_standardization(
                    dataset=valuation_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                valuation_ratios_growth[self._tickers[0]]
                if growth
                else valuation_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (valuation_ratios_growth if growth else valuation_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_earnings_per_share(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        - Earnings per Share (EPS) = (Net Income — Preferred Dividends Paid) / Weighted Average Shares

        Also known as: EPS, net income per share.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the EPS calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted earnings per share. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 5.614  | 6.1132 | 6.1341 | 6.0836 | 7.465  |
        | TSLA | 1.6341 | 3.6213 | 4.3067 | 2.0383 | 1.0754 |
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

        return finalize_dataset(
            dataset=earnings_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_revenue_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: sales per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | 21.691  | 24.1536 | 24.2393 | 25.3785 | 27.7354 |
        | TSLA | 15.8957 | 23.4423 | 27.7864 | 27.9274 | 26.8784 |
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

        return finalize_dataset(
            dataset=revenue_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

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
        standardize: bool = False,
        trailing: int | None = None,
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

        Also known as: PE ratio, P/E ratio, price-earnings ratio, earnings multiple.

        Args:
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

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

        Which returns:

        |      |     2021 |    2022 |    2023 |     2024 |    2025 |
        |:-----|---------:|--------:|--------:|---------:|--------:|
        | AAPL |  31.6299 | 21.254  | 31.3868 |  41.1631 |  36.418 |
        | TSLA | 215.568  | 34.0154 | 57.6961 | 198.126  | 418.189 |
        """
        eps = self.get_earnings_per_share(
            include_dividends,
            diluted,
            trailing=(
                trailing if trailing is not None else (4 if self._quarterly else None)
            ),
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

        return finalize_dataset(
            dataset=price_to_earnings_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

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
        standardize: bool = False,
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

        Also known as: PEG ratio, price earnings growth.

        Args:
            use_ebitda_growth_rate (bool, optional): Whether to use EBITDA growth rate for the calculation.
                Defaults to False.
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |    2023 |     2024 |    2025 |
        |:-----|-------:|-------:|--------:|---------:|--------:|
        | AAPL |  0.443 | 2.3908 | 92.3141 | -50.1989 |  1.6036 |
        | TSLA |  0.322 | 0.2797 |  3.0479 |  -3.7616 | -8.8524 |
        """
        trailing_metric = 4 if self._quarterly else 1

        if use_ebitda_growth_rate:
            growth_rate = (
                self._income_statement.loc[:, "EBITDA", :]
                .T.rolling(trailing_metric)
                .sum()
                .T
            )

            growth_rate = calculate_growth(dataset=growth_rate)
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

        return finalize_dataset(
            dataset=price_to_earnings_growth_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    def _get_or_fetch_analyst_estimates(self) -> pd.DataFrame:
        """
        Return the cached analyst estimates, fetching them on first use only.

        Used exclusively by `get_forward_price_earnings_ratio` and
        `get_forward_price_earnings_growth_ratio` — no other ratio touches analyst estimates,
        so initializing or using the rest of the Ratios module never triggers this fetch.

        The cache lives in `self._analyst_estimates_cache`, a dict the Toolkit passes in by
        reference (see `analyst_estimates_cache` on `__init__`). Every access to `toolkit.ratios`
        constructs a brand new Ratios instance, so caching the DataFrame directly on `self` would
        not survive across separate `toolkit.ratios.<method>()` calls — the shared dict is what
        makes the fetch-once behaviour actually hold.

        Returns:
            pd.DataFrame: The analyst estimates, or an empty DataFrame if no `api_key` was
            provided or the fetch failed (e.g. no Premium FMP subscription).
        """
        if "data" not in self._analyst_estimates_cache and self._api_key:
            # Same source, dataset and parameters as Toolkit.get_analyst_estimates.
            fetched_analyst_estimates, _ = ticker_model.collect_per_ticker(
                cache=get_active_cache(),
                source=policy_model.FINANCIAL_MODELING_PREP,
                dataset="analyst_estimates",
                tickers=self._tickers_without_portfolio,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={
                    "quarter": self._quarterly,
                    "start_date": self._start_date,
                },
                collector=lambda tickers: _get_analyst_estimates(
                    tickers=tickers,
                    api_key=self._api_key,
                    quarter=self._quarterly,
                    start_date=self._start_date,
                    rounding=self._rounding,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._user_subscription,
                ),
            )
            self._analyst_estimates_cache["data"] = fetched_analyst_estimates

        return self._analyst_estimates_cache.get("data", pd.DataFrame())

    @handle_portfolio
    @handle_errors
    def get_forward_price_earnings_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the forward price earnings ratio (forward P/E), a valuation ratio that compares
        the current stock price to the analyst consensus (average) EPS estimate for a future period,
        instead of the trailing EPS used by the regular P/E ratio.

        The formula is as follows:

            Forward P/E = Stock Price / Estimated EPS Average

        Note that this requires an `api_key` to be set on the Toolkit and a Premium FMP
        subscription. Analyst estimates are fetched once per Toolkit instance and cached
        across calls to this (or the other forward-looking ratio) method — never on
        initialization or on any other ratio.

        Also known as: forward P/E, projected P/E.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Forward P/E ratio for each period covered by the analyst estimates.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.get_forward_price_earnings_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | 56.1094 | 51.5965 | 48.7491 | 46.9062 | 42.6259 |
        | MSFT | 49.0076 | 41.0618 | 39.5875 | 31.2615 | 28.3729 |
        """
        analyst_estimates = self._get_or_fetch_analyst_estimates()

        if analyst_estimates.empty:
            logger.error(
                "No analyst estimates available. This requires an api_key to be set and a "
                "Premium FMP subscription."
            )
            return None

        estimated_eps = analyst_estimates.loc[:, "Estimated EPS Average", :]

        stock_price = (
            self._daily_historical_data["Adj Close"]
            .ffill()
            .iloc[-1][self._tickers_without_portfolio]
            .reindex(estimated_eps.index)
        )
        stock_price_broadcast = pd.DataFrame(
            {period: stock_price for period in estimated_eps.columns},
            index=estimated_eps.index,
        )

        forward_price_earnings_ratio = valuation_model.get_price_to_earnings_ratio(
            stock_price=stock_price_broadcast, earnings_per_share=estimated_eps
        )

        return finalize_dataset(
            dataset=forward_price_earnings_ratio,
            start_date=self._start_date,
            end_date=None,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_forward_price_earnings_growth_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the forward price earnings to growth (forward PEG) ratio, a valuation metric
        that measures the forward P/E ratio relative to the growth implied by the analyst
        consensus (average) EPS estimate versus the company's most recently reported actual EPS.

        The formula is as follows:

            - Estimated EPS Growth Rate = (Estimated EPS - Trailing EPS) / |Trailing EPS|
            - Forward PEG = Forward P/E / (Estimated EPS Growth Rate * 100)

        Note that this requires an `api_key` to be set on the Toolkit and a Premium FMP
        subscription. Analyst estimates are fetched once per Toolkit instance and cached
        across calls to this (or the other forward-looking ratio) method — never on
        initialization or on any other ratio.

        Also known as: forward PEG, PEG on forward growth.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Forward PEG ratio for each period covered by the analyst estimates.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.ratios.get_forward_price_earnings_growth_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |     2025 |
        |:-----|--------:|--------:|--------:|--------:|---------:|
        | AAPL | -2.26   | -2.8269 | -3.6172 | -4.6544 | -40.6636 |
        | MSFT | -1.1385 | -1.2822 | -1.3423 | -2.918  | -17.48   |
        """
        analyst_estimates = self._get_or_fetch_analyst_estimates()

        if analyst_estimates.empty:
            logger.error(
                "No analyst estimates available. This requires an api_key to be set and a "
                "Premium FMP subscription."
            )
            return None

        estimated_eps = analyst_estimates.loc[:, "Estimated EPS Average", :]

        trailing_eps = (
            self._income_statement.loc[:, "EPS Diluted", :]
            .ffill(axis=1)
            .iloc[:, -1]
            .reindex(estimated_eps.index)
        )
        trailing_eps_broadcast = pd.DataFrame(
            {period: trailing_eps for period in estimated_eps.columns},
            index=estimated_eps.index,
        )

        estimated_eps_growth_rate = valuation_model.get_estimated_eps_growth_rate(
            estimated_eps=estimated_eps, trailing_eps=trailing_eps_broadcast
        )

        stock_price = (
            self._daily_historical_data["Adj Close"]
            .ffill()
            .iloc[-1][self._tickers_without_portfolio]
            .reindex(estimated_eps.index)
        )
        stock_price_broadcast = pd.DataFrame(
            {period: stock_price for period in estimated_eps.columns},
            index=estimated_eps.index,
        )
        forward_price_earnings_ratio = valuation_model.get_price_to_earnings_ratio(
            stock_price=stock_price_broadcast, earnings_per_share=estimated_eps
        )

        forward_price_earnings_growth_ratio = (
            valuation_model.get_price_to_earnings_growth_ratio(
                forward_price_earnings_ratio, estimated_eps_growth_rate * 100
            )
        )

        return finalize_dataset(
            dataset=forward_price_earnings_growth_ratio,
            start_date=self._start_date,
            end_date=None,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_book_value_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the book value per share, a valuation ratio that measures the
        amount of common equity value per share outstanding.

        The book value per share is a fundamental valuation metric that reflects
        the net worth of a company attributed to each outstanding share of common stock.

        The formula is as follows:

        - Book Value per Share = (Total Shareholder Equity — Preferred Stock) / Weighted Average (Diluted) Shares

        Also known as: BVPS, net asset value per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|-------:|--------:|--------:|--------:|--------:|
        | AAPL | 3.7409 |  3.1038 |  3.9302 |  3.6961 |  4.914  |
        | TSLA | 8.9158 | 12.8645 | 17.9841 | 20.8442 | 23.2815 |
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

        return finalize_dataset(
            dataset=book_value_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_book_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: P/B ratio.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | 47.4672 | 41.8616 | 48.9873 | 67.7525 | 55.3236 |
        | TSLA | 39.5096 |  9.5752 | 13.8166 | 19.3742 | 19.3166 |
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

        return finalize_dataset(
            dataset=price_to_book_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_interest_debt_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the interest debt per share, a valuation ratio that measures the
        amount of interest expense incurred per outstanding share of a company's stock.

        The interest debt per share ratio provides insight into how much interest a company
        pays on its debt relative to its shareholder base. It can help investors assess
        the financial burden of interest expenses on the company's profitability.

        The formula is as follows:

        - Interest Debt per Share = (Interest Expense + Total Debt) / Weighted Average (Diluted) Shares

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|------------:|------------:|------------:|------------:|------------:|
        | AAPL | 3.26744e+08 | 3.61194e+08 | 5.01822e+08 | 0           | 0           |
        | TSLA | 1.41576e+08 | 1.15471e+08 | 5.67543e+07 | 8.98701e+07 | 1.42367e+08 |
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

        return finalize_dataset(
            dataset=interest_debt_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_capex_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: capital expenditure per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | -0.6573 | -0.6559 | -0.6931 | -0.6131 | -0.8474 |
        | TSLA | -2.3668 | -2.0639 | -2.5552 | -3.2424 | -2.417  |
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

        return finalize_dataset(
            dataset=capex_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_dividend_yield(
        self,
        show_daily: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: annual dividend yield, income yield.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |   2026 |
        |:-----|-------:|-------:|-------:|-------:|-------:|-------:|
        | AAPL |      0 |      0 |      0 |      0 | 0.0029 | 0.0017 |
        | TSLA |      0 |      0 |      0 |      0 | 0      | 0      |
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
            (
                (
                    dividends.rolling(trailing).sum()
                    if show_daily
                    else dividends.T.rolling(trailing).sum().T
                )
                if trailing
                else dividends
            ),
            share_prices,
        )

        return finalize_dataset(
            dataset=dividend_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_weighted_dividend_yield(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: blended dividend yield.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0048 |  0.007 | 0.0049 | 0.0039 | 0.0038 |
        | TSLA | 0      |  0     | 0      | 0      | 0      |
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
                (
                    dividends_paid.rolling(trailing).sum()
                    if show_daily
                    else dividends_paid.T.rolling(trailing).sum().T
                ),
                average_shares,
                share_prices,
            )
        else:
            weighted_dividend_yield = valuation_model.get_weighted_dividend_yield(
                dividends_paid,
                average_shares,
                share_prices,
            )

        return finalize_dataset(
            dataset=weighted_dividend_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_cash_flow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: P/CF ratio.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |     2021 |    2022 |    2023 |    2024 |     2025 |
        |:-----|---------:|--------:|--------:|--------:|---------:|
        | AAPL |  28.7847 | 17.3655 | 27.5403 | 32.6289 |  36.5905 |
        | TSLA | 103.745  | 29.0716 | 65.2832 | 94.6614 | 107.589  |
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
                (
                    cash_flow_from_operations.rolling(trailing).sum()
                    if show_daily
                    else cash_flow_from_operations.T.rolling(trailing).sum().T
                ),
            )
        else:
            price_to_cash_flow_ratio = valuation_model.get_price_to_cash_flow_ratio(
                market_cap, cash_flow_from_operations
            )

        return finalize_dataset(
            dataset=price_to_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_free_cash_flow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the price to free cash flow ratio, a valuation ratio that compares a
        company's market price to its free cash flow per share.

        This ratio provides insight into how the market values a company's ability to generate free cash flow.

        The formula is as follows:

        - Price to Free Cash Flow Ratio = Market Cap / Free Cash Flow

        Also known as: P/FCF ratio.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |     2021 |    2022 |     2023 |     2024 |    2025 |
        |:-----|---------:|--------:|---------:|---------:|--------:|
        | AAPL |  32.2174 | 19.0341 |  30.5711 |  35.4618 |  41.301 |
        | TSLA | 342.45   | 56.6804 | 198.621  | 394.48   | 255.082 |
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
                    (
                        free_cash_flow.rolling(trailing).sum()
                        if show_daily
                        else free_cash_flow.T.rolling(trailing).sum().T
                    ),
                )
            )
        else:
            price_to_free_cash_flow_ratio = (
                valuation_model.get_price_to_free_cash_flow_ratio(
                    market_cap, free_cash_flow
                )
            )

        return finalize_dataset(
            dataset=price_to_free_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_sales_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the price to sales ratio (P/S), a valuation ratio that compares a
        company's market capitalization to its total revenue.

        The price to sales ratio is particularly useful for valuing companies that are
        not yet profitable (and therefore have no meaningful P/E ratio), since revenue
        is typically positive even when earnings are not, and is less susceptible to
        accounting distortions than earnings-based multiples. It is, however, less
        informative than earnings- or cash-flow-based multiples for mature, profitable
        companies since it ignores profitability and cost structure entirely.

        The formula is as follows:

        - Price to Sales Ratio = Market Cap / Revenue

        Also known as: P/S ratio, sales multiple.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Price to sales ratio values.

        Notes:
        - The method retrieves historical data and calculates the price to sales ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the price to sales ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        price_to_sales_ratio = toolkit.ratios.get_price_to_sales_ratio()
        ```
        """
        market_cap = self.get_market_cap(
            diluted=diluted,
            trailing=trailing if trailing else None,
            show_daily=show_daily,
        )

        revenue = self._income_statement.loc[:, "Revenue", :]

        if show_daily:
            revenue = map_period_data_to_daily_data(
                period_data=revenue,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
            )

        if trailing:
            price_to_sales_ratio = valuation_model.get_price_to_sales_ratio(
                market_cap,
                (
                    revenue.rolling(trailing).sum()
                    if show_daily
                    else revenue.T.rolling(trailing).sum().T
                ),
            )
        else:
            price_to_sales_ratio = valuation_model.get_price_to_sales_ratio(
                market_cap, revenue
            )

        return finalize_dataset(
            dataset=price_to_sales_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
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
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculates the market capitalization of the company.

        Market capitalization, often referred to as "market cap," is the total value of a company's
        outstanding shares of stock in the stock market. It is calculated by multiplying the current
        market price per share by the total number of outstanding shares.

        The formula is as follows:

        - Market Capitalization = Share Price * Weighted Average (Diluted) Shares

        Also known as: market capitalization.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|------------:|------------:|------------:|------------:|------------:|
        | AAPL | 2.9947e+12  | 2.12121e+12 | 3.04439e+12 | 3.8585e+12  | 4.07918e+12 |
        | TSLA | 1.19275e+12 | 4.2805e+11  | 8.65394e+11 | 1.41263e+12 | 1.58661e+12 |
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
                (
                    share_prices.rolling(trailing).mean()
                    if show_daily
                    else share_prices.T.rolling(trailing).mean().T
                ),
                average_shares,
            )
        else:
            market_cap = valuation_model.get_market_cap(share_prices, average_shares)

        return finalize_dataset(
            dataset=market_cap,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_enterprise_value(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: EV, total enterprise value.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|------------:|------------:|------------:|------------:|------------:|
        | AAPL | 3.09629e+12 | 2.23005e+12 | 3.13835e+12 | 3.94761e+12 | 4.15562e+12 |
        | TSLA | 1.18544e+12 | 4.1874e+11  | 8.59544e+11 | 1.41088e+12 | 1.5792e+12  |
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
                (
                    total_debt.rolling(trailing).mean()
                    if show_daily
                    else total_debt.T.rolling(trailing).mean().T
                ),
                (
                    minority_interest.rolling(trailing).mean()
                    if show_daily
                    else minority_interest.T.rolling(trailing).mean().T
                ),
                (
                    preferred_stock.rolling(trailing).mean()
                    if show_daily
                    else preferred_stock.T.rolling(trailing).mean().T
                ),
                (
                    cash_and_cash_equivalents.rolling(trailing).mean()
                    if show_daily
                    else cash_and_cash_equivalents.T.rolling(trailing).mean().T
                ),
            )
        else:
            enterprise_value = valuation_model.get_enterprise_value(
                market_cap,
                total_debt,
                minority_interest,
                preferred_stock,
                cash_and_cash_equivalents,
            )

        return finalize_dataset(
            dataset=enterprise_value,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_sales_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: EV/Sales.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |   2022 |   2023 |    2024 |    2025 |
        |:-----|--------:|-------:|-------:|--------:|--------:|
        | AAPL |  8.464  | 5.6553 | 8.188  | 10.0953 |  9.9856 |
        | TSLA | 22.0248 | 5.1403 | 8.8821 | 14.4425 | 16.6535 |
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
                (
                    revenue.rolling(trailing).sum()
                    if show_daily
                    else revenue.T.rolling(trailing).sum().T
                ),
            )
        else:
            ev_to_sales_ratio = valuation_model.get_ev_to_sales_ratio(
                enterprise_value, revenue
            )

        return finalize_dataset(
            dataset=ev_to_sales_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_ebitda_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the enterprise value over EBITDA ratio, a valuation ratio that
        measures a company's total value (including debt and equity) relative to its EBITDA.

        This ratio helps investors understand how many times the enterprise value exceeds the
        company's EBITDA, providing insights into the company's debt load and operating performance.

        The formula is as follows:

        - Enterprise Value to EBITDA Ratio = Enterprise Value / EBITDA

        Also known as: EV/EBITDA.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |     2021 |    2022 |    2023 |     2024 |     2025 |
        |:-----|---------:|--------:|--------:|---------:|---------:|
        | AAPL |  25.7524 | 17.0831 | 24.9432 |  29.3152 |  28.7093 |
        | TSLA | 125.656  | 24.3467 | 63.3975 | 113.379  | 150.357  |
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
                (
                    operating_income.rolling(trailing).sum()
                    if show_daily
                    else operating_income.T.rolling(trailing).sum().T
                ),
                (
                    depreciation_and_amortization.rolling(trailing).sum()
                    if show_daily
                    else depreciation_and_amortization.T.rolling(trailing).sum().T
                ),
            )
        else:
            ev_to_ebitda_ratio = valuation_model.get_ev_to_ebitda_ratio(
                enterprise_value, operating_income, depreciation_and_amortization
            )

        return finalize_dataset(
            dataset=ev_to_ebitda_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_operating_cashflow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |     2021 |    2022 |    2023 |    2024 |     2025 |
        |:-----|---------:|--------:|--------:|--------:|---------:|
        | AAPL |  29.7611 | 18.2565 | 28.3904 | 33.3825 |  37.2762 |
        | TSLA | 103.109  | 28.4392 | 64.8419 | 94.5442 | 107.086  |
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
                    (
                        cash_flow_from_operations.rolling(trailing).sum()
                        if show_daily
                        else cash_flow_from_operations.T.rolling(trailing).sum().T
                    ),
                )
            )
        else:
            ev_to_operating_cashflow_ratio = (
                valuation_model.get_ev_to_operating_cashflow_ratio(
                    enterprise_value, cash_flow_from_operations
                )
            )

        return finalize_dataset(
            dataset=ev_to_operating_cashflow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
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
        standardize: bool = False,
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

        Also known as: E/P ratio, inverse PE ratio.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            include_dividends (bool, optional): Whether to include dividends in the calculation. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0316 | 0.047  | 0.0319 | 0.0243 | 0.0275 |
        | TSLA | 0.0046 | 0.0294 | 0.0173 | 0.005  | 0.0024 |
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

        return finalize_dataset(
            dataset=earnings_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_dividend_payout_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: payout ratio.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.1528 | 0.1487 | 0.1549 | 0.1625 | 0.1377 |
        | TSLA | 0      | 0      | 0      | 0      | 0      |
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

        return finalize_dataset(
            dataset=payout_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_reinvestment_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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

        Also known as: plowback ratio, retention rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.8472 | 0.8513 | 0.8451 | 0.8375 | 0.8623 |
        | TSLA | 1      | 1      | 1      | 1      | 1      |
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

        return finalize_dataset(
            dataset=reinvestment_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_tangible_asset_value(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the tangible asset value, a financial metric that represents the total value
        of a company's assets that can be used to generate revenue. Tangible assets are those
        physical assets that have a finite monetary value and can be sold, used, or consumed.

        The formula is as follows:

        - Tangible Asset Value = Total Assets — Total Liabilities — Goodwill

        Also known as: tangible assets, net tangible assets.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |       2021 |       2022 |       2023 |       2024 |       2025 |
        |:-----|-----------:|-----------:|-----------:|-----------:|-----------:|
        | AAPL | 6.309e+10  | 5.0672e+10 | 6.2146e+10 | 5.695e+10  | 7.3733e+10 |
        | TSLA | 3.1383e+10 | 4.5704e+10 | 6.3356e+10 | 7.3436e+10 | 8.2608e+10 |
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

        return finalize_dataset(
            dataset=tangible_asset_value,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_net_current_asset_value(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the net current asset value, a financial metric that represents the total value
        of a company's current assets minus its current liabilities. It indicates the extent to
        which a company's short-term assets exceed its short-term liabilities.

        The formula is as follows:

        - Net Current Asset Value = Total Current Assets — Total Current Liabilities

        Also known as: NCAV, net current asset value, Graham number.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |      2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|----------:|------------:|------------:|------------:|------------:|
        | AAPL | 9.355e+09 | -1.8577e+10 | -1.742e+09  | -2.3405e+10 | -1.7674e+10 |
        | TSLA | 7.395e+09 |  1.4208e+10 |  2.0868e+10 |  2.9539e+10 |  3.6928e+10 |
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

        return finalize_dataset(
            dataset=net_current_asset_value,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_ebit(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        Which returns:

        |      |    2021 |   2022 |    2023 |     2024 |     2025 |
        |:-----|--------:|-------:|--------:|---------:|---------:|
        | AAPL |  27.682 | 18.274 | 26.671  |  31.9683 |  31.3091 |
        | TSLA | 179.531 | 30.11  | 84.6508 | 151.431  | 284.285  |
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
                (
                    ebit.rolling(trailing).sum()
                    if show_daily
                    else ebit.T.rolling(trailing).sum().T
                ),
            )
        else:
            ev_to_ebit = valuation_model.get_ev_to_ebit(enterprise_value, ebit)

        return finalize_dataset(
            dataset=ev_to_ebit,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_free_cash_flow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the EV to free cash flow ratio, a valuation ratio that compares a
        company's enterprise value (EV) to its free cash flow.

        Unlike EV to Operating Cash Flow, this ratio nets out capital expenditures,
        giving a valuation multiple based on the cash actually available to all
        capital providers after reinvestment in the business.

        The formula is as follows:

        - EV to Free Cash Flow Ratio = Enterprise Value / Free Cash Flow

        Also known as: EV/FCF.

        Args:
            show_daily (bool, optional): Whether to show daily data. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: EV to free cash flow ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ev_to_fcf_ratio = toolkit.ratios.get_ev_to_free_cash_flow_ratio()
        ```

        Which returns:

        |      |     2021 |    2022 |     2023 |     2024 |    2025 |
        |:-----|---------:|--------:|---------:|---------:|--------:|
        | AAPL |  33.3102 | 20.0107 |  31.5146 |  36.2809 |  42.075 |
        | TSLA | 340.351  | 55.4475 | 197.279  | 393.991  | 253.891 |
        """
        enterprise_value = self.get_enterprise_value(
            diluted=diluted,
            trailing=trailing if trailing else None,
            show_daily=show_daily,
        )

        free_cash_flow = self._cash_flow_statement.loc[:, "Free Cash Flow", :]

        if show_daily:
            free_cash_flow = map_period_data_to_daily_data(
                period_data=free_cash_flow,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
            )

        if trailing:
            ev_to_free_cash_flow_ratio = valuation_model.get_ev_to_free_cash_flow_ratio(
                enterprise_value,
                (
                    free_cash_flow.rolling(trailing).sum()
                    if show_daily
                    else free_cash_flow.T.rolling(trailing).sum().T
                ),
            )
        else:
            ev_to_free_cash_flow_ratio = valuation_model.get_ev_to_free_cash_flow_ratio(
                enterprise_value, free_cash_flow
            )

        return finalize_dataset(
            dataset=ev_to_free_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_buyback_yield(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the buyback yield, a valuation ratio that measures the net amount
        of common stock repurchased (net of new shares issued) relative to the
        company's market capitalization.

        A positive buyback yield means the company is a net repurchaser of its own
        stock (shareholder-friendly), while a negative buyback yield means the
        company is a net issuer of new shares (dilutive).

        The formula is as follows:

        - Buyback Yield = -(Common Stock Purchased + Common Stock Issued) / Market Capitalization

        Also known as: net repurchase yield.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Buyback yield values.

        Notes:
        - The method retrieves historical data and calculates the buyback yield for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        buyback_yields = toolkit.ratios.get_buyback_yield()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL |  0.0283 |  0.0421 |  0.0255 |  0.0246 |  0.0222 |
        | TSLA | -0.0006 | -0.0013 | -0.0008 | -0.0009 | -0.0001 |
        """
        market_cap = self.get_market_cap(diluted=diluted, trailing=trailing)

        if trailing:
            buyback_yield = valuation_model.get_buyback_yield(
                self._cash_flow_statement.loc[:, "Common Stock Purchased", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._cash_flow_statement.loc[:, "Common Stock Issued", :]
                .T.rolling(trailing)
                .sum()
                .T,
                market_cap,
            )
        else:
            buyback_yield = valuation_model.get_buyback_yield(
                self._cash_flow_statement.loc[:, "Common Stock Purchased", :],
                self._cash_flow_statement.loc[:, "Common Stock Issued", :],
                market_cap,
            )

        return finalize_dataset(
            dataset=buyback_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_shareholder_yield(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the total shareholder yield, a valuation ratio that combines the
        dividend yield and the buyback yield to measure the total cash returned to
        shareholders relative to the company's market capitalization.

        The formula is as follows:

        - Shareholder Yield = Dividend Yield + Buyback Yield

        Also known as: total shareholder yield, total return of capital.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: Shareholder yield values.

        Notes:
        - The method retrieves historical data and calculates the shareholder yield for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        shareholder_yields = toolkit.ratios.get_shareholder_yield()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL |  0.0283 |  0.0421 |  0.0255 |  0.0246 |  0.0251 |
        | TSLA | -0.0006 | -0.0013 | -0.0008 | -0.0009 | -0.0001 |
        """
        dividend_yield = self.get_dividend_yield(trailing=trailing)
        buyback_yield = self.get_buyback_yield(diluted=diluted, trailing=trailing)

        dividend_yield_columns = [
            column
            for column in dividend_yield.columns
            if column in buyback_yield.columns
        ]

        shareholder_yield = valuation_model.get_shareholder_yield(
            dividend_yield.loc[:, dividend_yield_columns],
            buyback_yield.loc[:, dividend_yield_columns],
        )

        return finalize_dataset(
            dataset=shareholder_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_sbc_adjusted_free_cash_flow(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the stock-based compensation (SBC) adjusted free cash flow, which
        deducts non-cash SBC expenses from free cash flow to give a more conservative
        view of the cash actually available to shareholders.

        Free cash flow already excludes SBC as a cash expense (it is added back in
        the cash flow from operations), which can overstate the cash available to
        shareholders. Subtracting SBC treats it as if it were a real cash cost, which
        is a common quality-of-earnings adjustment, especially for companies that
        rely heavily on equity compensation.

        The formula is as follows:

        - SBC-Adjusted Free Cash Flow = Free Cash Flow - Stock Based Compensation

        Also known as: SBC-adjusted FCF.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: SBC-adjusted free cash flow values.

        Notes:
        - The method retrieves historical data and calculates the SBC-adjusted free cash flow
        for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        sbc_adjusted_fcf = toolkit.ratios.get_sbc_adjusted_free_cash_flow()
        ```

        Which returns:

        |      |       2021 |        2022 |       2023 |       2024 |       2025 |
        |:-----|-----------:|------------:|-----------:|-----------:|-----------:|
        | AAPL | 8.5047e+10 | 1.02405e+11 | 8.8751e+10 | 9.7119e+10 | 8.5904e+10 |
        | TSLA | 1.362e+09  | 5.992e+09   | 2.545e+09  | 1.582e+09  | 3.395e+09  |
        """
        if trailing:
            sbc_adjusted_free_cash_flow = (
                valuation_model.get_sbc_adjusted_free_cash_flow(
                    self._cash_flow_statement.loc[:, "Free Cash Flow", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    self._cash_flow_statement.loc[:, "Stock Based Compensation", :]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            sbc_adjusted_free_cash_flow = (
                valuation_model.get_sbc_adjusted_free_cash_flow(
                    self._cash_flow_statement.loc[:, "Free Cash Flow", :],
                    self._cash_flow_statement.loc[:, "Stock Based Compensation", :],
                )
            )

        return finalize_dataset(
            dataset=sbc_adjusted_free_cash_flow,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
        )
