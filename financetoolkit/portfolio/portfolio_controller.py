"""Portfolio Module"""

import os
import shutil
from importlib import resources

import pandas as pd

from financetoolkit.portfolio import helpers, overview_model, portfolio_model
from financetoolkit.risk import risk_model
from financetoolkit.toolkit_controller import Toolkit
from financetoolkit.utilities import logger_model, validation_model
from financetoolkit.utilities.statistics_model import apply_rounding, to_period_index

logger = logger_model.get_logger()

# pylint: disable=too-many-instance-attributes,too-many-lines,line-too-long,too-many-locals
# pylint: disable=too-many-function-args,too-many-public-methods
# ruff: noqa: E501


class Portfolio:
    """
    A class for managing and analyzing your portfolio.

    The Portfolio class provides functionality for managing and analyzing your portfolio. It allows
    you to read and consolidate cash flow data, calculate key statistics and performance metrics,
    and generate visualizations to gain insights into your portfolio's performance.

    The class uses a configuration file in YAML format to define various settings and columns used
    in cash flow analysis. The configuration file should specify date columns, description columns,
    amount columns, and optionally cost/income columns.

    The Portfolio class also provides methods for collecting historical price data, calculating
    performance metrics, and generating visualizations to help you understand your portfolio's
    performance and make informed investment decisions.
    """

    def __init__(
        self,
        portfolio_dataset: pd.DataFrame | str | None = None,
        benchmark_ticker: str | None = None,
        api_key: str = "",
        quarterly: bool = False,
        example: bool = False,
        configuration_file: str | None = None,
        rounding: int = 4,
    ):
        """
        Initialize the Portfolio class with the provided configuration file and portfolio dataset.

        This constructor sets up a portfolio management instance, configuring it with a dataset (either as a
        pandas DataFrame or a file path to an Excel/CSV file), a benchmark ticker for performance comparison,
        and various settings specified through a configuration file (YAML format). The configuration file defines
        key columns for cash flow analysis, such as date columns, asset tickers, price, volume, and transaction costs.
        It also allows for the use of quarterly data and fetching of historical financial data via an API.

        Args:
            portfolio_dataset (pd.DataFrame | str | None): A pandas DataFrame containing the portfolio dataset,
                or a file path to an Excel/CSV file containing the portfolio data. If None, the dataset must
                be loaded later using the `read_portfolio_dataset` method.
            benchmark_ticker (str | None): The ticker symbol for the benchmark asset used for performance comparison.
                If None, the benchmark ticker specified in the configuration file will be used.
            api_key (str): The API key for accessing financial data and historical metrics. If not provided, only
                basic historical data and indicators are available.
            quarterly (bool): Flag to specify whether to use quarterly data for performance metrics. Defaults to False
                (yearly data).
            example (bool): Flag to use example configuration and dataset files for demonstration purposes.
                If True, example files are downloaded and used.
            configuration_file (str | None): Path to a YAML configuration file defining portfolio settings.
                If None, the default configuration file is used.
            rounding (int): The number of decimal places to round the outputs. Defaults to 4 decimal places.

        Raises:
            ValueError: If the provided configuration file is not in YAML format.
            ValueError: If no portfolio dataset is provided and `example` is set to False.

        As an example:

        ```python
        from financetoolkit import Portfolio

        # Download porfolio files
        portfolio = Portfolio()

        # Load the portfolio dataset
        portfolio = Portfolio(portfolio_dataset="portfolio_template.xlsx")

        # Load an example portfolio instead
        portfolio = Portfolio(example=True)

        # Use an API key to access all features
        portfolio = Portfolio(
            portfolio_dataset="portfolio_template.xlsx",
            api_key="FINANCIAL_MODELING_PREP_KEY")
        ```
        """
        example_xlsx_path = str(
            resources.files(__package__).joinpath(
                "example_datasets/example_portfolio.xlsx"
            )
        )
        example_csv_path = str(
            resources.files(__package__).joinpath(
                "example_datasets/example_portfolio.csv"
            )
        )

        # An explicit None check is required because the truth value of a DataFrame is ambiguous, which made passing a DataFrame raise a ValueError here.
        if portfolio_dataset is None and not example:
            example = True
            logger.info(
                "No portfolio dataset provided thus loading the example portfolio for demonstration purposes.\n"
                "Please find the templates in your current directory under the names 'portfolio_template.xlsx' "
                "and 'portfolio_template.csv'.\nChoose your preferred format and provide the path within the "
                "portfolio_dataset parameter."
            )

            if "portfolio_template.xlsx" not in os.listdir():
                shutil.copy(example_xlsx_path, "portfolio_template.xlsx")
            if "portfolio_template.csv" not in os.listdir():
                shutil.copy(example_csv_path, "portfolio_template.csv")

        portfolio_dataset = (
            example_xlsx_path
            if example or portfolio_dataset is None
            else portfolio_dataset
        )

        self._configuration_file = (
            configuration_file
            if configuration_file
            else str(resources.files(__package__).joinpath("config.yaml"))
        )

        if self._configuration_file.endswith(".yaml"):
            self._cfg: dict[str, dict] = helpers.read_yaml_file(
                location=self._configuration_file
            )
        else:
            raise ValueError("File type not supported. Please use .yaml")

        self._rounding: int = rounding
        self._quarterly: bool = quarterly
        self._benchmark_ticker = (
            benchmark_ticker
            if benchmark_ticker
            else self._cfg["general"]["benchmark_ticker"]
        )
        self._yearly_overview: pd.DataFrame = pd.DataFrame()
        self._quarterly_overview: pd.DataFrame = pd.DataFrame()
        self._monthly_overview: pd.DataFrame = pd.DataFrame()
        self._yearly_cash_flow_dataset: pd.DataFrame = pd.DataFrame()
        self._quarterly_cash_flow_dataset: pd.DataFrame = pd.DataFrame()
        self._monthly_cash_flow_dataset: pd.DataFrame = pd.DataFrame()

        # Tickers
        self._ticker_combinations: dict[str, str] = {}
        self._original_ticker_combinations: dict[str, str] = {}

        # Historical Data
        self._daily_historical_data: pd.DataFrame = pd.DataFrame()
        self._weekly_historical_data: pd.DataFrame = pd.DataFrame()
        self._monthly_historical_data: pd.DataFrame = pd.DataFrame()
        self._quarterly_historical_data: pd.DataFrame = pd.DataFrame()
        self._yearly_historical_data: pd.DataFrame = pd.DataFrame()
        self._historical_statistics: pd.DataFrame = pd.DataFrame()

        # Benchmark Historical Data
        self._benchmark_tickers: dict[str, str] = {}
        self._daily_benchmark_data: pd.DataFrame = pd.DataFrame()
        self._weekly_benchmark_data: pd.DataFrame = pd.DataFrame()
        self._monthly_benchmark_data: pd.DataFrame = pd.DataFrame()
        self._quarterly_benchmark_data: pd.DataFrame = pd.DataFrame()
        self._yearly_benchmark_data: pd.DataFrame = pd.DataFrame()
        self._benchmark_prices: pd.DataFrame = pd.DataFrame()
        self._benchmark_specific_prices: pd.Series = pd.Series()
        self._benchmark_prices_per_ticker: pd.DataFrame = pd.DataFrame()
        self._latest_benchmark_price: pd.Series = pd.Series()
        self._portfolio_volatilities: pd.Series = pd.Series()
        self._portfolio_beta: pd.Series = pd.Series()

        # Portfolio Overveiw
        self._portfolio_overview: pd.DataFrame = pd.DataFrame()
        self._portfolio_performance: pd.DataFrame = pd.DataFrame()
        self._transactions_performance: pd.DataFrame = pd.DataFrame()
        self._portfolio_dataset: pd.DataFrame = pd.DataFrame()
        self._positions_overview: pd.DataFrame = pd.DataFrame()
        self._transactions_overview: pd.DataFrame = pd.DataFrame()

        # Finance Toolkit Initialization
        # A copied documentation example passes the placeholder key, treated as no key at all.
        self._api_key: str = validation_model.resolve_api_key(api_key)
        self._tickers: list = []
        self._toolkit: Toolkit | None = None
        self._toolkit_instance: Toolkit | None = None
        self._benchmark_toolkit: Toolkit | None = None
        self._currency_toolkit: Toolkit | None = None
        self._latest_price: pd.Series = pd.Series()
        self._daily_currency_data: pd.DataFrame = pd.DataFrame()

        # Column Names
        self._date_column: str = self._cfg["general"]["date_columns"]
        self._name_column: str = self._cfg["general"]["name_columns"]
        self._ticker_column: str = self._cfg["general"]["ticker_columns"]
        self._price_column: str = self._cfg["general"]["price_columns"]
        self._volume_column: str = self._cfg["general"]["volume_columns"]
        self._costs_column: str = self._cfg["general"]["costs_columns"]

        # Portfolio Dataset
        self._portfolio_dataset_path: str | list = (
            portfolio_dataset if isinstance(portfolio_dataset, str) else []
        )

        self._raw_portfolio_dataset: pd.DataFrame = (
            portfolio_dataset
            if isinstance(portfolio_dataset, pd.DataFrame)
            else pd.DataFrame()
        )

        self.read_portfolio_dataset()

    @property
    def toolkit(self) -> Toolkit | pd.DataFrame:
        """
        Converts the Portfolio to a Finance Toolkit object.

        This method converts the Portfolio object to a Finance Toolkit object, enabling the
        use of the Toolkit's 500+ financial methods for the portfolio's assets. If the
        historical data of the assets or the benchmark cannot be collected, an empty
        DataFrame is returned instead.

        Next to the historical data, the portfolio weights are also
        loaded in the Toolkit class. This, together with the "Portfolio" ticker, enables
        the possibility to calculate any Toolkit metric for all assets in the portfolio
        in combination with the Portfolio itself which is a weighted average of other
        results based on the portfolio weights over time.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit = portfolio.toolkit

        toolkit.ratios.get_net_profit_margin()
        ```

        Which returns:

        |           |    2020 |   2021 |    2022 |   2023 |   2024 |
        |:----------|--------:|-------:|--------:|-------:|-------:|
        | AAPL      |  0.2091 | 0.2588 |  0.2531 | 0.2531 | 0.2397 |
        | ALGN      |  0.7184 | 0.1953 |  0.0968 | 0.1152 | 0.1054 |
        | AMD       |  0.2551 | 0.1924 |  0.0559 | 0.0377 | 0.0636 |
        | AMZN      |  0.0553 | 0.071  | -0.0053 | 0.0529 | 0.0929 |
        | ASML      |  0.2542 | 0.3161 |  0.2656 | 0.2844 | 0.2679 |
        | AVGO      |  0.1115 | 0.2345 |  0.338  | 0.3931 | 0.1143 |
        | BAC       |  0.1757 | 0.3256 |  0.2261 | 0.1446 | 0.1325 |
        | BLDR      |  0.0366 | 0.0867 |  0.121  | 0.0901 | 0.0657 |
        | CAMT      |  0.1397 | 0.2239 |  0.2525 | 0.2528 | 0.2787 |
        | CWST      |  0.1176 | 0.0462 |  0.0489 | 0.0201 | 0.0087 |
        | FICO      |  0.1826 | 0.2978 |  0.2712 | 0.2837 | 0.2986 |
        | FIX       |  0.0526 | 0.0466 |  0.0594 | 0.0621 | 0.0743 |
        | GOOGL     |  0.2206 | 0.2951 |  0.212  | 0.2401 | 0.286  |
        | KHC       |  0.0136 | 0.0389 |  0.0892 | 0.1072 | 0.1062 |
        | META      |  0.339  | 0.3338 |  0.199  | 0.2898 | 0.3791 |
        | MPWR      |  0.1947 | 0.2004 |  0.2439 | 0.2347 | 0.8095 |
        | MSFT      |  0.3096 | 0.3645 |  0.3669 | 0.3415 | 0.3596 |
        | NFLX      |  0.1105 | 0.1723 |  0.1421 | 0.1604 | 0.2234 |
        | NVDA      |  0.2598 | 0.3623 |  0.1619 | 0.4885 | 0.5585 |
        | OXY       | -0.9146 | 0.0582 |  0.3428 | 0.1324 | 0.0872 |
        | SKY       |  0.0597 | 0.1124 |  0.1542 | 0.0724 | 0.0799 |
        | WMT       |  0.0242 | 0.0239 |  0.0191 | 0.0239 | 0.0285 |
        | Portfolio |  0.2509 | 0.2266 |  0.2051 | 0.2171 | 0.2769 |
        """
        if self._api_key is None:
            logger.error(
                "The parameter api_key is not set. Therefore, only historical data and "
                "indicators are available. Consider obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan has a limit of 5 years of fundamental data and has no quarterly data. "
                "You can get 15% off by using the above affiliate link to get access to 30+ years "
                "of (quarterly) data."
            )

        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()
        if self._weekly_benchmark_data.empty:
            self.collect_benchmark_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        symbols = list(self._tickers) + ["Portfolio"]

        historical_columns = self._daily_historical_data.columns.get_level_values(
            0
        ).unique()

        benchmark_data = self._daily_benchmark_data
        portfolio_weights: dict | None = {}

        for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            self.get_portfolio_performance(period=period)

            portfolio_weights[period] = self._portfolio_performance[
                "Current Weight"
            ].unstack()

        for column in historical_columns:
            self._daily_historical_data[column, "Benchmark"] = benchmark_data[column]

        historical = (
            self._daily_historical_data.sort_index(axis=1)
            .reindex(historical_columns, axis=1, level=0)
            .reindex(list(self._tickers) + ["Benchmark"], axis=1, level=1)
        )

        if not self._toolkit_instance:
            self._toolkit_instance = Toolkit(
                tickers=symbols,
                api_key=self._api_key,
                historical=historical,
                start_date=self._start_date,
                quarterly=self._quarterly,
                benchmark_ticker=self._benchmark_ticker,
                rounding=self._rounding,
            )

            self._toolkit_instance._portfolio_weights = portfolio_weights

        return self._toolkit_instance

    def read_portfolio_dataset(
        self,
        adjust_duplicates: bool | None = None,
        date_column: list[str] | None = None,
        date_format_options: list[str] | None = None,
        name_columns: list[str] | None = None,
        ticker_columns: list[str] | None = None,
        price_columns: list[str] | None = None,
        volume_columns: list[str] | None = None,
        currency_columns: list[str] | None = None,
        costs_columns: list[str] | None = None,
        column_mapping: dict[str, str] | None = None,
    ):
        """
        Read and consolidate cash flow data from Excel or CSV files into a single DataFrame.

        This method consolidates portfolio data from one or more Excel or CSV files. It processes the data by
        identifying and handling duplicate entries, adjusting for the required date format, and renaming columns
        based on configuration or user inputs. It ensures consistency across transaction details, including
        descriptions, amounts, costs, and currencies, before returning the data as a structured DataFrame.

        The function also allows for customization of the column names used in the dataset, including columns
        for transaction dates, asset tickers, prices, volumes, and costs/incomes. If necessary, adjustments
        can be made to handle duplicated entries in the dataset based on configuration settings.

        Args:
            adjust_duplicates (bool | None): Flag to indicate whether to adjust duplicate rows in the dataset.
                If None, defaults to the configuration setting.
            date_column (list[str] | None): List of column names for date information.
                Defaults to configuration settings.
            date_format_options (list[str] | None): List of date format strings to attempt when parsing
                date columns (e.g. ['%Y-%m-%d', '%d/%m/%Y']). Defaults to configuration.
            name_columns (list[str] | None): List of column names for transaction descriptions.
                Defaults to configuration.
            ticker_columns (list[str] | None): List of column names for asset tickers.
                Defaults to configuration.
            price_columns (list[str] | None): List of column names for asset prices.
                Defaults to configuration.
            volume_columns (list[str] | None): List of column names for asset volumes.
                Defaults to configuration.
            currency_columns (list[str] | None): List of column names for transaction currencies.
                Defaults to configuration.
            costs_columns (list[str] | None): List of column names for costs or income categories.
                Defaults to configuration.
            column_mapping (dict[str, str] | None): Dictionary mapping dataset columns to the appropriate field names.
                Defaults to configuration.

        Returns:
            pd.DataFrame: A DataFrame containing the consolidated and processed portfolio data.

        Raises:
            FileNotFoundError: If any of the files or directories specified in 'excel_location' cannot be found.
            ValueError: If essential columns (date, description, amount) are missing in the dataset or configuration.
                - Columns can be specified in the configuration or provided explicitly.
                - For missing cost or income columns, an exception is raised if no valid configuration is found.

        Notes:
            - Duplicates are handled according to configuration settings ('self._cfg["general"]["adjust_duplicates"]').
            - If duplicate data is found in the combination of datasets, it will be removed to prevent double-counting.
            - The date columns are converted to datetime objects, and transaction descriptions are treated as
            categorical data.
            - Transaction amount columns are converted to float, with support for different decimal separators.
            - Cost or income columns are processed as categorical data, with optional customization.
            - The dataset is sorted by the date column in ascending order, and the index is set to both
            the date and ticker columns.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.read_portfolio_dataset()
        ```

        Which returns:

        |                                      | Name   |    Price |   Volume |   Costs | Currency   |
        |:-------------------------------------|:-------|---------:|---------:|--------:|:-----------|
        | (Period('2024-05-14', 'D'), 'CAMT')  | CAMT   |  94.4243 |        4 |       0 | USD        |
        | (Period('2024-06-11', 'D'), 'META')  | META   | 505.574  |        8 |       0 | USD        |
        | (Period('2024-06-18', 'D'), 'MPWR')  | MPWR   | 847.6    |       14 |      -1 | USD        |
        | (Period('2024-06-21', 'D'), 'GOOGL') | GOOGL  | 179.908  |       -2 |       0 | USD        |
        | (Period('2024-07-30', 'D'), 'AMD')   | AMD    | 139.57   |        3 |       0 | USD        |
        | (Period('2024-08-29', 'D'), 'AMD')   | AMD    | 146.358  |       -5 |      -2 | USD        |
        | (Period('2024-10-25', 'D'), 'MCHI')  | MCHI   |  48.8436 |        6 |       0 | USD        |
        | (Period('2024-11-05', 'D'), 'EMXC')  | EMXC   |  58.2921 |       -5 |      -1 | USD        |
        | (Period('2024-11-13', 'D'), 'VOO')   | VOO    | 552.136  |       11 |       0 | USD        |
        | (Period('2024-12-05', 'D'), 'OXY')   | OXY    |  48.2517 |       -2 |       0 | USD        |
        """
        date_column = (
            date_column if date_column else self._cfg["general"]["date_columns"]
        )
        date_format_options = (
            date_format_options
            if date_format_options
            else self._cfg["general"]["date_format"]
        )
        name_columns = (
            name_columns if name_columns else self._cfg["general"]["name_columns"]
        )
        ticker_columns = (
            ticker_columns if ticker_columns else self._cfg["general"]["ticker_columns"]
        )

        price_columns = (
            price_columns if price_columns else self._cfg["general"]["price_columns"]
        )

        volume_columns = (
            volume_columns if volume_columns else self._cfg["general"]["volume_columns"]
        )

        currency_columns = (
            currency_columns
            if currency_columns
            else self._cfg["general"]["currency_columns"]
        )

        costs_columns = (
            costs_columns if costs_columns else self._cfg["general"]["costs_columns"]
        )

        column_mapping = (
            column_mapping if column_mapping else self._cfg["general"]["column_mapping"]
        )

        if not self._raw_portfolio_dataset.empty:
            (
                self._portfolio_dataset,
                self._date_column,
                self._name_column,
                self._ticker_column,
                self._price_column,
                self._volume_column,
                self._currency_column,
                self._costs_column,
            ) = portfolio_model.format_portfolio_dataset(
                dataset=self._raw_portfolio_dataset,
                date_columns=date_column,
                date_format_options=date_format_options,
                name_columns=name_columns,
                tickers_columns=ticker_columns,
                price_columns=price_columns,
                volume_columns=volume_columns,
                column_mapping=column_mapping,
                currency_columns=currency_columns,
                costs_columns=costs_columns,
            )
        else:
            if isinstance(self._portfolio_dataset_path, str):
                self._portfolio_dataset_path = [self._portfolio_dataset_path]

            adjust_duplicates = (
                adjust_duplicates
                if adjust_duplicates
                else self._cfg["general"]["adjust_duplicates"]
            )
            (
                self._portfolio_dataset,
                self._date_column,
                self._name_column,
                self._ticker_column,
                self._price_column,
                self._volume_column,
                self._currency_column,
                self._costs_column,
            ) = portfolio_model.read_portfolio_dataset(
                excel_location=self._portfolio_dataset_path,
                adjust_duplicates=adjust_duplicates,
                date_column=date_column,
                date_format_options=date_format_options,
                name_columns=name_columns,
                ticker_columns=ticker_columns,
                price_columns=price_columns,
                volume_columns=volume_columns,
                currency_columns=currency_columns,
                costs_columns=costs_columns,
                column_mapping=column_mapping,
            )

        self._original_tickers = list(
            self._portfolio_dataset[self._ticker_column].unique()
        )

        self._portfolio_dataset = self._portfolio_dataset.sort_values(
            by=self._date_column, ascending=True, kind="stable"
        )
        self._tickers = list(self._portfolio_dataset[self._ticker_column].unique())
        self._start_date = (
            self._portfolio_dataset[self._date_column].min().strftime("%Y-%m-%d")
        )
        self._transactions_currencies = list(
            self._portfolio_dataset[self._currency_column].unique()
        )

        self._portfolio_dataset = self._portfolio_dataset.set_index(
            [self._date_column, self._ticker_column]
        )

        return self._portfolio_dataset

    def collect_benchmark_historical_data(
        self,
        benchmark_ticker: str | None = None,
    ):
        """
        Collect and align historical benchmark data with the portfolio's data.

        The following columns are included:

        - Open: the opening price for the benchmark over time.
        - High: the highest price for the benchmark over time.
        - Low: the lowest price for the benchmark over time.
        - Close: the closing price for the benchmark over time.
        - Adj Close: the adjusted closing price for the benchmark over time.
        - Volume: the volume of the benchmark over time.
        - Dividends: the dividends of the benchmark over time.
        - Returns: the returns of the benchmark over time.
        - Cumulative Return: the cumulative return of the benchmark over time.

        Volatility, Excess Return and Excess Volatility are no longer included here. These are
        available via the Risk module (e.g. toolkit.risk.get_volatility) and the Performance
        module (e.g. toolkit.performance.get_excess_return) instead.

        This method retrieves historical benchmark data (daily, weekly, monthly, quarterly, and yearly)
        for the portfolio, based on a specified benchmark ticker or a mapping of portfolio tickers to
        their corresponding benchmark tickers. The retrieved benchmark data is then aligned with the
        portfolio's historical data, ensuring that the dates of the benchmark data match the dates of
        the portfolio's transactions.

        The method can retrieve data for a single benchmark ticker or for multiple benchmarks depending
        on the portfolio tickers. The resulting benchmark data is returned in a structured DataFrame.

        Args:
            benchmark_ticker (str | None): The default benchmark ticker symbol to use if no per-ticker mapping
                is provided. If None, the default benchmark ticker is retrieved from the configuration.

        Returns:
            pd.DataFrame: A DataFrame containing the benchmark data for the portfolio, indexed by the portfolio's dates.

        Notes:
            - The benchmark data is retrieved in daily, weekly, monthly, quarterly, and yearly periods.
            - If a specific date in the benchmark data does not exist, the method uses the previous available value.
            - The benchmark prices are aligned with the portfolio's data, using a backfill method for missing dates.
            - The method updates several internal attributes:
                - `self._daily_benchmark_data`: Daily benchmark data.
                - `self._weekly_benchmark_data`: Weekly benchmark data.
                - `self._monthly_benchmark_data`: Monthly benchmark data.
                - `self._quarterly_benchmark_data`: Quarterly benchmark data.
                - `self._yearly_benchmark_data`: Yearly benchmark data.
                - `self._benchmark_prices`: Adjusted close prices for the benchmark.
                - `self._benchmark_prices_per_ticker`: Benchmark prices for each ticker.
                - `self._latest_benchmark_price`: The latest available benchmark price.
                - `self._benchmark_specific_prices`: The specific benchmark price for each portfolio transaction.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.collect_benchmark_historical_data()
        ```

        Which returns:

        |            |    Open |    High |     Low |   Close |   Adj Close |      Volume |   Dividends |   Return |   Cumulative Return |
        |:-----------|--------:|--------:|--------:|--------:|------------:|------------:|------------:|---------:|--------------------:|
        | 2026-09-24 | 7666.99 | 7719.01 | 7662.57 | 7704.13 |     7704.13 | 5.31639e+09 |           0 |  -0.0002 |              5.3446 |
        | 2026-09-25 | 7709.86 | 7752.07 | 7693.08 | 7743.41 |     7743.41 | 4.49918e+09 |           0 |   0.0051 |              5.3719 |
        | 2026-09-28 | 7721.7  | 7724.15 | 7666.6  | 7683.69 |     7683.69 | 5.12971e+09 |           0 |  -0.0077 |              5.3305 |
        | 2026-09-29 | 7699.6  | 7699.6  | 7653.55 | 7670.84 |     7670.84 | 5.02359e+09 |           0 |  -0.0017 |              5.3215 |
        | 2026-09-30 | 7688.99 | 7722.88 | 7651.54 | 7651.54 |     7651.54 | 4.27655e+09 |           0 |  -0.0025 |              5.3082 |
        | 2026-10-01 | 7666.47 | 7684.75 | 7616.78 | 7666.45 |     7666.45 | 5.7838e+09  |           0 |   0.0019 |              5.3185 |
        | 2026-10-02 | 7726.24 | 7754.67 | 7700.51 | 7722.72 |     7722.72 | 5.27743e+09 |           0 |   0.0073 |              5.3575 |
        | 2026-10-05 | 7730.86 | 7794.35 | 7727.59 | 7773.95 |     7773.95 | 5.8409e+09  |           0 |   0.0066 |              5.3931 |
        | 2026-10-06 | 7805.96 | 7844.52 | 7805.96 | 7818.93 |     7818.93 | 5.01965e+09 |           0 |   0.0058 |              5.4243 |
        | 2026-10-07 | 7792.98 | 7807.02 | 7763.34 | 7801.77 |     7801.77 | 5.11115e+09 |           0 |  -0.0022 |              5.4124 |
        """
        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        benchmark_ticker = (
            benchmark_ticker
            if benchmark_ticker
            else self._cfg["general"]["benchmark_ticker"]
        )

        if not self._benchmark_toolkit:
            self._benchmark_tickers = {}
            for ticker in self._original_tickers:
                self._benchmark_tickers[ticker] = benchmark_ticker

            self._benchmark_toolkit = Toolkit(
                api_key=self._api_key,
                tickers=list(set(self._benchmark_tickers.values())),
                historical=self._daily_benchmark_data,
                benchmark_ticker=None,
                start_date=self._start_date,
            )

        # Reindex the benchmark data to the dates of the historical dataset so that they are matched up.
        self._daily_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="daily"
        ).reindex(self._daily_historical_data.index, method="backfill")

        self._weekly_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="weekly"
        )
        self._monthly_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="monthly"
        )
        self._quarterly_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="quarterly"
        )
        self._yearly_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="yearly"
        )

        # A date missing for the benchmark falls back to the previous value.
        self._benchmark_prices = self._daily_benchmark_data["Adj Close"].iloc[
            self._daily_benchmark_data["Adj Close"].index.get_indexer(
                self._portfolio_dataset.index.get_level_values(0), method="backfill"
            )
        ]

        # Reindexed onto the portfolio dates so the two line up again.
        self._benchmark_prices.index = self._portfolio_dataset.index

        self._benchmark_prices = self._benchmark_prices.sort_index()

        benchmark_specific_prices = []
        benchmark_latest_price = {}
        benchmark_prices_per_ticker = pd.DataFrame(
            columns=self._tickers, index=self._daily_benchmark_data.index
        )

        for date, ticker in self._portfolio_dataset.index:
            original_ticker = self._original_ticker_combinations[ticker]
            benchmark_ticker = self._benchmark_tickers[original_ticker]

            # Report the benchmark price once even with several orders on the same day.
            benchmark_specific_price = self._benchmark_prices.loc[(date, ticker)]

            if isinstance(benchmark_specific_price, float):
                benchmark_specific_prices.append(pd.Series(benchmark_specific_price))
            else:
                benchmark_specific_prices.append(
                    benchmark_specific_price.drop_duplicates()
                )

            benchmark_latest_price[ticker] = self._daily_benchmark_data[
                "Adj Close"
            ].iloc[-1]
            benchmark_prices_per_ticker[ticker] = self._daily_benchmark_data[
                "Adj Close"
            ]

        self._benchmark_specific_prices = pd.concat(benchmark_specific_prices)
        self._latest_benchmark_price = pd.Series(benchmark_latest_price)
        self._benchmark_prices_per_ticker = benchmark_prices_per_ticker

        return self._daily_benchmark_data

    def collect_historical_data(
        self,
        rounding: int | None = None,
    ):
        """
        Collect and adjust historical price data for the portfolio's tickers.

        The following columns are included:

        - Open: the opening price of each asset over time.
        - High: the highest price of each asset over time.
        - Low: the lowest price of each asset over time.
        - Close: the closing price of each asset over time.
        - Adj Close: the adjusted closing price of each asset over time.
        - Volume: the volume of each asset over time.
        - Dividends: the dividends of each asset over time.
        - Returns: the returns of each asset over time.
        - Cumulative Return: the cumulative return of each asset over time.

        Volatility, Excess Return and Excess Volatility are no longer included here. These are
        available via the Risk module (e.g. toolkit.risk.get_volatility) and the Performance
        module (e.g. toolkit.performance.get_excess_return) instead.

        This method retrieves historical price data (daily, weekly, monthly, quarterly, and yearly)
        for the portfolio's tickers and adjusts for any currency mismatches if necessary. It fetches
        data from a specified data source, applies currency conversion where applicable, and stores the
        adjusted data in separate DataFrames for different time periods (daily, weekly, monthly, quarterly, yearly).

        The method uses the Toolkit class to fetch historical price data and the Currency Toolkit to
        handle currency conversions if the portfolio’s transaction currency does not match the historical
        data's currency.

        Args:
            rounding (int | None): An optional integer specifying the number of decimal places to round the
                historical price data. If None, the default rounding value is used.

        Returns:
            pd.DataFrame: A DataFrame containing the adjusted daily historical price data for the portfolio.

        Notes:
            - This method utilizes the `Toolkit` class to fetch historical price data for different periods.
            - Currency adjustments are made if there's a mismatch between the transaction and historical data currencies.
            - The method handles ISIN-to-ticker mapping when ISIN codes are provided in the portfolio data.
            - The adjusted historical data is returned in separate DataFrames for daily, weekly, monthly, quarterly,
              and yearly price data.
            - If any currency mismatch is found between the portfolio's transaction and historical data,
              a warning message is displayed.
            - The method rounds the data according to the specified or default rounding precision.
            - The latest adjusted price is also captured and available in the `self._latest_price` attribute.
            - If currency conversions are applied, a warning is displayed when mismatches between transaction and
              historical data currencies are found (e.g., for ISIN codes).

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.collect_historical_data()
        ```

        Which returns:


        |            |   ('Open', 'ASML') |   ('Open', 'SKY') |   ('Open', 'AMZN') |   ('Open', 'FIX') |   ('Open', 'MSFT') |   ('Open', 'AMD') |   ('Open', 'AAPL') |
        |:-----------|-------------------:|------------------:|-------------------:|------------------:|-------------------:|------------------:|-------------------:|
        | 2026-09-24 |            1708.85 |             85.38 |             246.02 |           1584    |             495.07 |            600.27 |             336.72 |
        | 2026-09-25 |            1741.25 |             85.91 |             248.56 |           1656.27 |             499.04 |            634.54 |             336.04 |
        | 2026-09-28 |            1748.82 |             86.83 |             246.63 |           1655.82 |             505.47 |            624.9  |             340.37 |
        | 2026-09-29 |            1811.51 |             87.41 |             246.71 |           1681.64 |             508.31 |            616.74 |             336.97 |
        | 2026-09-30 |            1828.6  |             85.85 |             246.79 |           1677.78 |             511.61 |            609.89 |             330.8  |
        | 2026-10-01 |            1799.1  |             84.62 |             251.61 |           1658    |             519.88 |            612.7  |             330    |
        | 2026-10-02 |            1849.74 |             89.57 |             251.51 |           1714.05 |             519.39 |            635.95 |             333.26 |
        | 2026-10-05 |            1852.53 |             87.53 |             250.19 |           1723.93 |             521.62 |            630.61 |             332.82 |
        | 2026-10-06 |            1870    |             85.9  |             253.4  |           1725    |             531.68 |            648.04 |             332.28 |
        | 2026-10-07 |            1788.92 |             84.2  |             254.23 |           1770.01 |             530.73 |            634.77 |             336.96 |
        """
        if not self._toolkit:
            self._toolkit = Toolkit(
                api_key=self._api_key,
                tickers=self._tickers,
                benchmark_ticker=None,
                start_date=self._start_date,
                historical=self._daily_historical_data,
            )

        # Used when ISIN codes are provided and must be matched to tickers.
        self._ticker_combinations = dict(zip(self._toolkit._tickers, self._tickers))
        self._original_ticker_combinations = dict(
            zip(self._tickers, self._original_tickers)
        )

        self._daily_historical_data = self._toolkit.get_historical_data(period="daily")

        if self._daily_historical_data.empty and not self._api_key:
            logger.error(
                "Failed to collect historical data. Please ensure you have provided valid tickers. "
                "Yahoo Finance is unstable and has rate limits which you could have reached.\n"
                "Therefore, consider obtaining an API key with the following link: "
                "https://www.jeroenbouma.com/fmp\nYou can get 15% off by using the "
                "affiliate link which also supports the project."
            )
            return pd.DataFrame()

        self._daily_historical_data = self._daily_historical_data.rename(
            columns=self._ticker_combinations, level=1
        )

        currency_conversions = {}
        if self._currency_column:
            self._historical_statistics = self._toolkit.get_historical_statistics()
            self._historical_statistics = self._historical_statistics.rename(
                columns=self._ticker_combinations, level=0
            )

            if not self._historical_statistics.empty:
                for (_, ticker), currency in self._portfolio_dataset[
                    self._currency_column
                ].items():
                    data_currency = self._historical_statistics.loc["Currency", ticker]

                    # A missing or NaN currency code would send a bogus 'NAN=X' and 404.
                    if (
                        not currency
                        or not data_currency
                        or pd.isna(currency)
                        or pd.isna(data_currency)
                    ):
                        continue

                    if data_currency != currency:
                        currency_conversions[ticker] = (
                            f"{currency}{data_currency}=X".upper()
                        )

        if currency_conversions:
            self._currency_toolkit = Toolkit(
                tickers=list(set(currency_conversions.values())),
                benchmark_ticker=None,
                start_date=self._start_date,
            )

            self._daily_currency_data = self._currency_toolkit.get_historical_data(
                period="daily"
            )

            for ticker, currency in currency_conversions.items():
                for column in self._cfg["adjustments"]["currency_adjustment_columns"]:
                    self._daily_historical_data.loc[
                        :, (column, ticker)
                    ] = self._daily_historical_data.loc[:, (column, ticker)] / (
                        self._daily_currency_data.loc[:, (column, currency)]
                        if self._daily_currency_data.columns.nlevels > 1
                        else self._daily_currency_data[column]
                    )

        self._daily_historical_data = apply_rounding(
            self._daily_historical_data,
            rounding if rounding is not None else self._rounding,
        )

        self._weekly_historical_data = self._toolkit.get_historical_data(
            period="weekly"
        )
        self._weekly_historical_data = self._weekly_historical_data.rename(
            columns=self._ticker_combinations, level=1
        )

        self._monthly_historical_data = self._toolkit.get_historical_data(
            period="monthly"
        )
        self._monthly_historical_data = self._monthly_historical_data.rename(
            columns=self._ticker_combinations, level=1
        )

        self._quarterly_historical_data = self._toolkit.get_historical_data(
            period="quarterly"
        )
        self._quarterly_historical_data = self._quarterly_historical_data.rename(
            columns=self._ticker_combinations, level=1
        )

        self._yearly_historical_data = self._toolkit.get_historical_data(
            period="yearly"
        )
        self._yearly_historical_data = self._yearly_historical_data.rename(
            columns=self._ticker_combinations, level=1
        )

        self._latest_price = self._daily_historical_data["Adj Close"].iloc[-1]

        if currency_conversions:
            logger.warning(
                "Found a mismatch between the currency of the transaction and the currency of the historical data. "
                "This is usually due to working with ISIN codes.\nCorrect this by finding the correct ticker "
                "on for example Yahoo Finance (e.g. S&P 500 ETF can be VUSA.AS). The currencies are "
                "automatically converted but this can lead to some inaccuracies."
            )

        return self._daily_historical_data

    def get_positions_overview(self, rounding: int | None = None):
        """
        Calculate and provide an overview of the portfolio's positions, including key statistics and performance metrics.

        The following columns are included:

        - Volume: the net volume of each asset over time, i.e. every buy minus every sell.
        - Costs: the cumulative transaction costs of each asset over time.
        - Invested Amount: the cumulative capital deployed over time, i.e. the value of every buy plus
        the absolute transaction costs. Sale proceeds are not netted off.
        - Realized Proceeds: the cumulative cash received from selling units of the asset over time.
        - Current Value: the value of the position still held, marked at the dividend-adjusted closing
        price of that day.
        - Cumulative Return: the total return on the capital deployed, i.e.
        (Current Value + Realized Proceeds - Invested Amount) / Invested Amount.
        - Invested Weight: the weight of the asset in the portfolio based on the invested amount over time.
        - Current Weight: the weight of the asset in the portfolio based on the current value over time.

        Positions are marked at the adjusted closing price, so the cumulative return is a total return
        that includes reinvested dividends. The historical "Current Value" is therefore a total-return
        equivalent value rather than the price quoted on that date; the two coincide on the latest date.

        A transaction booked on a day without a price, such as a weekend or an exchange holiday, is
        carried forward to the first following day that does have one.

        This method computes an overview of the portfolio's positions by calculating important statistics and performance
        metrics based on the historical data and transactions. If necessary data has not been collected, it will trigger
        the collection of historical and benchmark data using the `collect_historical_data` and
        `collect_benchmark_historical_data` methods. Additionally, it will compute an overview of
        transactions using the `get_transactions_overview` method.

        The resulting overview includes information about the positions, such as the value, performance, and other
        key metrics. The data is rounded to the specified precision before being returned.

        Args:
            rounding (int | None): An optional integer specifying the number of decimal places to round the data.
                If None, the default rounding precision is used.

        Returns:
            pd.DataFrame: A DataFrame containing an overview of the portfolio's positions, with key statistics and
                performance metrics.

        Raises:
            Exception: If data collection for historical or benchmark data fails, or if the positions overview cannot
                be created. Specific error messages will be raised for each failure.

        Notes:
            - This method ensures that all necessary data is available before calculating the positions overview.
            - The method handles the collection of missing data (historical, benchmark, and transactions) automatically.
            - The positions overview is calculated based on portfolio tickers, transaction data,
            and historical price data.
            - The resulting overview DataFrame is rounded to the specified or default precision.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_positions_overview()
        ```

        Which returns:

        |            |   ('Volume', 'ASML') |   ('Volume', 'SKY') |   ('Volume', 'AMZN') |   ('Volume', 'FIX') |   ('Volume', 'MSFT') |   ('Volume', 'AMD') |
        |:-----------|---------------------:|--------------------:|---------------------:|--------------------:|---------------------:|--------------------:|
        | 2026-09-24 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-09-25 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-09-28 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-09-29 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-09-30 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-10-01 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-10-02 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-10-05 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-10-06 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        | 2026-10-07 |                  129 |                 126 |                  116 |                 122 |                  105 |                  78 |
        """
        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if self._weekly_benchmark_data.empty:
            self.collect_benchmark_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if self._transactions_overview.empty:
            try:
                self.get_transactions_overview()
            except ValueError as error:
                raise ValueError(
                    f"Failed to get transactions overview due to {error}"
                ) from error

        if self._positions_overview.empty:
            try:
                self._positions_overview = overview_model.create_positions_overview(
                    portfolio_tickers=self._tickers,
                    period_dates=to_period_index(
                        self._daily_historical_data.index.get_level_values(0)
                    ),
                    portfolio_dataset=self._transactions_overview,
                    historical_prices=self._daily_historical_data,
                    volume_column=self._volume_column,
                    price_column=self._price_column,
                    costs_column=self._costs_column,
                )
            except ValueError as error:
                raise ValueError(
                    f"Failed to create positions overview due to {error}"
                ) from error

        # Only the returned view is rounded; rounding the cached frame in place would make every later call, including the portfolio performance, read back the precision of whichever call happened to run first.
        return self._positions_overview.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

    def get_portfolio_overview(
        self,
        include_portfolio: bool = True,
        exclude_sold_positions: bool = True,
        rounding: int | None = None,
    ):
        """
        Calculate and provide an overview of the portfolio's key statistics, including performance metrics and
        cost-related information.

        The following columns are included:

        - Identifier: The name of the asset, specifically the ticker (e.g. AAPL)
        - Volume: The net volume of the asset, i.e. every buy minus every sell.
        - Costs: The total costs associated with the asset transactions.
        - Price: The volume-weighted average price paid for the units that were bought. Sells do
        not enter this figure, so it stays a purchase price rather than a net cash figure.
        - Invested: The total capital deployed in the asset, i.e. the value of every buy plus the
        absolute transaction costs. Sale proceeds are deliberately not netted off, because a
        denominator that shrinks with every profitable sale inflates the reported return and flips
        its sign once more cash has come out than went in.
        - Latest Price: The latest available price of the asset obtained from historical data.
        - Latest Value: The market value of the position still held, i.e. Volume times Latest Price.
        - Return: The total return on the capital deployed, i.e. Return Value divided by Invested.
        This covers realized and unrealized results together and is NaN when nothing was invested.
        - Return Value: The absolute profit or loss, i.e. Latest Value plus all sale proceeds minus
        Invested. This equals realized PnL plus unrealized PnL minus the transaction costs, where the
        realized PnL is the figure reported by get_transactions_overview.
        - Benchmark Return: The return the identical cash flows would have produced in the benchmark.
        Every transaction buys or sells benchmark units for the exact cash amount of that transaction
        on that date, so the comparison is matched in money rather than in share count.
        - Volatility: The annualized volatility of the asset over the most recent year, calculated via the
        Risk module (risk_model.get_volatility). For the aggregated "Portfolio" row, this is derived from
        the full covariance matrix of the underlying asset returns (Var_p = w^T * Cov * w, Markowitz, 1952)
        rather than a weighted average of individual volatilities, since the latter ignores diversification
        from imperfectly correlated assets.
        - Benchmark Volatility: The annualized volatility of the asset's benchmark over the most recent year,
        calculated via the Risk module (risk_model.get_volatility).
        - Alpha: The alpha is based on the difference between the asset's return and the benchmark return.
        - Beta: The beta is based on the asset's return and the benchmark return. It measures the asset's volatility
        compared to the benchmark. A beta >1 indicates that the asset is more volatile than the benchmark and a beta <1
        indicates that the asset is less volatile than the benchmark.
        - Weight: The weight of the asset in the portfolio based on the latest market value and the total
        market value of the portfolio.

        No inventory method (FIFO, LIFO or average cost) is applied here: "Return" measures the result of
        every unit of currency put into the position rather than the basis of the units that happen to
        remain. The inventory methods drive the realized PnL in get_transactions_overview instead.

        The "Portfolio" row sums or averages units of different assets for "Volume", "Price" and
        "Latest Price", so those three carry no economic meaning at the portfolio level.

        When recalculating these numbers, it is important to note that results are calculated before the
        rounding parameter is applied which can lead to some discrepancies in the results.

        This method computes a detailed overview of the portfolio, calculating various key statistics such as performance,
        costs, and returns. If necessary data has not been collected, it will automatically trigger data collection using
        the `collect_historical_data` and `collect_benchmark_historical_data` methods. The portfolio overview is
        generated based on the portfolio dataset and benchmark data, and is rounded to the specified precision
        before being returned.

        Args:
            include_portfolio (bool): A boolean flag indicating whether the portfolio itself should be included
                in the overview. Defaults to `True`.
            exclude_sold_positions (bool): A flag indicating whether to exclude sold positions from the overview.
            rounding (int | None): An optional integer specifying the number of decimal places to round the data.
                If None, the default rounding precision is used.

        Returns:
            pd.DataFrame: A DataFrame containing key statistics and an overview of the portfolio, including
                performance metrics, costs, and returns.

        Raises:
            ValueError: If data collection for historical or benchmark data fails.
            ValueError: If the creation of the portfolio overview fails.

        Notes:
            - This method ensures that all necessary data is available before calculating the portfolio overview.
            - The method handles the collection of missing data (historical, benchmark) automatically.
            - The portfolio overview includes important metrics such as returns, costs, and volume, and
            is based on both the portfolio's dataset and benchmark data.
            - The resulting DataFrame is rounded to the specified or default precision.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_portfolio_overview()
        ```

        Which returns:

        | Identifier   |   Volume |   Costs |    Price |   Invested |   Latest Price |     Latest Value |   Return |   Return Value |   Benchmark Return |   Volatility |   Benchmark Volatility |   Alpha |   Beta |   Weight |
        |:-------------|---------:|--------:|---------:|-----------:|---------------:|-----------------:|---------:|---------------:|-------------------:|-------------:|-----------------------:|--------:|-------:|---------:|
        | MPWR         |      116 |     -27 | 246.737  |   30128.9  |       1425.98  | 165414           |   4.5041 |     135705     |             0.7868 |       0.571  |                 0.1304 |  3.7173 | 1.831  |   0.1491 |
        | MSFT         |      105 |     -11 |  39.7562 |    4384.18 |        529.76  |  55624.8         |  11.7117 |      51346.2   |             3.1773 |       0.3593 |                 0.1304 |  8.5344 | 1.1728 |   0.0501 |
        | NFLX         |      114 |     -32 | 131.32   |   16446.9  |         69.7   |   7945.8         |  -0.3883 |      -6386.76  |             1.4756 |       0.3693 |                 0.1304 | -1.864  | 1.0748 |   0.0072 |
        | NVDA         |       69 |     -27 |   2.1316 |     199.66 |        237.47  |  16385.4         |  81.2212 |      16216.6   |             1.0756 |       0.378  |                 0.1304 | 80.1456 | 1.8295 |   0.0148 |
        | OXY          |       27 |     -15 |  37.5907 |    1443.45 |         58.21  |   1571.67        |   0.3352 |        483.812 |             2.5002 |       0.3739 |                 0.1304 | -2.165  | 1.1471 |   0.0014 |
        | SKY          |      126 |     -23 |  18.1967 |    2497.75 |         81.03  |  10209.8         |   3.1659 |       7907.64  |             4.9302 |       0.4426 |                 0.1304 | -1.7643 | 1.4039 |   0.0092 |
        | VOO          |       77 |     -12 | 236.365  |   18684.8  |        714.34  |  55004.2         |   1.9603 |      36627.7   |             1.6199 |       0.13   |                 0.1304 |  0.3403 | 0.9961 |   0.0496 |
        | VSS          |       98 |     -21 |  77.1834 |    8433.99 |        154.53  |  15143.9         |   0.8902 |       7507.8   |             2.2382 |       0.1807 |                 0.1304 | -1.348  | 0.7908 |   0.0136 |
        | WMT          |       92 |     -18 |  17.4419 |    1779.63 |        108.16  |   9950.72        |   4.6578 |       8289.19  |             3.3052 |       0.2712 |                 0.1304 |  1.3526 | 0.4865 |   0.009  |
        | Portfolio    |     2142 |    -532 |  57.823  |  139539    |        518.001 |      1.10956e+06 |   7.0216 |     979783     |             1.7376 |       0.3071 |                 0.1304 |  5.284  | 1.3986 |   1      |
        """
        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if self._weekly_benchmark_data.empty:
            self.collect_benchmark_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if self._portfolio_volatilities.empty:
            asset_volatility = risk_model.get_volatility(
                self._daily_historical_data["Return"], "yearly"
            ).iloc[-1]
            benchmark_volatility = risk_model.get_volatility(
                self._daily_benchmark_data["Return"], "yearly"
            ).iloc[-1]

            self._portfolio_volatilities = pd.concat(
                [
                    asset_volatility,
                    pd.Series([benchmark_volatility], index=["Benchmark"]),
                ]
            )

        if self._portfolio_beta.empty:
            # Calculate daily returns for portfolio tickers and benchmark
            portfolio_returns = (
                self._daily_historical_data["Adj Close"].pct_change().dropna()
            )
            benchmark_returns = (
                self._daily_benchmark_data["Adj Close"].pct_change().dropna()
            )

            # Align dates between portfolio and benchmark returns
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[common_dates]
            benchmark_returns = benchmark_returns.loc[common_dates]

            # Compute beta for each portfolio ticker against the benchmark; the covariance and the variance are taken over the same non-missing sample, since a ticker with a shorter history would otherwise be divided by a variance measured over a longer window than its own covariance.
            betas = {}
            for ticker in portfolio_returns.columns:
                paired_returns = pd.concat(
                    [portfolio_returns[ticker], benchmark_returns], axis=1
                ).dropna()

                if paired_returns.empty:
                    betas[ticker] = float("nan")
                    continue

                cov = paired_returns.iloc[:, 0].cov(paired_returns.iloc[:, 1])
                var = paired_returns.iloc[:, 1].var()
                beta = cov / var if var else float("nan")
                betas[ticker] = beta

            self._portfolio_beta = pd.Series(betas)

        try:
            self._portfolio_overview = overview_model.create_portfolio_overview(
                portfolio_name=self._portfolio_dataset[self._name_column],
                portfolio_volume=self._portfolio_dataset[self._volume_column],
                portfolio_price=self._portfolio_dataset[self._price_column],
                portfolio_costs=self._portfolio_dataset[self._costs_column],
                latest_returns=self._latest_price,
                benchmark_prices=self._benchmark_specific_prices,
                benchmark_latest_prices=self._latest_benchmark_price,
                volatilities=self._portfolio_volatilities,
                betas=self._portfolio_beta,
                include_portfolio=include_portfolio,
                asset_returns=self._daily_historical_data["Return"],
            )
        except ValueError as error:
            raise ValueError(f"Failed to create portfolio overview: {error}") from error

        if exclude_sold_positions:
            self._portfolio_overview = self._portfolio_overview[
                self._portfolio_overview["Volume"] > 0
            ]

        self._portfolio_overview = self._portfolio_overview.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        return self._portfolio_overview

    def get_portfolio_performance(
        self,
        period: str | None = None,
        exclude_sold_positions: bool = True,
        rounding: int | None = None,
    ):
        """
        Calculate portfolio performance metrics for a specified period.

        This method calculates key performance metrics, such as returns, for the portfolio
        over a specified period. The available periods are 'yearly', 'quarterly', 'monthly',
        'weekly', and 'daily'. It uses the positions overview dataset for these calculations.
        If the necessary data has not been collected, it triggers the collection of historical
        and benchmark data.

        Args:
            period (str | None): The time period for which portfolio performance metrics should be calculated.
                It can be one of the following: 'yearly', 'quarterly', 'monthly', 'weekly', or 'daily'.
                If None, the default period is 'quarterly' (if the 'quarterly' attribute is set to True),
                otherwise, it defaults to 'yearly'.
            exclude_sold_positions (bool): A flag indicating whether to exclude sold positions.
            rounding (int | None): The number of decimal places to round the output to.
                If None, it defaults to the rounding precision specified in the configuration.

        Returns:
            pd.DataFrame: A DataFrame containing the portfolio performance metrics for the specified period.

        Raises:
            ValueError: If an invalid or unsupported period is provided.
            ValueError: If there is an issue with collecting historical data or creating the portfolio performance.

        Notes:
            - This method ensures that the required historical and benchmark data is available before calculating
            performance.
            - The method uses the `overview_model.create_portfolio_performance` function to compute performance metrics.
            - The resulting DataFrame will be rounded to the specified number of decimal places
            (or the default configuration).

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_portfolio_performance(period='weekly')
        ```

        Which returns:

        |                                                    |   Volume |   Costs |   Invested Amount |   Realized Proceeds |   Current Value |   Invested Weight |   Current Weight |   Return |
        |:---------------------------------------------------|---------:|--------:|------------------:|--------------------:|----------------:|------------------:|-----------------:|---------:|
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'META') |       15 |      -1 |           4796.75 |              0      |        10819.6  |            0.0344 |           0.0098 |   1.2556 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'MPWR') |      116 |     -27 |          30128.9  |            419.755  |       165414    |            0.2159 |           0.1491 |   4.5041 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'MSFT') |      105 |     -11 |           4384.18 |            105.59   |        55624.8  |            0.0314 |           0.0501 |  11.7117 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'NFLX') |      114 |     -32 |          16446.9  |           2114.39   |         7945.8  |            0.1179 |           0.0072 |  -0.3883 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'NVDA') |       69 |     -27 |            199.66 |             30.8599 |        16385.4  |            0.0014 |           0.0148 |  81.2212 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'OXY')  |       27 |     -15 |           1443.45 |            355.587  |         1571.67 |            0.0103 |           0.0014 |   0.3352 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'SKY')  |      126 |     -23 |           2497.75 |            195.613  |        10209.8  |            0.0179 |           0.0092 |   3.1659 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'VOO')  |       77 |     -12 |          18684.8  |            308.375  |        55004.2  |            0.1339 |           0.0496 |   1.9603 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'VSS')  |       98 |     -21 |           8433.99 |            797.842  |        15143.9  |            0.0604 |           0.0136 |   0.8902 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'WMT')  |       92 |     -18 |           1779.63 |            118.103  |         9950.72 |            0.0128 |           0.009  |   4.6578 |
        """
        if not period:
            period = "quarterly" if self._quarterly else "yearly"

        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if self._weekly_benchmark_data.empty:
            self.collect_benchmark_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if self._positions_overview.empty:
            try:
                self.get_positions_overview()
            except ValueError as error:
                raise ValueError(
                    f"Failed to get positions overview: {error}"
                ) from error

        if not period:
            raise ValueError(
                "Please provide a period. This can be 'yearly', 'quarterly', 'monthly', 'weekly', or 'daily'"
            )

        period_string = period.lower()

        if period_string == "yearly":
            period_symbol = "Y"
        elif period_string == "quarterly":
            period_symbol = "Q"
        elif period_string == "monthly":
            period_symbol = "M"
        elif period_string == "weekly":
            period_symbol = "W"
        elif period_string == "daily":
            period_symbol = "D"
        else:
            raise ValueError(
                "Please provide a valid period. This can be 'yearly', 'quarterly', 'monthly', 'weekly', or 'daily'"
            )

        try:
            self._portfolio_performance = overview_model.create_portfolio_performance(
                positions_dataset=self._positions_overview,
                date_column=self._date_column,
                ticker_column=self._ticker_column,
                period_string=period_symbol,
            )
        except ValueError as error:
            raise ValueError(
                f"Failed to create portfolio performance: {error}"
            ) from error

        if exclude_sold_positions:
            self._portfolio_performance = self._portfolio_performance[
                self._portfolio_performance["Volume"] > 0
            ]

        self._portfolio_performance = self._portfolio_performance.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        return self._portfolio_performance

    def get_transactions_overview(
        self,
        rounding: int | None = None,
        exclude_sold_positions: bool = True,
        pnl_method: str = "FIFO",
    ):
        """
        Calculate and collect transaction overview ratios based on the provided data.

        This method calculates various transaction overview ratios, such as returns, costs,
        and profit & loss (PnL), based on the transaction dataset. The calculated ratios
        are added as new columns to the portfolio dataset. It also provides the option
        to use different methods for calculating PnL (FIFO, LIFO, or AVERAGE). The method
        ensures that necessary historical and benchmark data is available before performing
        calculations.

        Args:
            rounding (int | None): The number of decimal places to round the output to.
                If None, it defaults to the rounding specified in the configuration.
            exclude_sold_positions (bool): A flag indicating whether to exclude sold positions
            pnl_method (str): The method for calculating profit & loss. Options are:
                'FIFO' (First In, First Out), 'LIFO' (Last In, First Out), or 'AVERAGE'.
                Defaults to 'FIFO'.

        Returns:
            pd.DataFrame: The portfolio dataset with added transaction overview ratios and PnL columns.

        Raises:
            ValueError: If there is an issue with collecting historical data, creating the transaction overview,
                        or if an invalid PnL method is provided.

        Notes:
            - The method first checks and collects necessary historical data if not already available.
            - It uses the `overview_model.create_transactions_overview` and
              `overview_model.create_profit_and_loss_overview` functions to calculate the transaction ratios
              and profit & loss.
            - The transaction ratios and PnL are added to the original portfolio dataset as new columns.
            - "PnL" is the realized result of a sell transaction under the chosen inventory method and
              is zero on every buy. "Cumulative PnL" is the running realized total per ticker and
              restarts at zero for the next ticker rather than carrying across the portfolio.
            - Transaction costs are deliberately excluded from the PnL. They are carried by the
              invested amount instead, so a cost is never counted in two places. The identity
              Return Value = realized PnL + unrealized PnL - transaction costs therefore ties this
              output back to get_portfolio_overview exactly.
            - "Invested Amount" is per transaction and is negative on a sell, since it is the cash
              movement of that one transaction rather than the capital deployed in the position.
            - If no rounding is provided, the rounding precision specified in the configuration is used.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_transactions_overview()
        ```

        Which returns:

        |                                      | Name   |    Price |   Volume |   Costs | Currency   |   Invested Amount |   Current Value |   % Return |    Return |   PnL |   Cumulative PnL |
        |:-------------------------------------|:-------|---------:|---------:|--------:|:-----------|------------------:|----------------:|-----------:|----------:|------:|-----------------:|
        | (Period('2024-02-01', 'D'), 'GOOGL') | GOOGL  | 140.041  |       14 |       0 | USD        |          1960.57  |         4907    |     1.5028 | 2946.43   |     0 |           0      |
        | (Period('2024-02-12', 'D'), 'BAC')   | BAC    |  33.0126 |        5 |      -3 | USD        |           168.063 |          267.6  |     0.5923 |   99.5368 |     0 |          56.8807 |
        | (Period('2024-02-22', 'D'), 'MCHI')  | MCHI   |  38.4963 |        7 |      -1 | USD        |           270.474 |          361.48 |     0.3365 |   91.0058 |     0 |          -4.77   |
        | (Period('2024-03-12', 'D'), 'MPWR')  | MPWR   | 727.32   |       11 |       0 | USD        |          8000.52  |        15685.8  |     0.9606 | 7685.26   |     0 |         376.706  |
        | (Period('2024-05-14', 'D'), 'CAMT')  | CAMT   |  94.4243 |        4 |       0 | USD        |           377.697 |          608.8  |     0.6119 |  231.103  |     0 |          99.4194 |
        | (Period('2024-06-11', 'D'), 'META')  | META   | 505.574  |        8 |       0 | USD        |          4044.59  |         5770.48 |     0.4267 | 1725.89   |     0 |           0      |
        | (Period('2024-06-18', 'D'), 'MPWR')  | MPWR   | 847.6    |       14 |      -1 | USD        |         11867.4   |        19963.7  |     0.6822 | 8096.32   |     0 |         376.706  |
        | (Period('2024-07-30', 'D'), 'AMD')   | AMD    | 139.57   |        3 |       0 | USD        |           418.711 |         1937.58 |     3.6275 | 1518.87   |     0 |           9.795  |
        | (Period('2024-10-25', 'D'), 'MCHI')  | MCHI   |  48.8436 |        6 |       0 | USD        |           293.062 |          309.84 |     0.0573 |   16.7783 |     0 |          -4.77   |
        | (Period('2024-11-13', 'D'), 'VOO')   | VOO    | 552.136  |       11 |       0 | USD        |          6073.5   |         7857.74 |     0.2938 | 1784.24   |     0 |         131.082  |
        """
        pnl_method = pnl_method.upper()

        if pnl_method not in ["FIFO", "LIFO", "AVERAGE"]:
            raise ValueError(
                "Please provide a valid method. This can be 'FIFO', 'LIFO', or 'AVERAGE'"
            )

        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if self._weekly_benchmark_data.empty:
            self.collect_benchmark_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        try:
            new_columns = overview_model.create_transactions_overview(
                portfolio_volume=self._portfolio_dataset[self._volume_column],
                portfolio_price=self._portfolio_dataset[self._price_column],
                portfolio_costs=self._portfolio_dataset[self._costs_column],
                latest_returns=self._latest_price.loc[self._tickers],
            )
        except ValueError as error:
            raise ValueError(
                f"Failed to create transaction overview: {error}"
            ) from error

        try:
            self._transactions_overview = pd.concat(
                [self._portfolio_dataset, new_columns], axis=1
            )
        except ValueError as error:
            raise ValueError(
                f"Failed to add transaction overview to portfolio dataset: {error}"
            ) from error

        try:
            original_index = self._transactions_overview.index

            pnl_columns = overview_model.create_profit_and_loss_overview(
                transactions_overview=self._transactions_overview,
                ticker_column=self._ticker_column,
                volume_column=self._volume_column,
                price_column=self._price_column,
                method=pnl_method,
            )

            # Ensure the indices are unique before concatenation
            self._transactions_overview = self._transactions_overview.reset_index(
                drop=True
            )
            pnl_columns = pnl_columns.reset_index(drop=True)

            self._transactions_overview = pd.concat(
                [self._transactions_overview, pnl_columns], axis=1
            )

            self._transactions_overview.index = original_index
        except (ValueError, IndexError, KeyError) as error:
            logger.error("Failed to create PnL overview: %s", error)

        # The rounding and the filter are applied to the returned view only; the stored overview feeds create_positions_overview, so rounding it in place would push the rounding of whichever call ran first into every position metric, and removing the sell transactions would leave that function cumulating buys alone, keeping sold shares in the position forever.
        transactions_overview = self._transactions_overview.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if exclude_sold_positions:
            return transactions_overview[transactions_overview["Volume"] > 0]

        return transactions_overview

    def get_transactions_performance(
        self,
        period: str | None = None,
        exclude_sold_positions: bool = True,
        rounding: int | None = None,
    ):
        """
        Calculate transaction performance metrics for a specified period.

        This method calculates various transaction performance metrics, such as returns,
        costs, and benchmarks, for the specified period. The calculation is based on
        historical price data for the corresponding period, including both the portfolio
        and benchmark datasets. It provides an overview of how the portfolio's transactions
        have performed in comparison to the benchmark over the given period.

        Args:
            period (str | None): The time period for which transaction performance metrics
                should be calculated. This can be one of the following: 'yearly', 'quarterly',
                'monthly', 'weekly', or 'daily'. If None, the default is 'quarterly' if
                the 'quarterly' attribute is set to True, otherwise 'yearly'.
            exclude_sold_positions (bool): A flag indicating whether to exclude sold positions
            rounding (int | None): The number of decimal places to round the output to.
                If None, it defaults to the rounding specified in the configuration.

        Returns:
            pd.DataFrame: A DataFrame containing transaction performance metrics, including
                returns, costs, and benchmarks, for the specified period.

        Raises:
            ValueError: If an invalid or unsupported period is provided or if there is an issue
                        with creating the transaction performance metrics.

        Notes:
            - The method supports multiple time periods ('yearly', 'quarterly', 'monthly',
              'weekly', 'daily') for calculating transaction performance metrics.
            - If no period is provided, it defaults to 'quarterly' or 'yearly' based on the
              configuration.
            - The method uses historical price data from the specified period and benchmarks
              to calculate the metrics.
            - If no rounding is specified, the rounding precision from the configuration is used.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_transactions_performance(period='quarterly')
        ```

        Which returns:

        |                                      |   Volume |    Price |   Costs |   Invested Amount |   Realized Proceeds |   Current Value |   Return |   Benchmark Return |   Alpha |
        |:-------------------------------------|---------:|---------:|--------:|------------------:|--------------------:|----------------:|---------:|-------------------:|--------:|
        | (Period('2023Q4', 'Q-DEC'), 'MCHI')  |       15 |  41.0726 |      -3 |           619.088 |                   0 |         580.2   |  -0.0628 |             0.0853 | -0.1481 |
        | (Period('2024Q1', 'Q-DEC'), 'BAC')   |        5 |  33.0126 |      -3 |           168.063 |                   0 |         179.111 |   0.0657 |             0.0276 |  0.0381 |
        | (Period('2024Q1', 'Q-DEC'), 'GOOGL') |       14 | 140.041  |       0 |          1960.57  |                   0 |        2093.1   |   0.0676 |             0.071  | -0.0034 |
        | (Period('2024Q1', 'Q-DEC'), 'MCHI')  |        7 |  38.4963 |      -1 |           270.474 |                   0 |         264.04  |  -0.0238 |             0.0291 | -0.0529 |
        | (Period('2024Q1', 'Q-DEC'), 'MPWR')  |       11 | 727.32   |       0 |          8000.52  |                   0 |        7316.99  |  -0.0854 |             0.0153 | -0.1007 |
        | (Period('2024Q2', 'Q-DEC'), 'CAMT')  |        4 |  94.4243 |       0 |           377.697 |                   0 |         500.96  |   0.3264 |             0.0407 |  0.2856 |
        | (Period('2024Q2', 'Q-DEC'), 'META')  |        8 | 505.574  |       0 |          4044.59  |                   0 |        4003.68  |  -0.0101 |             0.0158 | -0.026  |
        | (Period('2024Q2', 'Q-DEC'), 'MPWR')  |       14 | 847.6    |      -1 |         11867.4   |                   0 |       11313.1   |  -0.0467 |            -0.0049 | -0.0418 |
        | (Period('2024Q4', 'Q-DEC'), 'MCHI')  |        6 |  48.8436 |       0 |           293.062 |                   0 |         273.24  |  -0.0676 |             0.0127 | -0.0803 |
        | (Period('2024Q4', 'Q-DEC'), 'VOO')   |       11 | 552.136  |       0 |          6073.5   |                   0 |        5804.48  |  -0.0443 |            -0.0173 | -0.027  |
        """
        if not period:
            period = "quarterly" if self._quarterly else "yearly"

        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if self._weekly_benchmark_data.empty:
            self.collect_benchmark_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        if not period:
            raise ValueError(
                "Please provide a period. This can be 'yearly', 'quarterly', 'monthly', 'weekly', or 'daily'"
            )

        period_string = period.lower()

        if period_string == "yearly":
            historical_dataset = self._yearly_historical_data["Adj Close"]
            benchmark_dataset = self._yearly_benchmark_data["Adj Close"]
            period_symbol = "Y"
        elif period_string == "quarterly":
            historical_dataset = self._quarterly_historical_data["Adj Close"]
            benchmark_dataset = self._quarterly_benchmark_data["Adj Close"]
            period_symbol = "Q"
        elif period_string == "monthly":
            historical_dataset = self._monthly_historical_data["Adj Close"]
            benchmark_dataset = self._monthly_benchmark_data["Adj Close"]
            period_symbol = "M"
        elif period_string == "weekly":
            historical_dataset = self._weekly_historical_data["Adj Close"]
            benchmark_dataset = self._weekly_benchmark_data["Adj Close"]
            period_symbol = "W"
        elif period_string == "daily":
            historical_dataset = self._daily_historical_data["Adj Close"]
            benchmark_dataset = self._daily_benchmark_data["Adj Close"]
            period_symbol = "D"
        else:
            raise ValueError(
                "Please provide a valid period. This can be "
                "'yearly', 'quarterly', 'monthly', 'weekly', "
                "or 'daily'"
            )

        try:
            self._transactions_performance = (
                overview_model.create_transactions_performance(
                    portfolio_dataset=self._portfolio_dataset,
                    ticker_column=self._ticker_column,
                    date_column=self._date_column,
                    volume_column=self._volume_column,
                    price_column=self._price_column,
                    costs_column=self._costs_column,
                    period_prices=historical_dataset,
                    period_string=period_symbol,
                    benchmark_specific_prices=self._benchmark_specific_prices,
                    benchmark_period_prices=benchmark_dataset,
                )
            )
        except ValueError as error:
            raise ValueError(
                f"Failed to create transaction performance metrics: {error}"
            ) from error

        if exclude_sold_positions:
            self._transactions_performance = self._transactions_performance[
                self._transactions_performance["Volume"] > 0
            ]

        self._transactions_performance = self._transactions_performance.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        return self._transactions_performance
