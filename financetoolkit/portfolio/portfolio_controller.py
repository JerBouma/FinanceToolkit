"""Portfolio Module"""

import os
import shutil
from importlib import resources

import pandas as pd

from financetoolkit.portfolio import helpers, overview_model, portfolio_model
from financetoolkit.toolkit_controller import Toolkit
from financetoolkit.utilities import logger_model

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

        if not portfolio_dataset and not example:
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
        self._api_key: str = api_key
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
    def toolkit(self) -> Toolkit:
        """
        Converts the Portfolio to a Finance Toolkit object.

        This method converts the Portfolio object to a Finance Toolkit object, enabling the
        use of the Toolkit's 150+ financial metrics and indicators for the portfolio's assets.

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

        |           |    2020 |   2021 |    2022 |   2023 |     2024 |
        |:----------|--------:|-------:|--------:|-------:|---------:|
        | AAPL      |  0.2091 | 0.2588 |  0.2531 | 0.2531 |   0.2397 |
        | ALGN      |  0.7184 | 0.1953 |  0.0968 | 0.1152 |   0.1054 |
        | AMD       |  0.255  | 0.1924 |  0.0559 | 0.0377 |   0.0636 |
        | AMZN      |  0.0553 | 0.071  | -0.0053 | 0.0529 |   0.0929 |
        | ASML      |  0.2542 | 0.3161 |  0.2656 | 0.2844 |   0.2679 |
        | AVGO      |  0.1239 | 0.2454 |  0.3462 | 0.3931 |   0.1143 |
        | BAC       |  0.2092 | 0.3588 |  0.2899 | 0.269  |   0.2663 |
        | BLDR      |  0.0366 | 0.0867 |  0.121  | 0.0901 |   0.0657 |
        | CAMT      |  0.1397 | 0.2235 |  0.2491 | 0.2493 | nan      |
        | CWST      |  0.1176 | 0.0462 |  0.0489 | 0.0201 |   0.0087 |
        | FICO      |  0.1826 | 0.2978 |  0.2712 | 0.2837 |   0.2986 |
        | FIX       |  0.0526 | 0.0466 |  0.0594 | 0.0621 |   0.0743 |
        | GOOGL     |  0.2206 | 0.2951 |  0.212  | 0.2401 |   0.286  |
        | KHC       |  0.0136 | 0.0389 |  0.0892 | 0.1072 |   0.1062 |
        | META      |  0.339  | 0.3338 |  0.199  | 0.2898 |   0.3791 |
        | MPWR      |  0.1947 | 0.2004 |  0.2439 | 0.2347 | nan      |
        | MSFT      |  0.3096 | 0.3645 |  0.3669 | 0.3415 |   0.3596 |
        | NFLX      |  0.1105 | 0.1723 |  0.1421 | 0.1604 |   0.2234 |
        | NVDA      |  0.2561 | 0.2598 |  0.3623 | 0.1619 |   0.4885 |
        | OXY       | -0.7599 | 0.0895 |  0.3632 | 0.1662 |   0.0889 |
        | SKY       |  0.0598 | 0.1124 |  0.1542 | 0.0724 | nan      |
        | WMT       |  0.0284 | 0.0242 |  0.0239 | 0.0191 |   0.0239 |
        | Portfolio |  0.2373 | 0.2183 |  0.2001 | 0.2098 |   0.202  |
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

        symbols = list(self._tickers) + ["Portfolio"]  # type: ignore

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
            .reindex(list(self._tickers) + ["Benchmark"], axis=1, level=1)  # type: ignore
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
            date_format (str | None): The format for date columns, specified as a string.
                Defaults to configuration.
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

        | Date       | Name   |    Price |   Volume |   Costs | Currency   |
        |:-----------|:-------|---------:|---------:|--------:|:-----------|
        | 2024-05-14 | CAMT   |  94.4243 |        4 |       0 | USD        |
        | 2024-06-11 | META   | 505.574  |        8 |       0 | USD        |
        | 2024-06-18 | MPWR   | 847.6    |       14 |      -1 | USD        |
        | 2024-06-21 | GOOGL  | 179.908  |       -2 |       0 | USD        |
        | 2024-07-30 | AMD    | 139.57   |        3 |       0 | USD        |
        | 2024-08-29 | AMD    | 146.358  |       -5 |      -2 | USD        |
        | 2024-10-25 | MCHI   |  48.8436 |        6 |       0 | USD        |
        | 2024-11-05 | EMXC   |  58.2921 |       -5 |      -1 | USD        |
        | 2024-11-13 | VOO    | 552.136  |       11 |       0 | USD        |
        | 2024-12-05 | OXY    |  48.2517 |       -2 |       0 | USD        |
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
            ) = portfolio_model.read_portfolio_dataset(  # type: ignore
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
            by=self._date_column, ascending=True
        )
        self._tickers = list(self._portfolio_dataset[self._ticker_column].unique())
        self._start_date = (
            self._portfolio_dataset[self._date_column].min().strftime("%Y-%m-%d")
        )
        self._transactions_currencies = list(
            self._portfolio_dataset[self._currency_column].unique()  # type: ignore
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
        - Volatility: the volatility of the benchmark over the whole period.
        - Excess Return: the excess return (return minus risk free rate) of the benchmark over time.
        - Excess Volatility: the excess volatility (return minus risk free rate) of the benchmark over time.
        - Cumulative Return: the cumulative return of the benchmark over time.

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

        | Date       |    Open |    High |     Low |   Close |   Adj Close |      Volume |   Dividends |   Return |   Volatility |   Excess Return |   Excess Volatility |   Cumulative Return |
        |:-----------|--------:|--------:|--------:|--------:|------------:|------------:|------------:|---------:|-------------:|----------------:|--------------------:|--------------------:|
        | 2025-02-14 | 6115.52 | 6127.47 | 6107.62 | 6114.63 |     6114.63 | 4.33519e+09 |           0 |  -0.0001 |       0.0122 |         -0.0448 |               0.018 |              4.2419 |
        | 2025-02-18 | 6121.6  | 6129.63 | 6099.51 | 6129.58 |     6129.58 | 4.68498e+09 |           0 |   0.0024 |       0.0122 |         -0.043  |               0.018 |              4.2523 |
        | 2025-02-19 | 6117.76 | 6147.43 | 6111.15 | 6144.15 |     6144.15 | 4.56233e+09 |           0 |   0.0024 |       0.0122 |         -0.0429 |               0.018 |              4.2624 |
        | 2025-02-20 | 6134.5  | 6134.5  | 6084.59 | 6117.52 |     6117.52 | 4.81369e+09 |           0 |  -0.0043 |       0.0122 |         -0.0493 |               0.018 |              4.2439 |
        | 2025-02-21 | 6114.1  | 6114.82 | 6008.56 | 6013.13 |     6013.13 | 5.43456e+09 |           0 |  -0.0171 |       0.0122 |         -0.0613 |               0.018 |              4.1715 |
        | 2025-02-24 | 6026.69 | 6043.65 | 5977.83 | 5983.25 |     5983.25 | 4.99012e+09 |           0 |  -0.005  |       0.0122 |         -0.0489 |               0.018 |              4.1508 |
        | 2025-02-25 | 5982.73 | 5992.65 | 5908.49 | 5955.25 |     5955.25 | 5.37469e+09 |           0 |  -0.0047 |       0.0122 |         -0.0477 |               0.018 |              4.1314 |
        | 2025-02-26 | 5970.87 | 6009.82 | 5932.69 | 5956.06 |     5956.06 | 4.86958e+09 |           0 |   0.0001 |       0.0122 |         -0.0424 |               0.018 |              4.1319 |
        | 2025-02-27 | 5981.88 | 5993.69 | 5858.78 | 5861.57 |     5861.57 | 5.05768e+09 |           0 |  -0.0159 |       0.0122 |         -0.0587 |               0.018 |              4.0664 |
        | 2025-02-28 | 5856.74 | 5959.4  | 5837.66 | 5954.5  |     5954.5  | 6.44114e+09 |           0 |   0.0159 |       0.0122 |         -0.0264 |               0.018 |              4.1309 |
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
            period="daily", progress_bar=False
        ).reindex(self._daily_historical_data.index, method="backfill")

        self._weekly_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="weekly", progress_bar=False
        )
        self._monthly_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="monthly", progress_bar=False
        )
        self._quarterly_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="quarterly", progress_bar=False
        )
        self._yearly_benchmark_data = self._benchmark_toolkit.get_historical_data(
            period="yearly", progress_bar=False
        )

        # It could be that a specific date does not exist for the given benchmark. In that case,
        # the previous value is used instead.
        self._benchmark_prices = self._daily_benchmark_data["Adj Close"].iloc[
            self._daily_benchmark_data["Adj Close"].index.get_indexer(
                self._portfolio_dataset.index.get_level_values(0), method="backfill"
            )
        ]

        # The index of the benchmark prices is set to the dates of the portfolio dataset
        # so that they are matched up again.
        self._benchmark_prices.index = self._portfolio_dataset.index

        self._benchmark_prices = self._benchmark_prices.sort_index()

        benchmark_specific_prices = []
        benchmark_latest_price = {}
        benchmark_prices_per_ticker = pd.DataFrame(
            columns=self._tickers, index=self._daily_benchmark_data.index
        )

        for (date, ticker), _ in self._portfolio_dataset.iterrows():
            original_ticker = self._original_ticker_combinations[ticker]
            benchmark_ticker = self._benchmark_tickers[original_ticker]

            # Add the specific benchmark price and, if multiple orders of the same ticker are made on the same day
            # (e.g. buying and selling), only report the benchmark price once.
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
        progress_bar: bool = True,
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
        - Volatility: the volatility of each asset over the whole period.
        - Excess Return: the excess return (return minus risk free rate) of each asset over time.
        - Excess Volatility: the excess volatility (return minus risk free rate) of each asset over time.
        - Cumulative Return: the cumulative return of each asset over time.

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
            progress_bar (bool): A boolean indicating whether to show a progress bar during data retrieval.

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


        | Date       |   Open |   High |   Low |   Close |   Adj Close |      Volume |   Dividends |   Return |   Volatility |   Excess Return |   Excess Volatility |   Cumulative Return |
        |:-----------|-------:|-------:|------:|--------:|------------:|------------:|------------:|---------:|-------------:|----------------:|--------------------:|--------------------:|
        | 2025-02-14 |  48.23 |  49.09 | 48.01 |   48.06 |       48.06 | 1.04361e+07 |           0 |   0.0038 |       0.0253 |         -0.0409 |              0.0284 |              9.2171 |
        | 2025-02-18 |  48.86 |  49.14 | 47.91 |   48.84 |       48.84 | 1.53775e+07 |           0 |   0.0162 |       0.0253 |         -0.0292 |              0.0284 |              9.3667 |
        | 2025-02-19 |  49.5  |  52.17 | 49.3  |   50.99 |       50.99 | 2.72406e+07 |           0 |   0.044  |       0.0253 |         -0.0013 |              0.0284 |              9.7791 |
        | 2025-02-20 |  51.17 |  52.58 | 50.49 |   52.09 |       52.09 | 1.41794e+07 |           0 |   0.0216 |       0.0253 |         -0.0234 |              0.0284 |              9.99   |
        | 2025-02-21 |  51.8  |  51.99 | 50.39 |   50.42 |       50.42 | 1.38929e+07 |           0 |  -0.0321 |       0.0253 |         -0.0763 |              0.0284 |              9.6698 |
        | 2025-02-24 |  50.07 |  50.4  | 49.5  |   49.86 |       49.86 | 1.11417e+07 |           0 |  -0.0111 |       0.0253 |         -0.055  |              0.0284 |              9.5624 |
        | 2025-02-25 |  49.81 |  49.96 | 48.57 |   48.89 |       48.89 | 1.11453e+07 |           0 |  -0.0195 |       0.0253 |         -0.0625 |              0.0284 |              9.3763 |
        | 2025-02-26 |  49.02 |  49.2  | 48.26 |   48.55 |       48.55 | 7.2188e+06  |           0 |  -0.007  |       0.0253 |         -0.0495 |              0.0284 |              9.3111 |
        | 2025-02-27 |  48.9  |  49.43 | 48.34 |   48.65 |       48.65 | 1.04084e+07 |           0 |   0.0021 |       0.0253 |         -0.0407 |              0.0284 |              9.3303 |
        | 2025-02-28 |  48.48 |  48.93 | 47.75 |   48.84 |       48.84 | 1.26817e+07 |           0 |   0.0039 |       0.0253 |         -0.0384 |              0.0284 |              9.3667 |
        """
        if not self._toolkit:
            self._toolkit = Toolkit(
                api_key=self._api_key,
                tickers=self._tickers,
                benchmark_ticker=None,
                start_date=self._start_date,
                historical=self._daily_historical_data,
            )

        # This is used in case ISIN codes are provided and therefore ISIN codes need to
        # be matched to the corresponding tickers
        self._ticker_combinations = dict(zip(self._toolkit._tickers, self._tickers))  # type: ignore
        self._original_ticker_combinations = dict(
            zip(self._tickers, self._original_tickers)  # type: ignore
        )

        self._daily_historical_data = self._toolkit.get_historical_data(
            period="daily", progress_bar=progress_bar
        )

        if self._daily_historical_data.empty and not self._api_key:
            logger.error(
                "Failed to collect historical data. Please ensure you have provided valid tickers. "
                "Yahoo Finance is unstable and has rate limits which you could have reached.\n"
                "Therefore, consider obtaining an API key with the following link: "
                "https://www.jeroenbouma.com/fmp. You can get 15% off by using the "
                "affiliate link which also supports the project."
            )
            return pd.DataFrame()

        self._daily_historical_data = self._daily_historical_data.rename(
            columns=self._ticker_combinations, level=1
        )

        currency_conversions = {}
        if self._currency_column:  # type: ignore
            self._historical_statistics = self._toolkit.get_historical_statistics(
                progress_bar=False
            )
            self._historical_statistics = self._historical_statistics.rename(
                columns=self._ticker_combinations, level=0
            )

            if not self._historical_statistics.empty:
                for (_, ticker), currency in self._portfolio_dataset[
                    self._currency_column  # type: ignore
                ].items():
                    data_currency = self._historical_statistics.loc["Currency", ticker]
                    if self._historical_statistics.loc["Currency", ticker] != currency:
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
                period="daily", progress_bar=False
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

        self._daily_historical_data = round(
            self._daily_historical_data, rounding if rounding else self._rounding
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

        - Volume: the volume of each asset over time.
        - Costs: the costs of each asset over time.
        - Invested Amount: the invested amount over time.
        - Current Value: the current value of the asset based on the market value on that specific day over time.
        - Cumulative Return: the cumulative return of the asset over time.
        - Invested Weight: the weight of the asset in the portfolio based on the invested amount over time.
        - Current Weight: the weight of the asset in the portfolio based on the current value over time.

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

        | Date       |   Volume |   Costs |   Invested Amount |   Current Value |   Cumulative Return |   Invested Weight |   Current Weight |
        |:-----------|---------:|--------:|------------------:|----------------:|--------------------:|------------------:|-----------------:|
        | 2025-02-14 |      101 |     -14 |           1747.63 |         4854.06 |              2.7775 |            0.0126 |           0.0052 |
        | 2025-02-18 |      101 |     -14 |           1747.63 |         4932.84 |              2.8226 |            0.0126 |           0.0053 |
        | 2025-02-19 |      101 |     -14 |           1747.63 |         5149.99 |              2.9468 |            0.0126 |           0.0056 |
        | 2025-02-20 |      101 |     -14 |           1747.63 |         5261.09 |              3.0104 |            0.0126 |           0.0058 |
        | 2025-02-21 |      101 |     -14 |           1747.63 |         5092.42 |              2.9139 |            0.0126 |           0.0057 |
        | 2025-02-24 |      101 |     -14 |           1747.63 |         5035.86 |              2.8815 |            0.0126 |           0.0056 |
        | 2025-02-25 |      101 |     -14 |           1747.63 |         4937.89 |              2.8255 |            0.0126 |           0.0054 |
        | 2025-02-26 |      101 |     -14 |           1747.63 |         4903.55 |              2.8058 |            0.0126 |           0.0053 |
        | 2025-02-27 |      101 |     -14 |           1747.63 |         4913.65 |              2.8116 |            0.0126 |           0.0055 |
        | 2025-02-28 |      101 |     -14 |           1747.63 |         4932.84 |              2.8226 |            0.0126 |           0.0054 |
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
                    period_dates=self._daily_historical_data.index.get_level_values(0),
                    portfolio_dataset=self._transactions_overview,
                    historical_prices=self._daily_historical_data,
                )
            except ValueError as error:
                raise ValueError(
                    f"Failed to create positions overview due to {error}"
                ) from error

        self._positions_overview = self._positions_overview.round(
            rounding if rounding else self._rounding
        )

        return self._positions_overview

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
        - Volume: The total volume of the asset representing all transactions.
        - Costs: The total costs associated with the asset transactions.
        - Price: The mean price of the asset based on the transactions.
        - Invested: The total amount invested in the asset, this is the Price times the Volume minus Costs.
        - Latest Price: The latest available price of the asset obtained from historical data.
        - Latest Value: The total value of the asset based on the latest price and the total volume minus costs.
        - Return: The return of the asset based on the latest value and invested amount.
        - Return Value: The absolute return value of the asset based on the latest value and invested amount.
        - Benchmark Return: The return of the asset's benchmark based on the latest value and invested amount.
        - Volatility: The yearly volatility of the asset based on the historical data, this computes the volatility
        over the entire period and multiplies this number by SQRT(252).
        - Benchmark Volatility: The yearly volatility of the asset's benchmark based on the historical data, this computes
        the volatility over the entire period and multiplies this number by SQRT(252).
        - Alpha: The alpha is based on the difference between the asset's return and the benchmark return.
        - Beta: The beta is based on the asset's return and the benchmark return. It measures the asset's volatility
        compared to the benchmark. A beta >1 indicates that the asset is more volatile than the benchmark and a beta <1
        indicates that the asset is less volatile than the benchmark.
        - Weight: The weight of the asset in the portfolio based on the latest value and the total value of the portfolio.

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

        | Identifier   |   Volume |   Costs |    Price |   Invested |   Latest Price |   Latest Value |   Return |   Return Value |   Benchmark Return |   Volatility |   Benchmark Volatility |   Alpha |   Beta |   Weight |
        |:-------------|---------:|--------:|---------:|-----------:|---------------:|---------------:|---------:|---------------:|-------------------:|-------------:|-----------------------:|--------:|-------:|---------:|
        | MPWR         |      116 |     -27 | 255.881  |   29655.2  |        170.28  |       19752.5  |  -0.3339 |     -9902.68   |             1.1199 |       0.3064 |                 0.1937 | -1.4538 | 1.1553 |   0.025  |
        | MSFT         |      105 |     -11 |  40.6437 |    4256.59 |        102.5   |       10762.5  |   1.5284 |      6505.91   |             2.9661 |       0.5366 |                 0.1937 | -1.4377 | 1.3618 |   0.0136 |
        | NFLX         |      114 |     -32 | 125.444  |   14268.6  |        124.92  |       14240.9  |  -0.0019 |       -27.6764 |             2.4024 |       0.5953 |                 0.1937 | -2.4044 | 1.7687 |   0.018  |
        | NVDA         |       69 |     -27 |   2.0551 |     114.8  |         74.52  |        5141.88 |  43.7898 |      5027.08   |             2.8726 |       0.6826 |                 0.1937 | 40.9172 | 1.2835 |   0.0065 |
        | OXY          |       27 |     -15 |  39.7355 |    1057.86 |         46.1   |        1244.7  |   0.1766 |       186.842  |             3.2188 |       0.4365 |                 0.1937 | -3.0421 | 1.2279 |   0.0016 |
        | SKY          |      126 |     -23 |  18.0884 |    2256.14 |        709.08  |       89344.1  |  38.6004 |     87087.9    |             3.4986 |       0.4572 |                 0.1937 | 35.1018 | 1.4054 |   0.1132 |
        | VOO          |       77 |     -12 | 238.499  |   18352.5  |         53.58  |        4125.66 |  -0.7752 |    -14226.8    |             1.1179 |       0.2683 |                 0.1937 | -1.8931 | 0.8366 |   0.0052 |
        | VSS          |       98 |     -21 |  77.7056 |    7594.14 |        611.01  |       59879    |   6.8849 |     52284.8    |             1.3078 |       0.4667 |                 0.1937 |  5.5771 | 1.6799 |   0.0759 |
        | WMT          |       92 |     -18 |  17.8645 |    1625.53 |         48.84  |        4493.28 |   1.7642 |      2867.75   |             2.4786 |       0.4016 |                 0.1937 | -0.7145 | 1.246  |   0.0057 |
        | Portfolio    |     2142 |    -532 |  59.8406 |  128710    |        368.549 |      789432    |   5.1334 |    660721      |             2.0773 |       0.4187 |                 0.1937 |  3.0561 | 1.3272 |   1      |
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
            self._portfolio_volatilities = pd.concat(
                [
                    self._daily_historical_data["Volatility"].iloc[-1],
                    pd.Series(
                        [self._daily_benchmark_data["Volatility"].iloc[-1]],
                        index=["Benchmark"],
                    ),
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

            # Compute beta for each portfolio ticker against the benchmark
            betas = {}
            for ticker in portfolio_returns.columns:
                cov = portfolio_returns[ticker].cov(benchmark_returns)
                var = benchmark_returns.var()
                beta = cov / var if var != 0 else float("nan")
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
            )
        except ValueError as error:
            raise ValueError(f"Failed to create portfolio overview: {error}") from error

        if exclude_sold_positions:
            self._portfolio_overview = self._portfolio_overview[
                self._portfolio_overview["Volume"] > 0
            ]

        self._portfolio_overview = self._portfolio_overview.round(
            rounding if rounding else self._rounding
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

        | Date                  | Identifier   |   Volume |   Costs |   Invested Amount |   Current Value |   Invested Weight |   Current Weight |   Return |
        |:----------------------|:-------------|---------:|--------:|------------------:|----------------:|------------------:|-----------------:|---------:|
        | 2025-02-24/2025-03-02 | META         |       15 |      -1 |           4794.75 |          803.7  |            0.0346 |           0.0009 |  -0.8324 |
        | 2025-02-24/2025-03-02 | MPWR         |      122 |     -24 |          30077.9  |        20774.2  |            0.217  |           0.0228 |  -0.3093 |
        | 2025-02-24/2025-03-02 | MSFT         |      110 |     -11 |           4362.18 |        11275    |            0.0315 |           0.0124 |   1.5847 |
        | 2025-02-24/2025-03-02 | NFLX         |      125 |     -27 |          16387.9  |        15615    |            0.1183 |           0.0172 |  -0.0472 |
        | 2025-02-24/2025-03-02 | NVDA         |       81 |     -23 |            149.66 |         6036.12 |            0.0011 |           0.0066 |  39.3322 |
        | 2025-02-24/2025-03-02 | OXY          |       38 |      -4 |           1424.45 |         1751.8  |            0.0103 |           0.0019 |   0.2298 |
        | 2025-02-24/2025-03-02 | SKY          |      136 |     -19 |           2455.75 |        96434.9  |            0.0177 |           0.1059 |  38.269  |
        | 2025-02-24/2025-03-02 | VOO          |       79 |     -12 |          18660.8  |         4232.82 |            0.1347 |           0.0046 |  -0.7732 |
        | 2025-02-24/2025-03-02 | VSS          |      109 |     -19 |           8393.99 |        66600.1  |            0.0606 |           0.0732 |   6.9343 |
        | 2025-02-24/2025-03-02 | WMT          |      101 |     -14 |           1747.63 |         4932.84 |            0.0126 |           0.0054 |   1.8226 |
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

        self._portfolio_performance = self._portfolio_performance.round(
            rounding if rounding else self._rounding
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
            - If no rounding is provided, the rounding precision specified in the configuration is used.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_transactions_overview()
        ```

        Which returns:

        | Date       | Name   |    Price |   Volume |   Costs | Currency   |   Invested Amount |   Current Value |   % Return |     Return |   PnL |   Cumulative PnL |
        |:-----------|:-------|---------:|---------:|--------:|:-----------|------------------:|----------------:|-----------:|-----------:|------:|-----------------:|
        | 2024-02-01 | GOOGL  | 140.041  |       14 |       0 | USD        |          1960.57  |         2383.92 |     0.2159 |   423.351  |     0 |          4724.68 |
        | 2024-02-12 | BAC    |  33.0126 |        5 |      -3 | USD        |           162.063 |         9431.75 |    57.198  |  9269.69   |     0 |          4724.68 |
        | 2024-02-22 | MCHI   |  38.4963 |        7 |      -1 | USD        |           268.474 |          375.06 |     0.397  |   106.586  |     0 |          4724.68 |
        | 2024-03-12 | MPWR   | 727.32   |       11 |       0 | USD        |          8000.52  |         1873.08 |    -0.7659 | -6127.44   |     0 |          4737.7  |
        | 2024-05-14 | CAMT   |  94.4243 |        4 |       0 | USD        |           377.697 |          195.36 |    -0.4828 |  -182.337  |     0 |          4737.7  |
        | 2024-06-11 | META   | 505.574  |        8 |       0 | USD        |          4044.59  |          428.64 |    -0.894  | -3615.95   |     0 |          4737.7  |
        | 2024-06-18 | MPWR   | 847.6    |       14 |      -1 | USD        |         11865.4   |         2383.92 |    -0.7991 | -9481.48   |     0 |          4737.7  |
        | 2024-07-30 | AMD    | 139.57   |        3 |       0 | USD        |           418.711 |         1089.99 |     1.6032 |   671.279  |     0 |          5092.28 |
        | 2024-10-25 | MCHI   |  48.8436 |        6 |       0 | USD        |           293.062 |          321.48 |     0.097  |    28.4183 |     0 |          5703.28 |
        | 2024-11-13 | VOO    | 552.136  |       11 |       0 | USD        |          6073.5   |          589.38 |    -0.903  | -5484.12   |     0 |          5745.3  |
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

        if exclude_sold_positions:
            self._transactions_overview = self._transactions_overview[
                self._transactions_overview["Volume"] > 0
            ]

        self._transactions_overview = self._transactions_overview.round(
            rounding if rounding else self._rounding
        )

        return self._transactions_overview

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

        | Date   |   Volume |    Price |   Costs |   Invested Amount |   Current Value |   Return |   Benchmark Return |   Alpha |
        |:-------|---------:|---------:|--------:|------------------:|----------------:|---------:|-------------------:|--------:|
        | 2023Q4 |       15 |  41.0726 |      -3 |           619.088 |         597.029 |  -0.0356 |             0.0906 | -0.1262 |
        | 2024Q1 |        5 |  33.0126 |      -3 |           168.063 |        6248.05  |  36.1768 |             0.0463 | 36.1305 |
        | 2024Q1 |       14 | 140.041  |       0 |          1960.57  |        2105.39  |   0.0739 |             0.071  |  0.0029 |
        | 2024Q1 |        7 |  38.4963 |      -1 |           270.474 |         271.706 |   0.0046 |             0.0329 | -0.0283 |
        | 2024Q1 |       11 | 727.32   |       0 |          8000.52  |        1654.23  |  -0.7932 |             0.0153 | -0.8085 |
        | 2024Q2 |        4 |  94.4243 |       0 |           377.697 |         249.923 |  -0.3383 |             0.0407 | -0.379  |
        | 2024Q2 |        8 | 505.574  |       0 |          4044.59  |         331.03  |  -0.9182 |             0.0158 | -0.934  |
        | 2024Q2 |       14 | 847.6    |      -1 |         11867.4   |        2543.81  |  -0.7856 |            -0.0048 | -0.7808 |
        | 2024Q4 |        6 |  48.8436 |       0 |           293.062 |         281.16  |  -0.0406 |             0.0127 | -0.0533 |
        | 2024Q4 |       11 | 552.136  |       0 |          6073.5   |         515.46  |  -0.9151 |            -0.0173 | -0.8978 |
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

        self._transactions_performance = self._transactions_performance.round(
            rounding if rounding else self._rounding
        )

        return self._transactions_performance
