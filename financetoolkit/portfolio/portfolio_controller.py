"""Portfolio Module"""

import os
import shutil
from importlib import resources

import pandas as pd

from financetoolkit.portfolio import helpers, overview_model, portfolio_model
from financetoolkit.toolkit_controller import Toolkit

# pylint: disable=too-many-instance-attributes,abstract-class-instantiated,
# pylint: disable=too-few-public-methods,protected-access,too-many-lines


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
            print(
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

        Returns:
            Toolkit:
                A Finance Toolkit object.
        """
        if self._api_key is None:
            print(
                "The parameter api_key is not set. Therefore, only historical data and "
                "indicators are available. Consider obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan has a limit of 5 years of fundamental data and has no quarterly data. "
                "You can get 15% off by using the above affiliate link to get access to 30+ years "
                "of (quarterly) data."
            )

        if self._weekly_historical_data.empty:
            self.collect_historical_data()
        if self._weekly_benchmark_data.empty:
            self.collect_benchmark_historical_data()

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

        Parameters:
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

        Note:
            - Duplicates are handled according to configuration settings ('self._cfg["general"]["adjust_duplicates"]').
            - If duplicate data is found in the combination of datasets, it will be removed to prevent double-counting.
            - The date columns are converted to datetime objects, and transaction descriptions are treated as
            categorical data.
            - Transaction amount columns are converted to float, with support for different decimal separators.
            - Cost or income columns are processed as categorical data, with optional customization.
            - The dataset is sorted by the date column in ascending order, and the index is set to both
            the date and ticker columns.
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
        """
        if self._weekly_historical_data.empty:
            self.collect_historical_data()

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

        Note:
            - If currency conversions are applied, a warning is displayed when mismatches between transaction and
              historical data currencies are found (e.g., for ISIN codes).
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
            print(
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
        """
        if self._weekly_historical_data.empty:
            try:
                self.collect_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect historical data due to {error}"
                ) from error

        if self._weekly_benchmark_data.empty:
            try:
                self.collect_benchmark_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect benchmark historical data due to {error}"
                ) from error

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
        """
        if self._weekly_historical_data.empty:
            try:
                self.collect_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect historical data: {error}"
                ) from error

        if self._weekly_benchmark_data.empty:
            try:
                self.collect_benchmark_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect benchmark historical data: {error}"
                ) from error

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
        """
        if not period:
            period = "quarterly" if self._quarterly else "yearly"

        if self._weekly_historical_data.empty:
            try:
                self.collect_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect historical data: {error}"
                ) from error

        if self._weekly_benchmark_data.empty:
            try:
                self.collect_benchmark_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect benchmark historical data: {error}"
                ) from error

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
        """
        pnl_method = pnl_method.upper()

        if pnl_method not in ["FIFO", "LIFO", "AVERAGE"]:
            raise ValueError(
                "Please provide a valid method. This can be 'FIFO', 'LIFO', or 'AVERAGE'"
            )

        if self._weekly_historical_data.empty:
            try:
                self.collect_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect historical data: {error}"
                ) from error

        if self._weekly_benchmark_data.empty:
            try:
                self.collect_benchmark_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect benchmark historical data: {error}"
                ) from error

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
            print(f"Failed to create PnL overview: {error}")

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
        """
        if not period:
            period = "quarterly" if self._quarterly else "yearly"

        if self._weekly_historical_data.empty:
            try:
                self.collect_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect historical data: {error}"
                ) from error

        if self._weekly_benchmark_data.empty:
            try:
                self.collect_benchmark_historical_data()
            except ValueError as error:
                raise ValueError(
                    f"Failed to collect benchmark historical data: {error}"
                ) from error

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
