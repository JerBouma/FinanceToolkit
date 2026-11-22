"""Portfolio Module"""

import os
import shutil
from importlib import resources

import numpy as np
import pandas as pd

from financetoolkit.helpers import get_non_trading_days
from financetoolkit.historical_model import convert_daily_to_other_period
from financetoolkit.portfolio import (
    definition_model,
    helpers,
    overview_model,
    portfolio_model,
)
from financetoolkit.portfolio.construction.attribution import attribution_model
from financetoolkit.portfolio.definition_model import PortfolioDefinition
from financetoolkit.toolkit_controller import Toolkit
from financetoolkit.utilities import logger_model, requests_model, validation_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding, to_period_index

logger = logger_model.get_logger()

# ruff: noqa: E501

_PERIOD_SYMBOLS = {
    "yearly": "Y",
    "quarterly": "Q",
    "monthly": "M",
    "weekly": "W",
    "daily": "D",
}


@filter_shown_columns
@track_sources
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
        use_cached_data: bool | str | None = None,
        weights: dict[str, float] | pd.Series | str | None = None,
        holdings: dict[str, float] | None = None,
        transactions: pd.DataFrame | str | None = None,
        definition: PortfolioDefinition | None = None,
        start_date: str | None = None,
        initial_value: float = 10000.0,
        rebalancing: str | None = None,
        currency: str | None = None,
    ):
        """
        Initialize the Portfolio class from weights, holdings, a classic portfolio
        (`Portfolio.from_preset`) or your own transactions.

        This constructor sets up a portfolio management instance, configuring it with a dataset (either as a
        pandas DataFrame or a file path to an Excel/CSV file), a benchmark ticker for performance comparison,
        and various settings specified through a configuration file (YAML format). The configuration file defines
        key columns for cash flow analysis, such as date columns, asset tickers, price, volume, and transaction costs.
        It also allows for the use of quarterly data and fetching of historical financial data via an API.

        A portfolio can be given in four ways:

        - weights, e.g. {"VTI": 0.6, "BND": 0.4} or "VTI:0.6, BND:0.4": a model portfolio that
          invests `initial_value` on `start_date` and is rebalanced to these weights. It trades
          at the closing prices, the prices the positions are valued at. The dividends are
          collected as cash and reinvested on the next rebalancing date, when the portfolio is
          brought back to its weights with the value of the holdings plus that cash, so it is
          a total-return backtest and its weights equal the targets on every rebalancing date.
          Holdings and transactions receive their dividends as cash that is not reinvested;
        - holdings, e.g. {"AAPL": 10, "MSFT": 5}: the number of shares, bought at the closing
          price of `start_date`;
        - transactions: a DataFrame, CSV or Excel file with one row per trade and the columns
          Date, Ticker (or Symbol, Identifier, ISIN), Volume (or Quantity, Shares; negative
          for a sale) and optionally Price, Costs (negative) and Currency. A missing Price is
          filled with the closing price of the day;
        - a definition (`PortfolioDefinition`), as the classic portfolios of
          `Portfolio.from_preset` are.

        Weights, holdings and definitions are valued in one currency, `currency`: the price of
        every ticker that trades in another currency is converted with the exchange rate of
        each day (pence to pounds included), so the weights and values add up.

        Args:
            portfolio_dataset (pd.DataFrame | str | None): The transactions, as a DataFrame or the path to an
                Excel or CSV file (or a folder of them). Equivalent to `transactions`. Defaults to None.
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
            use_cached_data (bool | str | None): Whether to cache the data retrieved from external sources, as
                in the Toolkit. None or True uses the shared cache, False retrieves everything every time and a
                string is the path to a dedicated cache folder or database file. Defaults to None, which caches
                unless the FINANCE_TOOLKIT_CACHE_ENABLED environment variable is set to 0.
            weights (dict[str, float] | pd.Series | str | None): The target weight per ticker, e.g.
                {"VTI": 0.6, "BND": 0.4}, "VTI:0.6, BND:0.4" or the table an optimizer of
                `toolkit.portfolio` returns. Defaults to None.
            holdings (dict[str, float] | None): The number of shares per ticker, bought on `start_date`.
                Defaults to None.
            transactions (pd.DataFrame | str | None): The transactions, as a DataFrame or the path to a CSV or
                Excel file. Defaults to None.
            definition (PortfolioDefinition | None): A portfolio definition with weights, rebalancing and
                optionally a schedule. Defaults to None.
            start_date (str | None): When weights or holdings are invested. Defaults to None, ten years ago
                for weights and five years ago for holdings, or the first day all tickers have a price.
            initial_value (float): The amount invested in a weights portfolio. Defaults to 10,000.
            rebalancing (str | None): How often a weights portfolio returns to its weights: "none",
                "monthly", "quarterly" or "yearly". Defaults to None, the definition's frequency or quarterly.
            currency (str | None): The currency weights, holdings and definitions are valued in, e.g. "USD" or
                "EUR". Defaults to None, the currency most of the portfolio trades in: the currency with the
                largest share of the weights, or for holdings the currency of most tickers.

        Raises:
            ValueError: If the provided configuration file is not in YAML format.
            ValueError: If no portfolio dataset is provided and `example` is set to False.

        As an example:

        ```python
        from financetoolkit import Portfolio

        # A model portfolio from weights, rebalanced every quarter
        portfolio = Portfolio(
            weights={"VTI": 0.6, "BND": 0.4},
            api_key="FINANCIAL_MODELING_PREP_KEY",
        )

        # A classic portfolio
        portfolio = Portfolio.from_preset("All Weather", api_key="FINANCIAL_MODELING_PREP_KEY")

        # The shares you hold
        portfolio = Portfolio(holdings={"AAPL": 10, "MSFT": 5}, start_date="2022-01-03")

        # Your own trades, from a CSV or Excel file or a DataFrame
        portfolio = Portfolio(transactions="my_trades.csv")

        # The example portfolio
        portfolio = Portfolio(example=True)
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

        if transactions is not None:
            portfolio_dataset = transactions
        self._definition: PortfolioDefinition | None = None
        built_portfolio = (
            weights is not None or holdings is not None or definition is not None
        )

        # An explicit None check: the truth value of a DataFrame is ambiguous.
        if portfolio_dataset is None and not example and not built_portfolio:
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

        if example:
            portfolio_dataset = example_xlsx_path

        self._currency: str | None = (
            portfolio_model.get_price_unit(currency)[0] if currency else None
        )
        # The currency the prices of each ticker of a built portfolio are quoted in, so the
        # historical data is converted with the same currencies as its transactions.
        self._price_currencies: dict[str, str] = {}

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
        self._use_cached_data: bool | str | None = use_cached_data
        self._benchmark_ticker = (
            benchmark_ticker
            if benchmark_ticker
            else self._cfg["general"]["benchmark_ticker"]
        )

        # Tickers
        self._ticker_combinations: dict[str, str] = {}
        self._data_tickers: dict[str, str] = {}

        # Historical Data
        self._daily_historical_data: pd.DataFrame = pd.DataFrame()
        self._weekly_historical_data: pd.DataFrame = pd.DataFrame()
        self._historical_statistics: pd.DataFrame = pd.DataFrame()

        # Benchmark Historical Data
        self._benchmark_tickers: dict[str, str] = {}
        self._daily_benchmark_data: pd.DataFrame = pd.DataFrame()
        self._weekly_benchmark_data: pd.DataFrame = pd.DataFrame()
        self._benchmark_specific_prices: pd.Series = pd.Series()
        self._benchmark_transaction_dividends: pd.Series = pd.Series()
        self._benchmark_prices_per_ticker: pd.DataFrame = pd.DataFrame()
        self._benchmark_dividends_per_ticker: pd.DataFrame = pd.DataFrame()
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
        self._latest_price: pd.Series = pd.Series()
        self._daily_currency_data: pd.DataFrame = pd.DataFrame()

        # Column Names
        self._date_column: str = self._cfg["general"]["date_columns"]
        self._name_column: str = self._cfg["general"]["name_columns"]
        self._ticker_column: str = self._cfg["general"]["ticker_columns"]
        self._price_column: str = self._cfg["general"]["price_columns"]
        self._volume_column: str = self._cfg["general"]["volume_columns"]
        self._costs_column: str = self._cfg["general"]["costs_columns"]

        # Weights, holdings and definitions are turned into the transactions that invest them,
        # so every analysis of the transactions applies to them as well.
        if built_portfolio:
            portfolio_dataset = self._build_portfolio(
                weights=weights,
                holdings=holdings,
                definition=definition,
                start_date=start_date,
                initial_value=initial_value,
                rebalancing=rebalancing,
            )
        elif portfolio_dataset is not None and not example:
            if isinstance(portfolio_dataset, pd.DataFrame) and portfolio_dataset.empty:
                raise ValueError("The portfolio has no transactions.")
            portfolio_dataset = self._complete_transactions(portfolio_dataset)

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

    def _price_toolkit(self, tickers: list[str], start_date: str) -> Toolkit:
        """A Toolkit for the tickers, reused later for the historical data."""
        # Unrounded, as the portfolio is calculated from the prices this Toolkit returns.
        self._toolkit = Toolkit(
            api_key=self._api_key,
            tickers=tickers,
            benchmark_ticker=None,
            start_date=start_date,
            use_cached_data=self._use_cached_data,
            progress_bar=False,
            rounding=None,
        )
        return self._toolkit

    def _ticker_currencies(self, toolkit: Toolkit) -> dict[str, str]:
        """The currency every ticker is quoted in, according to its historical statistics."""
        try:
            statistics = toolkit.get_historical_statistics()
        except Exception:  # noqa: BLE001 - the statistics are optional
            return {}
        if statistics is None or statistics.empty or "Currency" not in statistics.index:
            return {}
        currencies = statistics.loc["Currency"]
        # A single ticker comes back as one column, which gives a single value.
        if not isinstance(currencies, pd.Series):
            currencies = pd.Series({toolkit._tickers[0]: currencies})
        return {
            str(ticker): str(currency)
            for ticker, currency in currencies.items()
            if isinstance(currency, str) and len(currency) == 3  # noqa: PLR2004
        }

    def _get_exchange_rates(
        self, rate_tickers: list[str | None], start_date: str
    ) -> pd.DataFrame:
        """
        The daily exchange rates, one column per rate such as USDGBP=X (None, a conversion
        that needs no rate, is skipped), or an empty DataFrame when none is needed. They are
        retrieved once and kept, so the prices of a built portfolio's transactions and its
        historical data are converted with the same rates and a transaction's price equals
        the converted price of its day.
        """
        rate_tickers = sorted(
            {rate_ticker for rate_ticker in rate_tickers if rate_ticker}
        )
        if not rate_tickers:
            return pd.DataFrame()
        # Ten days earlier, so a transaction on a weekend or holiday at the start has the last rate before it.  # noqa: E501
        start_date = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime(
            "%Y-%m-%d"
        )

        retrieved = (
            set()
            if self._daily_currency_data.empty
            else set(self._daily_currency_data["Adj Close"].columns)
        )
        if not retrieved.issuperset(rate_tickers):
            # The rates retrieved before are retrieved again with the new ones, so one
            # Toolkit holds every rate the portfolio uses.
            collect = sorted(retrieved.union(rate_tickers))
            self._daily_currency_data = Toolkit(
                api_key=self._api_key,
                tickers=collect,
                benchmark_ticker=None,
                start_date=start_date,
                use_cached_data=self._use_cached_data,
                progress_bar=False,
                # Unrounded: a rate such as JPY to EUR (0.0061) loses digits at four decimals.
                rounding=None,
            ).get_historical_data(period="daily")
            # A single ticker comes back without the ticker level.
            if self._daily_currency_data.columns.nlevels == 1 and len(collect) == 1:
                self._daily_currency_data.columns = pd.MultiIndex.from_product(
                    [self._daily_currency_data.columns, collect]
                )

        # A rate that could not be retrieved is an empty column, which the conversion
        # reports by name.
        if self._daily_currency_data.empty:
            return pd.DataFrame(columns=rate_tickers, dtype=float)
        return self._daily_currency_data["Adj Close"].reindex(columns=rate_tickers)

    def _quoted_currencies(
        self, toolkit: Toolkit, tickers: list[str], fallback: dict[str, str]
    ) -> dict[str, str]:
        """
        The currency the prices of every ticker are quoted in: the one of the historical
        statistics, otherwise the one given (e.g. by a portfolio definition).
        """
        statistics = self._ticker_currencies(toolkit)
        currencies = {
            ticker: statistics.get(ticker) or fallback.get(ticker) for ticker in tickers
        }
        return {ticker: currency for ticker, currency in currencies.items() if currency}

    def _benchmark_currencies(
        self, benchmark_ticker: str
    ) -> tuple[str | None, dict[str, str]]:
        """
        The currency the benchmark is quoted in and per ticker the currency of its transactions,
        or no currencies when the transactions have no currency.
        """
        if (
            not self._currency_column
            or self._currency_column not in self._portfolio_dataset.columns
        ):
            return None, {}

        ticker_currencies = {
            str(ticker): str(currency)
            for (_, ticker), currency in self._portfolio_dataset[
                self._currency_column
            ].items()
            if isinstance(currency, str) and currency
        }

        benchmark_currencies = (
            self._ticker_currencies(self._benchmark_toolkit)
            if self._benchmark_toolkit
            else {}
        )
        benchmark_currency = benchmark_currencies.get(benchmark_ticker) or (
            next(iter(benchmark_currencies.values()))
            if len(benchmark_currencies) == 1
            else None
        )

        return benchmark_currency, ticker_currencies

    def _collect_data(self) -> bool:
        """
        Collect the historical data of the tickers and the benchmark when it is not there yet,
        False when the tickers or the benchmark have no historical data, for which every
        method returns an empty result.
        """
        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return False

        if self._weekly_benchmark_data.empty:
            self.collect_benchmark_historical_data()

        return not self._daily_benchmark_data.empty

    def _transaction_locations(self) -> np.ndarray:
        """
        The position of every transaction's trading day in the dates of the historical data: the
        transaction's date, or the first day after it that its ticker traded on when it did not
        trade that day (such as a weekend, also next to a cryptocurrency that did), or the last
        date for a transaction after it (see `helpers.get_trading_day_locations`).
        """
        return helpers.get_trading_day_locations(
            self._daily_historical_data,
            self._portfolio_dataset.index.get_level_values(0),
            self._portfolio_dataset.index.get_level_values(1),
        )

    def _values_at_transactions(self, daily_values: pd.DataFrame) -> pd.Series:
        """
        The value of every transaction's ticker on the transaction's date, from daily values with
        one column per ticker on the dates of the historical data. A transaction on a date its
        ticker did not trade on, such as a weekend, takes the ticker's first following trading day,
        the day its position starts on, and a transaction after the last date takes the last date,
        the close it is priced at.
        """
        locations = self._transaction_locations()
        tickers = self._portfolio_dataset.index.get_level_values(1)

        values = [
            (
                float(daily_values[ticker].iloc[location])
                if ticker in daily_values.columns
                else float("nan")
            )
            for location, ticker in zip(locations, tickers, strict=True)
        ]

        return pd.Series(values, index=self._portfolio_dataset.index, dtype=float)

    def _open_positions(self, volume: pd.Series) -> pd.Series:
        """
        Which positions are open, long or short, from their net volume indexed by identifier or by
        period and identifier. A volume left by floating point residue of fully sold fractional
        units is a closed position, on the scale of the largest transaction in the asset.
        """
        largest_volume = (
            self._portfolio_dataset[self._volume_column]
            .abs()
            .groupby(level=1, observed=True)
            .max()
        )

        return overview_model.is_open_position(volume, largest_volume)

    def _ticker_historical_data(self, period: str) -> pd.DataFrame:
        """
        The historical data of the portfolio's tickers for a period, with the ticker as the second
        column level under the portfolio's own ticker names. A Toolkit of a single ticker returns
        flat columns, so the ticker level is added for it.
        """
        data = self._toolkit.get_historical_data(period=period)  # type: ignore[union-attr]

        if data.empty:
            return data

        # One ticker comes back without the ticker level, which every position metric selects on.
        if data.columns.nlevels == 1:
            data = data.copy()
            data.columns = pd.MultiIndex.from_product(
                [data.columns, [self._toolkit._tickers[0]]]  # type: ignore[union-attr]
            )

        return data.rename(columns=self._ticker_combinations, level=1)

    def _cumulative_dividends(self) -> pd.DataFrame:
        """The cumulative cash dividends per unit of every ticker on the dates of the historical data."""
        if "Dividends" not in self._daily_historical_data.columns.get_level_values(0):
            return pd.DataFrame(
                0.0, index=self._daily_historical_data.index, columns=self._tickers
            )

        return (
            self._daily_historical_data["Dividends"]
            .reindex(columns=self._tickers)
            .astype(float)
            .fillna(0.0)
            .cumsum()
        )

    def _dividends_since_transactions(
        self, cumulative_dividends: pd.DataFrame
    ) -> pd.Series:
        """
        The dividends per unit paid between every transaction and the latest date, from cumulative
        dividends per unit with one column per ticker. A unit bought on an ex-dividend date does
        not receive that dividend, so the cumulative figure of the transaction date is subtracted.
        """
        latest = cumulative_dividends.iloc[-1]
        at_transactions = self._values_at_transactions(cumulative_dividends)
        tickers = self._portfolio_dataset.index.get_level_values(1)

        return pd.Series(
            latest.reindex(tickers).to_numpy() - at_transactions.to_numpy(),
            index=self._portfolio_dataset.index,
        )

    def _build_portfolio(
        self,
        weights: dict[str, float] | pd.Series | str | None,
        holdings: dict[str, float] | None,
        definition: PortfolioDefinition | None,
        start_date: str | None,
        initial_value: float,
        rebalancing: str | None,
    ) -> pd.DataFrame:
        """
        The transactions that invest weights, holdings or a definition, in the portfolio's
        currency: every price is converted to it, so the weights and values add up.
        """
        today = pd.Timestamp.today()

        if holdings is not None:
            start = start_date or (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
            tickers = list(holdings)
            toolkit = self._price_toolkit(tickers, start)
            # Shares are bought at the price that was paid for them, the traded close, as
            # transactions without a price are; the dividend-adjusted close of the past is
            # lower than what anyone paid.
            prices = toolkit.get_historical_data(period="daily")["Close"]
            prices = (
                prices.to_frame(tickers[0]) if isinstance(prices, pd.Series) else prices
            )
            prices = prices.reindex(columns=tickers)
            currencies = self._quoted_currencies(toolkit, tickers, {})
            base_currency = self._currency or portfolio_model.get_base_currency(
                currencies, dict.fromkeys(tickers, 1.0)
            )
            self._price_currencies = currencies
            first = portfolio_model.convert_prices(
                prices,
                currencies,
                base_currency,
                self._get_exchange_rates(
                    portfolio_model.get_exchange_rate_tickers(
                        currencies, base_currency
                    ),
                    start,
                ),
            ).dropna(how="any")
            if first.empty:
                raise ValueError(
                    "The holdings have no day on which all of them have a price."
                )
            day = first.index[0]
            self._definition = PortfolioDefinition.from_weights(
                {
                    ticker: shares * float(first.loc[day, ticker])
                    for ticker, shares in holdings.items()
                },
                name="Holdings",
                rebalancing="none",
            )
            return pd.DataFrame(
                [
                    {
                        "Date": day.strftime("%Y-%m-%d"),
                        "Identifier": ticker,
                        "Name": ticker,
                        "Price": float(first.loc[day, ticker]),
                        "Volume": float(shares),
                        "Costs": 0.0,
                        "Currency": base_currency,
                    }
                    for ticker, shares in holdings.items()
                ]
            )

        if definition is None:
            definition = PortfolioDefinition.from_weights(
                weights,  # type: ignore[arg-type]
                rebalancing=rebalancing or "quarterly",
            )
        elif rebalancing is not None:
            definition = PortfolioDefinition.from_dict(
                {**definition.to_dict(), "rebalancing": rebalancing}
            )
        self._definition = definition

        start = start_date or (today - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
        toolkit = self._price_toolkit(definition.tickers, start)
        # The transactions are made at the traded close, the price positions are valued at, and the dividends are reinvested on every rebalancing date.  # noqa: E501
        historical_data = toolkit.get_historical_data(period="daily")
        prices = historical_data["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(definition.tickers[0])
        dividends = (
            historical_data["Dividends"]
            if "Dividends" in historical_data.columns.get_level_values(0)
            else None
        )
        if isinstance(dividends, pd.Series):
            dividends = dividends.to_frame(definition.tickers[0])

        currencies = self._quoted_currencies(
            toolkit,
            definition.tickers,
            {
                holding.ticker: holding.currency
                for holding in definition.holdings
                if holding.currency
            },
        )
        base_currency = self._currency or portfolio_model.get_base_currency(
            currencies, definition.weights
        )
        self._price_currencies = currencies

        return definition_model.build_transactions(
            definition=definition,
            prices=prices,
            currencies=currencies,
            initial_value=initial_value,
            currency=base_currency,
            exchange_rates=self._get_exchange_rates(
                portfolio_model.get_exchange_rate_tickers(currencies, base_currency),
                start,
            ),
            dividends=dividends,
        )

    def _complete_transactions(
        self, dataset: pd.DataFrame | str | list
    ) -> pd.DataFrame | str | list:
        """
        Fill in the price of transactions that leave it out: the closing price of the day,
        in the currency the ticker trades in. A transaction on a day its ticker did not trade, such
        as a weekend or a holiday, takes the closing price of the ticker's first following trading
        day, the day its position, benchmark and dividends start on, also when another ticker (such
        as a cryptocurrency) traded that day. A transaction after the latest close, such
        as one of today before today's close is available, takes the latest close, with a warning.
        Files and folders that have a price column are passed on as they are.
        """
        if isinstance(dataset, list) or (
            isinstance(dataset, str) and not dataset.endswith(("csv", "xlsx", "xls"))
        ):
            return dataset

        frame = (
            dataset
            if isinstance(dataset, pd.DataFrame)
            else helpers.read_excel(dataset)
        )
        columns = {str(column).strip().lower(): column for column in frame.columns}
        general = self._cfg["general"]

        def find(names: list[str]):
            return next(
                (columns[name.lower()] for name in names if name.lower() in columns),
                None,
            )

        date = find(general["date_columns"])
        ticker = find(general["ticker_columns"])
        if find(general["price_columns"]) is not None or date is None or ticker is None:
            return dataset

        frame = frame.copy()
        # Parsed and written as the dataset is read later, so each price belongs to the day
        # and the ticker its transaction is booked on.
        dates = portfolio_model.parse_dates(frame[date], general["date_format"])
        symbols = frame[ticker].astype(str).str.strip().str.upper()
        tickers = list(symbols.unique())
        toolkit = self._price_toolkit(
            tickers, (dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        )
        historical_data = toolkit.get_historical_data(period="daily")
        closes = historical_data["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(tickers[0])
        else:
            # The dates only another ticker traded on carry a ticker's last close forward, which would price a Saturday purchase of a stock next to a cryptocurrency at Friday's close; only the ticker's own trading days hold its price.  # noqa: E501
            closes = closes.mask(get_non_trading_days(historical_data, "Close"))
        closes = closes.copy()
        closes.index = closes.index.to_timestamp()
        closes = closes.sort_index()

        def first_close(symbol: str, day: pd.Timestamp) -> float:
            # The first close of the ticker's own trading days on or after the date, as the positions, the benchmark and the dividends use: the previous close would price a weekend purchase on Friday while its position starts on Monday.  # noqa: E501
            if symbol not in closes.columns or pd.isna(day):
                return float("nan")
            available = closes[symbol].dropna()
            following = available.loc[day.normalize() :]
            if not following.empty:
                return float(following.iloc[0])
            # A trade after the latest close (the weekend after it, or today before the provider has today's bar) takes the latest close, the day its position, benchmark and dividends start on as well.  # noqa: E501
            if available.empty:
                return float("nan")
            logger.warning(
                "The %s transaction of %s has no close on or after that date and is priced at the last close of %s.",
                symbol,
                day.strftime("%Y-%m-%d"),
                available.index[-1].strftime("%Y-%m-%d"),
            )
            return float(available.iloc[-1])

        frame["Price"] = [
            first_close(str(symbol), day)
            for symbol, day in zip(symbols, dates, strict=True)
        ]

        # The prices are in the currency each ticker trades in, so the transactions are too.
        if find(general["currency_columns"]) is None:
            currencies = self._ticker_currencies(toolkit)
            frame["Currency"] = [currencies.get(symbol, "USD") for symbol in symbols]

        return frame

    @staticmethod
    def get_presets() -> pd.DataFrame:
        """
        List the classic portfolios that ship with the Finance Toolkit, to use with
        `Portfolio.from_preset`. Each is defined with real funds and the source of its
        allocation: the 60/40, Ray Dalio's All Weather, Harry Browne's Permanent Portfolio,
        the Golden Butterfly, the Bogleheads three-fund portfolio, Warren Buffett's 90/10,
        Nassim Taleb's barbell, a target-date glide path and more.

        Returns:
            pd.DataFrame: Per preset (by identifier) its Name, Description, Holdings,
            Rebalancing and Aliases.

        As an example:

        ```python
        from financetoolkit import Portfolio

        Portfolio.get_presets()[["Name", "Holdings", "Rebalancing"]]
        ```

        Which returns:

        |                       | Name                             | Holdings                                                                                          | Rebalancing   |
        |:----------------------|:---------------------------------|:--------------------------------------------------------------------------------------------------|:--------------|
        | all_weather           | All Weather                      | VTI 30.0%, TLT 40.0%, GOVT 15.0%, GLD 7.5%, DBC 7.5%                                              | quarterly     |
        | balanced_60_40        | 60/40 Balanced                   | VTI 60.0%, BND 40.0%                                                                              | quarterly     |
        | barbell               | Barbell                          | BIL 85.0%, VB 7.5%, VUG 7.5%                                                                      | yearly        |
        | bogleheads_three_fund | Bogleheads Three-Fund            | VTI 42.0%, VXUS 18.0%, BND 40.0%                                                                  | yearly        |
        | buffett_90_10         | Buffett's 90/10                  | VOO 90.0%, BIL 10.0%                                                                              | yearly        |
        | conservative_income   | Conservative Multi-Country Bonds | VTI 10.0%, GOVT 40.0%, IEGA.AS 30.0%, IGLT.L 20.0%                                                | quarterly     |
        | global_60_40          | Global 60/40                     | VTI 30.0%, VGK 15.0%, VWO 15.0%, GOVT 20.0%, IEGA.AS 10.0%, IGLT.L 10.0%                          | quarterly     |
        | global_equity_growth  | Global Equity Growth             | VTI 30.0%, VGK 15.0%, EWJ 10.0%, VWO 15.0%, MCHI 10.0%, VUG 20.0%                                 | yearly        |
        | golden_butterfly      | Golden Butterfly                 | VTI 20.0%, VBR 20.0%, TLT 20.0%, SHY 20.0%, GLD 20.0%                                             | yearly        |
        | permanent_portfolio   | Permanent Portfolio              | VTI 25.0%, TLT 25.0%, BIL 25.0%, GLD 25.0%                                                        | yearly        |
        | small_cap_value_tilt  | Small-Cap Value Tilt             | VB 40.0%, VTV 40.0%, GOVT 20.0%                                                                   | yearly        |
        | target_date_glidepath | Target-Date Glidepath            | VTI 90.0%, BND 10.0%                                                                              | quarterly     |
        | us_equity_sectors     | US Equity Sectors                | XLK 11.1%, XLF 11.1%, XLE 11.1%, XLV 11.1%, XLI 11.1%, XLY 11.1%, XLP 11.1%, XLU 11.1%, XLB 11.1% | quarterly     |
        """
        rows = {
            identifier: {
                "Name": definition.name,
                "Description": definition.description,
                "Holdings": ", ".join(
                    f"{holding.ticker} {holding.weight:.1%}"
                    for holding in definition.holdings
                ),
                "Rebalancing": definition.rebalancing,
                "Aliases": ", ".join(definition.aliases),
            }
            for identifier, definition in definition_model.load_presets().items()
        }
        return pd.DataFrame(rows).T

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "Portfolio":
        """
        Create a Portfolio from one of the classic portfolios, by its identifier, name or
        alias (see `Portfolio.get_presets`), e.g. "60/40", "All Weather", "Permanent",
        "Golden Butterfly" or "Three Fund". It invests `initial_value` on `start_date` and
        rebalances as the definition says, after which every Portfolio and Toolkit analysis
        applies.

        Args:
            name (str): The identifier, name or alias of the preset.
            **kwargs: The other arguments of Portfolio, such as api_key, start_date,
                initial_value, rebalancing and benchmark_ticker.

        Returns:
            Portfolio: The portfolio.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio.from_preset(
            "Permanent Portfolio",
            start_date="2016-01-01",
            api_key="FINANCIAL_MODELING_PREP_KEY",
        )

        portfolio.get_portfolio_overview()
        ```
        """
        return cls(definition=definition_model.find_preset(name), **kwargs)

    @property
    def definition(self) -> PortfolioDefinition:
        """
        The portfolio as a `PortfolioDefinition`: the definition it was built from (a preset
        or weights), or, for holdings and transactions, the current weight of every position.
        It can be stored with `to_yaml` and read by other tools.

        Returns:
            PortfolioDefinition: The definition.
        """
        if self._definition is not None and self._definition.name != "Holdings":
            return self._definition

        overview = self.get_portfolio_overview()
        weights = overview["Weight"].drop("Portfolio", errors="ignore")
        # A holding without a price leaves the value of the portfolio, and with it every weight, unknown.
        unpriced = overview.index[overview["Latest Value"].isna()].drop(
            "Portfolio", errors="ignore"
        )
        if weights.isna().all() and len(unpriced):
            raise ValueError(
                "The weights of the portfolio are unknown because there is no price for "
                f"{', '.join(map(str, unpriced))}."
            )
        return PortfolioDefinition.from_weights(
            weights[weights > 0], name="Portfolio", rebalancing="none"
        )

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

        if not self._collect_data():
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

        # A copy: the benchmark belongs in the Toolkit's data only, as a ticker of the Portfolio's own data it would duplicate the benchmark of every position.  # noqa: E501
        historical = self._daily_historical_data.copy()
        for column in historical_columns:
            historical[column, "Benchmark"] = benchmark_data[column]

        historical = (
            historical.sort_index(axis=1)
            .reindex(historical_columns, axis=1, level=0)
            .reindex(list(self._tickers) + ["Benchmark"], axis=1, level=1)
        )

        if not self._toolkit_instance:
            self._toolkit_instance = Toolkit(
                tickers=symbols,
                api_key=self._api_key,
                historical=historical,
                start_date=self._start_date,
                use_cached_data=self._use_cached_data,
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
                Defaults to None, which uses the configuration setting.
            date_column (list[str] | None): List of column names for date information.
                Defaults to None, which uses the configuration.
            date_format_options (list[str] | None): List of date format strings to attempt when parsing
                date columns (e.g. ['%Y-%m-%d', '%d/%m/%Y']). Defaults to None, which uses the configuration.
            name_columns (list[str] | None): List of column names for transaction descriptions.
                Defaults to None, which uses the configuration.
            ticker_columns (list[str] | None): List of column names for asset tickers.
                Defaults to None, which uses the configuration.
            price_columns (list[str] | None): List of column names for asset prices.
                Defaults to None, which uses the configuration.
            volume_columns (list[str] | None): List of column names for asset volumes.
                Defaults to None, which uses the configuration.
            currency_columns (list[str] | None): List of column names for transaction currencies.
                Defaults to None, which uses the configuration.
            costs_columns (list[str] | None): List of column names for costs or income categories.
                Defaults to None, which uses the configuration.
            column_mapping (dict[str, str] | None): Dictionary mapping dataset columns to the appropriate field names.
                Defaults to None, which uses the configuration.

        Returns:
            pd.DataFrame: A DataFrame containing the consolidated and processed portfolio data.

        Raises:
            FileNotFoundError: If any of the files or directories in the portfolio dataset path cannot be found.
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
                if adjust_duplicates is not None
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

        # Written the way the Toolkit writes them, so "aapl" and "AAPL" are one holding and
        # each ticker matches its own data. An ISIN is converted once here; when it resolves
        # to a ticker that is also in the dataset, both are the same holding.
        tickers = (
            self._portfolio_dataset[self._ticker_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        converted_tickers = {
            ticker: requests_model.convert_isin_to_ticker(ticker)
            for ticker in tickers.unique()
        }
        tickers = tickers.map(
            lambda ticker: (
                converted_tickers[ticker]
                if converted_tickers[ticker] in converted_tickers
                else ticker
            )
        )
        self._portfolio_dataset[self._ticker_column] = tickers
        self._data_tickers = {
            ticker: converted_tickers[ticker] for ticker in tickers.unique()
        }

        self._original_tickers = list(
            self._portfolio_dataset[self._ticker_column].unique()
        )

        self._portfolio_dataset = self._portfolio_dataset.sort_values(
            by=self._date_column, ascending=True, kind="stable"
        )
        self._tickers = list(self._portfolio_dataset[self._ticker_column].unique())
        if not self._tickers:
            raise ValueError("The portfolio has no transactions.")
        self._start_date = (
            self._portfolio_dataset[self._date_column].min().strftime("%Y-%m-%d")
        )

        self._portfolio_dataset = self._portfolio_dataset.set_index(
            [self._date_column, self._ticker_column]
        )

        return self._portfolio_dataset

    def collect_benchmark_historical_data(
        self,
        benchmark_ticker: str | None = None,
        show_columns: str | list[str] | None = None,
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

        Volatility, Excess Return and Excess Volatility are not included here. These are
        available via the Risk module (e.g. toolkit.risk.get_volatility) and the Performance
        module (e.g. toolkit.performance.get_excess_return) instead.

        This method retrieves historical benchmark data (daily and weekly)
        for the portfolio, based on a specified benchmark ticker or a mapping of portfolio tickers to
        their corresponding benchmark tickers. The retrieved benchmark data is then aligned with the
        portfolio's historical data, ensuring that the dates of the benchmark data match the dates of
        the portfolio's transactions.

        The method can retrieve data for a single benchmark ticker or for multiple benchmarks depending
        on the portfolio tickers. The resulting benchmark data is returned in a structured DataFrame.

        Args:
            benchmark_ticker (str | None): The benchmark ticker symbol. Defaults to None, the benchmark ticker
                given to the Portfolio, or the one of the configuration file when none was given.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the benchmark data for the portfolio, indexed by the portfolio's dates,
                or an empty DataFrame (with an error logged) when the benchmark has no historical data.

        Notes:
            - The benchmark data is retrieved in daily and weekly periods.
            - The daily data is aligned to the dates of the portfolio's historical data. A date without a
            benchmark quote, such as a holiday of the benchmark's exchange, carries the previous quote
            forward with a return of zero, and the Return and Cumulative Return are calculated again from
            the aligned Adj Close. Volume and dividends on a date missing from the portfolio's dates are
            added to the next date.
            - A transaction on a date without a price, such as a weekend, is matched with the benchmark's
            closing price of the first following trading day, the same day its position starts on.
            - When the transactions are in another currency than the benchmark, the benchmark's prices and
            dividends are converted to the currency of the transactions with the daily exchange rate, so
            the benchmark buys the same amount of money as the transaction. The returned data and the
            weekly data are in the currency most of the portfolio's tickers are traded in.
            - The method updates several internal attributes:
                - `self._daily_benchmark_data`: Daily benchmark data.
                - `self._weekly_benchmark_data`: Weekly benchmark data.
                - `self._benchmark_prices_per_ticker`: The daily benchmark closing price in the currency of
                each ticker.
                - `self._benchmark_dividends_per_ticker`: The cumulative benchmark dividends per unit in the
                currency of each ticker.
                - `self._latest_benchmark_price`: The latest available benchmark closing price per ticker.
                - `self._benchmark_specific_prices`: The benchmark closing price for each portfolio transaction.
                - `self._benchmark_transaction_dividends`: The cumulative benchmark dividends per unit up to
                each portfolio transaction.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.collect_benchmark_historical_data()
        ```

        Which returns:

        | Date       |    Open |    High |     Low |   Close |   Adj Close |      Volume |   Dividends |       Return |   Cumulative Return |
        |:-----------|--------:|--------:|--------:|--------:|------------:|------------:|------------:|-------------:|--------------------:|
        | 2026-09-24 | 7666.99 | 7719.01 | 7662.57 | 7704.13 |     7704.13 | 5.31639e+09 |           0 | -0.000246547 |             5.34463 |
        | 2026-09-25 | 7709.86 | 7752.07 | 7693.08 | 7743.41 |     7743.41 | 4.49918e+09 |           0 |  0.0050986   |             5.37188 |
        | 2026-09-28 | 7721.7  | 7724.15 | 7666.6  | 7683.69 |     7683.69 | 5.12971e+09 |           0 | -0.0077124   |             5.33045 |
        | 2026-09-29 | 7699.6  | 7699.6  | 7653.55 | 7670.84 |     7670.84 | 5.02359e+09 |           0 | -0.00167239  |             5.32154 |
        | 2026-09-30 | 7688.99 | 7722.88 | 7651.54 | 7651.54 |     7651.54 | 4.27655e+09 |           0 | -0.002516    |             5.30815 |
        | 2026-10-01 | 7666.47 | 7684.75 | 7616.78 | 7666.45 |     7666.45 | 5.7838e+09  |           0 |  0.00194865  |             5.31849 |
        | 2026-10-02 | 7726.24 | 7754.67 | 7700.51 | 7722.72 |     7722.72 | 5.27743e+09 |           0 |  0.00733977  |             5.35753 |
        | 2026-10-05 | 7730.86 | 7794.35 | 7727.59 | 7773.95 |     7773.95 | 5.8409e+09  |           0 |  0.00663367  |             5.39307 |
        | 2026-10-06 | 7805.96 | 7844.52 | 7805.96 | 7818.93 |     7818.93 | 5.01965e+09 |           0 |  0.00578599  |             5.42428 |
        | 2026-10-07 | 7792.98 | 7807.02 | 7763.34 | 7801.77 |     7801.77 | 5.11115e+09 |           0 | -0.0021947   |             5.41237 |
        """
        if self._weekly_historical_data.empty:
            self.collect_historical_data()

            if self._daily_historical_data.empty:
                return pd.DataFrame()

        # The constructor already fell back to the configuration's benchmark.
        benchmark_ticker = benchmark_ticker or self._benchmark_ticker

        if not self._benchmark_toolkit:
            self._benchmark_tickers = dict.fromkeys(
                self._original_tickers, benchmark_ticker
            )
            self._benchmark_toolkit = Toolkit(
                api_key=self._api_key,
                tickers=list(set(self._benchmark_tickers.values())),
                historical=self._daily_benchmark_data,
                benchmark_ticker=None,
                start_date=self._start_date,
                use_cached_data=self._use_cached_data,
                rounding=None,
            )

        benchmark_name = self._benchmark_tickers[self._original_tickers[0]]
        benchmark_currency, ticker_currencies = self._benchmark_currencies(
            benchmark_name
        )

        # The benchmark is bought with the money of each transaction, so its prices are expressed in the currency of the transactions; otherwise a transaction in euros would buy units priced in dollars.  # noqa: E501
        conversions = (
            {
                currency
                for currency in ticker_currencies.values()
                if portfolio_model.get_price_unit(currency)
                != portfolio_model.get_price_unit(benchmark_currency)
            }
            if benchmark_currency
            else set()
        )
        exchange_rates = self._get_exchange_rates(
            [
                portfolio_model.get_exchange_rate_ticker(
                    currency, str(benchmark_currency)
                )
                for currency in conversions
            ],
            self._start_date,
        )

        def convert(data: pd.DataFrame, currency: str | None) -> pd.DataFrame:
            if data.empty or currency not in conversions:
                return data
            data = data.copy()
            data.columns = pd.MultiIndex.from_product([data.columns, [benchmark_name]])
            data = portfolio_model.convert_historical_data(
                data,
                {benchmark_name: (str(benchmark_currency), str(currency))},
                exchange_rates,
                self._cfg["adjustments"]["currency_adjustment_columns"],
            )
            return data.droplevel(1, axis=1)

        def benchmark_period(period: str) -> pd.DataFrame:
            data = self._benchmark_toolkit.get_historical_data(period=period)  # type: ignore[union-attr]
            # A single benchmark can come back with the ticker as a second column level.
            if data.columns.nlevels > 1:
                data = data.droplevel(1, axis=1)
            return data

        if ticker_currencies and not benchmark_currency:
            logger.warning(
                "Could not determine the currency of the benchmark %s, so its prices are "
                "left unconverted and assumed to be in the currency of the transactions.",
                benchmark_name,
            )

        main_currency = (
            pd.Series(list(ticker_currencies.values())).mode().iloc[0]
            if ticker_currencies
            else None
        )

        daily_benchmark_data = benchmark_period("daily")

        if daily_benchmark_data.empty:
            logger.error(
                "Failed to collect the historical data of the benchmark %s. Please ensure it is a "
                "valid ticker, or pass another one with benchmark_ticker.",
                benchmark_name,
            )
            return pd.DataFrame()

        # Aligned with carried-forward prices: filling a missing date with the next quote would copy that day's return onto the earlier date as well and price it with information from the future.  # noqa: E501
        aligned_by_currency = {
            main_currency: portfolio_model.align_benchmark_data(
                convert(daily_benchmark_data, main_currency),
                self._daily_historical_data.index,
            )
        }
        self._daily_benchmark_data = aligned_by_currency[main_currency]

        # Aggregated from the converted daily data, so each day is converted at its own rate.
        self._weekly_benchmark_data = (
            convert_daily_to_other_period(
                "weekly",
                convert(daily_benchmark_data, main_currency),
                start=self._benchmark_toolkit._start_date,  # type: ignore[union-attr]
                end=self._benchmark_toolkit._end_date,  # type: ignore[union-attr]
            ).loc[
                self._benchmark_toolkit._start_date : self._benchmark_toolkit._end_date
            ]  # type: ignore[union-attr]
            if main_currency in conversions
            else benchmark_period("weekly")
        )

        benchmark_closes = {}
        benchmark_dividends = {}
        for ticker in self._tickers:
            currency = ticker_currencies.get(ticker, main_currency)
            if currency not in aligned_by_currency:
                aligned_by_currency[currency] = portfolio_model.align_benchmark_data(
                    convert(daily_benchmark_data, currency),
                    self._daily_historical_data.index,
                )
            aligned = aligned_by_currency[currency]
            benchmark_closes[ticker] = aligned["Close"]
            benchmark_dividends[ticker] = (
                aligned["Dividends"].astype(float).fillna(0.0).cumsum()
                if "Dividends" in aligned
                else pd.Series(0.0, index=aligned.index)
            )

        self._benchmark_prices_per_ticker = pd.DataFrame(
            benchmark_closes, index=self._daily_benchmark_data.index
        )
        self._benchmark_dividends_per_ticker = pd.DataFrame(
            benchmark_dividends, index=self._daily_benchmark_data.index
        )

        # The benchmark is bought at the traded close, the same basis as the transactions, and receives its dividends as cash.  # noqa: E501
        self._benchmark_specific_prices = self._values_at_transactions(
            self._benchmark_prices_per_ticker
        )
        self._benchmark_transaction_dividends = self._values_at_transactions(
            self._benchmark_dividends_per_ticker
        )
        self._latest_benchmark_price = self._benchmark_prices_per_ticker.ffill().iloc[
            -1
        ]

        return self._daily_benchmark_data

    def _convert_transactions_to_reporting_currency(self) -> None:
        """
        Express every transaction in one reporting currency when they are made in more than
        one, so the invested amounts, values and weights of different holdings add up. The
        reporting currency is the `currency` of the Portfolio or else the currency most is
        invested in, with the amounts compared in dollars. Prices and costs are converted at
        the exchange rate of each transaction's day (or the last one before it), and a
        transaction without such a rate raises a ValueError. Prices in pence (GBp) and pounds
        without a `currency` are expressed in pounds. The historical data then follows the currency of the
        transactions.
        """
        currencies = self._portfolio_dataset[self._currency_column]
        known = currencies.dropna().astype(str)
        price_units = {portfolio_model.get_price_unit(currency) for currency in known}
        # Nothing to convert when every transaction is in the same unit as the reporting currency;
        # pence and pounds are one major currency but still differ by a factor of 100.
        if not price_units or (
            len(price_units) == 1
            and (
                self._currency is None
                or portfolio_model.get_price_unit(self._currency) in price_units
            )
        ):
            return
        majors = {major for major, _ in price_units}
        if currencies.isna().any():
            logger.warning(
                "The currency of %s is unknown, so their transactions are taken to be in the "
                "reporting currency. Please add a currency column to convert them.",
                ", ".join(
                    sorted(
                        map(
                            str,
                            self._portfolio_dataset.index.get_level_values(1)[
                                currencies.isna().to_numpy()
                            ].unique(),
                        )
                    )
                ),
            )
        dates = pd.PeriodIndex(
            self._portfolio_dataset.index.get_level_values(0), freq="D"
        )

        def conversion_factors(reporting_currency: str) -> pd.Series:
            # The factor that turns a price of each transaction into the reporting currency at the rate of its day.  # noqa: E501
            _, reporting_units = portfolio_model.get_price_unit(reporting_currency)
            rate_tickers = {
                currency: portfolio_model.get_exchange_rate_ticker(
                    reporting_currency, currency
                )
                for currency in set(known)
            }
            exchange_rates = self._get_exchange_rates(
                list(rate_tickers.values()), self._start_date
            )
            # A transaction without a known currency is taken to be in the reporting currency (see the warning above).  # noqa: E501
            factors = pd.Series(1.0, index=self._portfolio_dataset.index)
            for currency, rate_ticker in rate_tickers.items():
                rows = (currencies == currency).to_numpy()
                _, units = portfolio_model.get_price_unit(currency)
                rate: np.ndarray | float = 1.0
                if rate_ticker is not None:
                    # The rate of each transaction's day, or the last one before it.
                    rate = (
                        portfolio_model.align_exchange_rate(
                            exchange_rates[rate_ticker], dates[rows]
                        ).to_numpy(dtype=float)
                        if rate_ticker in exchange_rates.columns
                        else np.full(rows.sum(), np.nan)
                    )
                    if np.isnan(rate).any():
                        missing = dates[rows][np.isnan(rate)].min()
                        raise ValueError(
                            f"Transactions in {currency} cannot be converted to the reporting "
                            f"currency {reporting_currency} without the exchange rate "
                            f"{rate_ticker} on or before {missing}."
                        )
                factors[rows] = reporting_units / units / rate
            return factors

        if self._currency:
            reporting_currency = self._currency
        elif len(majors) == 1:
            reporting_currency = next(iter(majors))
        else:
            # The currency most is invested in, with every amount compared in dollars at the rate of its day: 1,000,000 yen is less than 50,000 dollars.  # noqa: E501
            invested = (
                self._portfolio_dataset[self._price_column]
                * self._portfolio_dataset[self._volume_column]
            ).abs()
            try:
                invested = invested * conversion_factors("USD")
            except ValueError:
                # Without the dollar rates, the amounts are compared in the major units of their own currency.  # noqa: E501
                invested = invested / [
                    (
                        portfolio_model.get_price_unit(currency)[1]
                        if isinstance(currency, str)
                        else 1
                    )
                    for currency in currencies
                ]
            major_currencies = [
                (
                    portfolio_model.get_price_unit(currency)[0]
                    if isinstance(currency, str)
                    else None
                )
                for currency in currencies
            ]
            totals = invested.groupby(major_currencies, sort=False).sum()
            reporting_currency = str(totals.idxmax())

        factors = conversion_factors(reporting_currency)

        for column in (self._price_column, self._costs_column):
            if column in self._portfolio_dataset.columns:
                self._portfolio_dataset[column] = (
                    self._portfolio_dataset[column] * factors
                )
        self._portfolio_dataset[self._currency_column] = reporting_currency

        logger.info(
            "The transactions are made in %s and are all expressed in %s.",
            ", ".join(sorted(majors)),
            reporting_currency,
        )

    def collect_historical_data(
        self, rounding: int | None = None, show_columns: str | list[str] | None = None
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

        Volatility, Excess Return and Excess Volatility are not included here. These are
        available via the Risk module (e.g. toolkit.risk.get_volatility) and the Performance
        module (e.g. toolkit.performance.get_excess_return) instead.

        This method retrieves historical price data (daily and weekly)
        for the portfolio's tickers and adjusts for any currency mismatches if necessary. It fetches
        data from a specified data source, applies currency conversion where applicable, and stores the
        adjusted data in separate DataFrames for the daily and weekly periods.

        The method uses the Toolkit class to fetch historical price data and the Currency Toolkit to
        handle currency conversions if the portfolio’s transaction currency does not match the historical
        data's currency.

        Args:
            rounding (int | None): An optional integer specifying the number of decimal places to round the
                historical price data. If None, the default rounding value is used.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the adjusted daily historical price data for the portfolio.

        Notes:
            - This method utilizes the `Toolkit` class to fetch historical price data for different periods.
            - Currency adjustments are made if there's a mismatch between the transaction and historical data currencies.
            - The method handles ISIN-to-ticker mapping when ISIN codes are provided in the portfolio data.
            - The adjusted historical data is stored in separate DataFrames for daily and weekly price data.
            - If any currency mismatch is found between the portfolio's transaction and historical data,
              a warning message is displayed.
            - The method rounds the data according to the specified or default rounding precision.
            - The latest closing price of every ticker, carried forward over a holiday of its exchange, is
            also captured and available in the `self._latest_price` attribute.
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
                tickers=[
                    self._data_tickers.get(ticker, ticker) for ticker in self._tickers
                ],
                benchmark_ticker=None,
                start_date=self._start_date,
                use_cached_data=self._use_cached_data,
                historical=self._daily_historical_data,
                rounding=None,
            )

        # Matches the data of each ticker (an ISIN's converted ticker) to its holding.
        self._ticker_combinations = {
            self._data_tickers.get(ticker, ticker): ticker for ticker in self._tickers
        }

        self._daily_historical_data = self._ticker_historical_data("daily")

        if self._daily_historical_data.empty and not self._api_key:
            logger.error(
                "Failed to collect historical data. Please ensure you have provided valid tickers. "
                "Yahoo Finance is unstable and has rate limits which you could have reached.\n"
                "Therefore, consider obtaining an API key with the following link: "
                "https://www.jeroenbouma.com/fmp\nYou can get 15% off by using the "
                "affiliate link which also supports the project."
            )
            return pd.DataFrame()

        # Per ticker the currency its historical data is quoted in and the currency of its
        # transactions, when the two differ.
        currency_conversions: dict[str, tuple[str, str]] = {}
        # Tickers traded in another currency than their data is quoted in.
        mismatched_tickers: set[str] = set()
        if self._currency_column:
            statistics = self._toolkit.get_historical_statistics()
            # A Toolkit of a single ticker returns its statistics as one Series.
            if isinstance(statistics, pd.Series):
                statistics = statistics.to_frame(self._toolkit._tickers[0])
            self._historical_statistics = statistics.rename(
                columns=self._ticker_combinations, level=0
            )

            # A built portfolio converted its prices with these currencies, so its historical
            # data is converted with the same ones.
            data_currencies: dict[str, str] = {}
            for ticker in self._tickers:
                data_currency = self._price_currencies.get(ticker) or (
                    self._historical_statistics.loc["Currency", ticker]
                    if not self._historical_statistics.empty
                    and "Currency" in self._historical_statistics.index
                    and ticker in self._historical_statistics.columns
                    else None
                )
                # A missing or NaN currency code would send a bogus 'NAN=X' and 404.
                if data_currency and not pd.isna(data_currency):
                    data_currencies[ticker] = str(data_currency)

            # A transaction without a currency (a dataset without a currency column) was made
            # in the currency its ticker is quoted in.
            transaction_currencies = self._portfolio_dataset[self._currency_column]
            missing = transaction_currencies.isna() | (
                transaction_currencies.astype(str).str.strip() == ""
            )
            self._portfolio_dataset.loc[missing, self._currency_column] = [
                data_currencies.get(ticker)
                for ticker in self._portfolio_dataset.index[missing].get_level_values(1)
            ]

            # The currency each ticker was traded in, before a reporting currency replaces it.
            traded_currencies = (
                self._portfolio_dataset[self._currency_column]
                .groupby(level=1, observed=True)
                .last()
                .to_dict()
            )
            self._convert_transactions_to_reporting_currency()

            for (_, ticker), currency in self._portfolio_dataset[
                self._currency_column
            ].items():
                data_currency = data_currencies.get(ticker)
                if not data_currency or not currency or pd.isna(currency):
                    continue

                traded_currency = traded_currencies.get(ticker)
                if isinstance(traded_currency, str) and portfolio_model.get_price_unit(
                    data_currency
                ) != portfolio_model.get_price_unit(traded_currency):
                    mismatched_tickers.add(ticker)

                # GBp and GBP are one currency in different units, so they are compared
                # by unit as well: a price in pence is a hundred times one in pounds.
                if portfolio_model.get_price_unit(
                    data_currency
                ) != portfolio_model.get_price_unit(str(currency)):
                    currency_conversions[ticker] = (data_currency, str(currency))

        exchange_rates = self._get_exchange_rates(
            [
                portfolio_model.get_exchange_rate_ticker(currency, data_currency)
                for data_currency, currency in currency_conversions.values()
            ],
            self._start_date,
        )

        def convert(historical_data: pd.DataFrame) -> pd.DataFrame:
            # Every price, dividend and return of the historical data is expressed in the
            # currency of the transactions, so the two can be compared and added up.
            if not currency_conversions or historical_data.empty:
                return historical_data
            return portfolio_model.convert_historical_data(
                historical_data,
                currency_conversions,
                exchange_rates,
                self._cfg["adjustments"]["currency_adjustment_columns"],
            )

        self._daily_historical_data = convert(self._daily_historical_data)

        # The weekly data is aggregated from the converted daily data, so each day's Open, High,
        # Low and Dividends is converted at that day's exchange rate rather than the week's last.
        self._weekly_historical_data = (
            convert_daily_to_other_period(
                "weekly",
                self._daily_historical_data,
                start=self._toolkit._start_date,  # type: ignore[union-attr]
                end=self._toolkit._end_date,  # type: ignore[union-attr]
            ).loc[
                self._toolkit._start_date : self._toolkit._end_date
            ]  # type: ignore[union-attr]
            if currency_conversions and not self._daily_historical_data.empty
            else self._ticker_historical_data("weekly")
        )

        # Positions are valued at the traded close, the basis of the transaction prices; a ticker whose exchange is closed on the last date keeps its previous close.  # noqa: E501
        self._latest_price = self._daily_historical_data["Close"].ffill().iloc[-1]

        if currency_conversions and self._definition is not None:
            logger.info(
                "Converted the prices of %s to the portfolio's currency.",
                ", ".join(
                    f"{ticker} ({data_currency} to {currency})"
                    for ticker, (
                        data_currency,
                        currency,
                    ) in currency_conversions.items()
                ),
            )
        elif mismatched_tickers:
            logger.warning(
                "Found a mismatch between the currency of the transaction and the currency of the historical data. "
                "This is usually due to working with ISIN codes.\nCorrect this by finding the correct ticker "
                "on for example Yahoo Finance (e.g. S&P 500 ETF can be VUSA.AS). The currencies are "
                "automatically converted but this can lead to some inaccuracies."
            )

        # Only the returned copy is rounded: the positions, returns and weights are calculated
        # from the stored prices, which rounding them would bias.
        return apply_rounding(
            self._daily_historical_data,
            rounding if rounding is not None else self._rounding,
        )

    def get_positions_overview(
        self, rounding: int | None = None, show_columns: str | list[str] | None = None
    ):
        """
        Calculate and provide an overview of the portfolio's positions, including key statistics and performance metrics.

        The following columns are included:

        - Volume: the net volume of each asset over time, i.e. every buy minus every sell.
        - Costs: the cumulative transaction costs of each asset over time.
        - Invested Amount: the cumulative capital deployed over time, i.e. the value of every buy plus
        the absolute transaction costs. Sale proceeds are not netted off.
        - Realized Proceeds: the cumulative cash received from selling units of the asset over time.
        - Dividends: the cumulative cash dividends received over time, i.e. on every ex-dividend date the
        units held at the previous close times the dividend per unit.
        - Current Value: the market value of the position still held, marked at the closing price of that day.
        - Cumulative Return: the total return on the capital deployed, i.e.
        (Current Value + Realized Proceeds + Dividends - Invested Amount) / Invested Amount. It is NaN
        before the first purchase, when no capital has been deployed yet.
        - Invested Weight: the weight of the asset in the portfolio based on the invested amount over time.
        - Current Weight: the weight of the asset in the portfolio based on the current value over time.

        Positions are marked at the traded closing price, the basis of the transaction prices, and the
        dividends are added as cash when they are paid, so the cumulative return is a total return and
        "Current Value" is the market value quoted on that date.

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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        if not self._collect_data():
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
        show_columns: str | list[str] | None = None,
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
        - Latest Price: The latest closing price of the asset obtained from historical data.
        - Latest Value: The market value of the position still held, i.e. Volume times Latest Price.
        - Dividends: The cash dividends received on the units held on every ex-dividend date.
        - Return: The total return on the capital deployed, i.e. Return Value divided by Invested.
        This covers realized and unrealized results and the dividends together and is NaN when nothing
        was invested. The "Portfolio" row divides by the capital contributed to the portfolio instead:
        the peak of the cumulative net contribution, i.e. every buy plus the costs minus the sale
        proceeds and the dividends received up to that date, summed over the assets per date. The buys
        of a rebalancing are paid for by the sells and dividends of the portfolio itself, so a portfolio
        that invests 10,000 once has 10,000 of capital however often it is rebalanced, while its
        "Invested" adds up every rebalancing buy. A portfolio of buys has its "Invested" as its capital,
        less the part of a later buy that a dividend received before it paid for.
        - Return Value: The absolute profit or loss, i.e. Latest Value plus all sale proceeds plus
        Dividends minus Invested. This equals realized PnL plus unrealized PnL plus dividends minus the
        transaction costs, where the realized PnL is the figure reported by get_transactions_overview.
        - Benchmark Return: The return the identical cash flows would have produced in the benchmark.
        Every transaction buys or sells benchmark units at the benchmark's closing price for the exact
        cash amount of that transaction on that date, so the comparison is matched in money rather than
        in share count, and the benchmark's dividends are received as cash as well. The "Portfolio"
        row divides by the same capital as its Return, so its Alpha compares like with like.
        - Volatility: The annualized volatility of the asset's daily returns over the most recent 252
        trading days. For the aggregated "Portfolio" row, this is derived from the full covariance matrix
        of the underlying asset returns over the same days (Var_p = w^T * Cov * w, Markowitz, 1952)
        rather than a weighted average of individual volatilities, since the latter ignores diversification
        from imperfectly correlated assets.
        - Benchmark Volatility: The annualized volatility of the benchmark's daily returns over the most
        recent 252 trading days.
        - Alpha: The alpha is based on the difference between the asset's return and the benchmark return.
        - Beta: The beta of the asset's daily returns against the benchmark's over the most recent 252
        trading days, the window of the Volatility column. A beta >1 indicates that the asset is more
        volatile than the benchmark and a beta <1 indicates that the asset is less volatile than the benchmark.
        - Weight: The weight of the asset in the portfolio based on the latest market value and the total
        market value of the portfolio. With short positions the total is the net value, so a weight can
        be negative or above 1.

        Positions are valued at the traded closing price, the basis of the transaction prices, and dividends
        are added as cash, for the asset and the benchmark alike, so Return and Benchmark Return are total
        returns on the same basis.

        No inventory method (FIFO, LIFO or average cost) is applied here: "Return" measures the result of
        every unit of currency put into the position rather than the basis of the units that happen to
        remain. The inventory methods drive the realized PnL in get_transactions_overview instead.

        The "Portfolio" row sums or averages units of different assets for "Volume", "Price" and
        "Latest Price", so those three carry no economic meaning at the portfolio level.

        Units held of an asset without a price, such as a ticker without data, are worth an unknown
        amount rather than nothing. The value of the portfolio is then unknown, so every "Weight" and the
        "Portfolio" row's value, return, alpha, volatility and beta are NaN; its "Invested", "Costs" and
        "Dividends" remain.

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
            exclude_sold_positions (bool): A flag indicating whether to exclude sold positions from the overview,
                those whose sales add up to their purchases. Open positions stay, long or short, and so does the
                Portfolio row while any position is open. Defaults to True.
            rounding (int | None): An optional integer specifying the number of decimal places to round the data.
                If None, the default rounding precision is used.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        | Identifier   |   Volume |   Costs |    Price |   Invested |   Latest Price |   Latest Value |   Dividends |   Return |   Return Value |   Benchmark Return |   Volatility |   Benchmark Volatility |   Alpha |    Beta |   Weight |
        |:-------------|---------:|--------:|---------:|-----------:|---------------:|---------------:|------------:|---------:|---------------:|-------------------:|-------------:|-----------------------:|--------:|--------:|---------:|
        | MPWR         |      116 |     -27 | 250.893  |  30636     |       1314     |      152424    |   3301.39   |   4.0978 |     125539     |             0.7169 |       0.5874 |                 0.1374 |  3.3809 |  2.6395 |   0.163  |
        | MSFT         |      105 |     -11 |  50.1726 |   5529.99  |        517.99  |       54388.9  |   3367.52   |   9.4709 |      52374.2   |             3.2604 |       0.3393 |                 0.1374 |  6.2105 |  1.0243 |   0.0582 |
        | NFLX         |      114 |     -32 |  13.0791 |   1666.89  |         60.75  |        6925.5  |      0      |   3.2816 |       5470.13  |             1.3201 |       0.3632 |                 0.1374 |  1.9616 |  0.292  |   0.0074 |
        | NVDA         |       69 |     -27 |   2.1717 |    202.909 |        156.14  |       10773.7  |     49.6373 |  52.4962 |      10651.9   |             1.004  |       0.388  |                 0.1374 | 51.4922 |  1.7966 |   0.0115 |
        | OXY          |       27 |     -15 |  52.3833 |   2005.56  |         57.36  |        1548.72 |    876.061  |   0.4264 |        855.115 |             2.6148 |       0.3558 |                 0.1374 | -2.1884 | -0.6839 |   0.0017 |
        | SKY          |      126 |     -23 |  24.4134 |   3343.23  |         92.02  |       11594.5  |    901.062  |   2.8171 |       9418.34  |             4.6739 |       0.4394 |                 0.1374 | -1.8567 |  1.1355 |   0.0124 |
        | VOO          |       77 |     -12 | 256.027  |  20238.2   |        680.26  |       52380    |   3925.88   |   1.8004 |      36436.5   |             1.606  |       0.1368 |                 0.1374 |  0.1944 |  0.994  |   0.056  |
        | VSS          |       98 |     -21 |  94.9125 |  10366.5   |        144.89  |       14199.2  |   3069.98   |   0.7621 |       7900.4   |             2.3488 |       0.175  |                 0.1374 | -1.5867 |  1.0539 |   0.0152 |
        | WMT          |       92 |     -18 |  22.2477 |   2265.02  |        128.41  |       11813.7  |    965.737  |   4.716  |      10681.8   |             3.2913 |       0.2686 |                 0.1374 |  1.4248 | -0.06   |   0.0126 |
        | Portfolio    |     2142 |    -532 |  55.8595 | 134818     |        436.564 |      935121    |  31003.8    |   8.0831 |     839982     |             2.3122 |       0.2971 |                 0.1374 |  5.7709 |  1.7916 |   1      |
        """
        if not self._collect_data():
            return pd.DataFrame()

        # A trailing year of 252 trading days for the volatility and the beta alike, rather than the current calendar year, which holds only a few returns early in January.  # noqa: E501
        if self._portfolio_volatilities.empty:
            asset_volatility = overview_model.calculate_trailing_volatility(
                self._daily_historical_data["Return"]
            )
            benchmark_volatility = overview_model.calculate_trailing_volatility(
                self._daily_benchmark_data["Return"]
            )

            self._portfolio_volatilities = pd.concat(
                [
                    asset_volatility,
                    pd.Series([benchmark_volatility], index=["Benchmark"]),
                ]
            )

        if self._portfolio_beta.empty:
            self._portfolio_beta = overview_model.calculate_trailing_beta(
                self._daily_historical_data["Return"],
                self._daily_benchmark_data["Return"],
            )

        cumulative_dividends = self._cumulative_dividends()

        # The Portfolio row measures its return against the capital put into the portfolio, which needs the dates of every buy, sale and dividend that the positions overview holds.  # noqa: E501
        contributed_capital = None
        if include_portfolio:
            if self._positions_overview.empty:
                self.get_positions_overview()
            contributed_capital = overview_model.calculate_contributed_capital(
                self._positions_overview
            )

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
                # The dividends per unit paid after every transaction up to the latest date.
                dividends=self._dividends_since_transactions(cumulative_dividends),
                benchmark_dividends=self._dividends_since_transactions(
                    self._benchmark_dividends_per_ticker
                ),
                contributed_capital=contributed_capital,
            )
        except ValueError as error:
            raise ValueError(f"Failed to create portfolio overview: {error}") from error

        if exclude_sold_positions:
            open_positions = self._open_positions(self._portfolio_overview["Volume"])
            # The Portfolio row stays as long as any of its positions is open; its volume adds up units of different assets, long and short, and says nothing about that.  # noqa: E501
            if "Portfolio" in open_positions.index:
                open_positions["Portfolio"] = open_positions.drop("Portfolio").any()
            self._portfolio_overview = self._portfolio_overview[open_positions]

        self._portfolio_overview = self._portfolio_overview.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        return self._portfolio_overview

    def get_portfolio_performance(
        self,
        period: str | None = None,
        exclude_sold_positions: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
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
            exclude_sold_positions (bool): A flag indicating whether to exclude the periods in which a position was
                sold, its sales adding up to its purchases. Open positions stay, long or short. Defaults to True.
            rounding (int | None): The number of decimal places to round the output to.
                If None, it defaults to the rounding precision specified in the configuration.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        |                                                    |   Volume |   Costs |   Invested Amount |   Realized Proceeds |   Dividends |   Current Value |   Invested Weight |   Current Weight |   Return |
        |:---------------------------------------------------|---------:|--------:|------------------:|--------------------:|------------:|----------------:|------------------:|-----------------:|---------:|
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'META') |       15 |      -1 |          4813.61  |               0     |     81.125  |        10780    |            0.0357 |           0.0099 |   1.2563 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'MPWR') |      116 |     -27 |         30636     |             449.4   |   3301.39   |       160810    |            0.2272 |           0.147  |   4.3715 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'MSFT') |      105 |     -11 |          5529.99  |             147.7   |   3367.52   |        56182.4  |            0.041  |           0.0513 |   9.7952 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'NFLX') |      114 |     -32 |          1666.89  |             211.515 |      0      |         8014.2  |            0.0124 |           0.0073 |   3.9348 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'NVDA') |       69 |     -27 |           202.909 |              31.547 |     49.6373 |        15820.3  |            0.0015 |           0.0145 |  77.3678 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'OXY')  |       27 |     -15 |          2005.56  |             435.898 |    876.061  |         1622.97 |            0.0149 |           0.0015 |   0.4634 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'SKY')  |      126 |     -23 |          3343.23  |             265.99  |    901.062  |        10033.4  |            0.0248 |           0.0092 |   2.3502 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'VOO')  |       77 |     -12 |         20238.2   |             368.76  |   3925.88   |        55100.4  |            0.1501 |           0.0504 |   1.9348 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'VSS')  |       98 |     -21 |         10366.5   |             997.66  |   3069.98   |        15201.8  |            0.0769 |           0.0139 |   0.8588 |
        | (Period('2026-10-05/2026-10-11', 'W-SUN'), 'WMT')  |       92 |     -18 |          2265.02  |             167.39  |    965.737  |        10244.2  |            0.0168 |           0.0094 |   4.0231 |
        """
        if not period:
            period = "quarterly" if self._quarterly else "yearly"

        if not self._collect_data():
            return pd.DataFrame()

        if self._positions_overview.empty:
            try:
                self.get_positions_overview()
            except ValueError as error:
                raise ValueError(
                    f"Failed to get positions overview: {error}"
                ) from error

        period_symbol = _PERIOD_SYMBOLS.get(period.lower())
        if period_symbol is None:
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
                self._open_positions(self._portfolio_performance["Volume"])
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
        show_columns: str | list[str] | None = None,
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
            exclude_sold_positions (bool): Whether to leave out the sell transactions. Profit & loss
                is realized when shares are sold, so it is booked on the sell transactions: with
                True, every remaining (buy) transaction shows a PnL of 0 and the realized total is
                in the Cumulative PnL column. Pass False to see the PnL of each sale. Defaults to True.
            pnl_method (str): The method for calculating profit & loss. Options are:
                'FIFO' (First In, First Out), 'LIFO' (Last In, First Out), or 'AVERAGE'.
                Defaults to 'FIFO'.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
              Return Value = realized PnL + unrealized PnL + dividends - transaction costs therefore ties
              this output back to get_portfolio_overview exactly.
            - "Invested Amount" is per transaction and is negative on a sell, since it is the cash
              movement of that one transaction rather than the capital deployed in the position.
            - "Current Value" values the units of a buy at the latest closing price and "Dividends" holds
              the cash dividends paid on them since the transaction, so "% Return" and "Return" are total
              returns of the units bought. A sell has no current value, dividends or return; its result
              is the PnL.
            - If no rounding is provided, the rounding precision specified in the configuration is used.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_transactions_overview()
        ```

        Which returns:

        |                                      | Name   |   Price |   Volume |   Costs | Currency   |   Invested Amount |   Current Value |   Dividends |   % Return |    Return |   PnL |   Cumulative PnL |
        |:-------------------------------------|:-------|--------:|---------:|--------:|:-----------|------------------:|----------------:|------------:|-----------:|----------:|------:|-----------------:|
        | (Period('2024-02-01', 'D'), 'GOOGL') | GOOGL  |  141.16 |       14 |       0 | USD        |           1976.24 |         4482.94 |     29.12   |     1.2832 | 2535.82   |     0 |           0      |
        | (Period('2024-02-12', 'D'), 'BAC')   | BAC    |   33.62 |        5 |      -3 | USD        |            171.1  |          252.8  |     14.8    |     0.564  |   96.5    |     0 |          22.7994 |
        | (Period('2024-02-22', 'D'), 'MCHI')  | MCHI   |   39.77 |        7 |      -1 | USD        |            279.39 |          343    |     18.9952 |     0.2957 |   82.6052 |     0 |         -19.26   |
        | (Period('2024-03-12', 'D'), 'MPWR')  | MPWR   |  736.55 |       11 |       0 | USD        |           8102.05 |        12541.3  |    189.64   |     0.5713 | 4628.91   |     0 |         399.48   |
        | (Period('2024-05-14', 'D'), 'CAMT')  | CAMT   |   94.11 |        4 |       0 | USD        |            376.44 |          484.64 |      0      |     0.2874 |  108.2    |     0 |         102.7    |
        | (Period('2024-06-11', 'D'), 'META')  | META   |  507.47 |        8 |       0 | USD        |           4059.76 |         5140.96 |     41.4    |     0.2765 | 1122.6    |     0 |           0      |
        | (Period('2024-06-18', 'D'), 'MPWR')  | MPWR   |  852.56 |       14 |      -1 | USD        |          11936.8  |        15961.7  |    223.86   |     0.3559 | 4248.7    |     0 |         399.48   |
        | (Period('2024-07-30', 'D'), 'AMD')   | AMD    |  138.44 |        3 |       0 | USD        |            415.32 |         1550.37 |      0      |     2.733  | 1135.05   |     0 |          13.3825 |
        | (Period('2024-10-25', 'D'), 'MCHI')  | MCHI   |   49.85 |        6 |       0 | USD        |            299.1  |          294    |     15.138  |     0.0336 |   10.038  |     0 |         -19.26   |
        | (Period('2024-11-13', 'D'), 'VOO')   | VOO    |  549.03 |       11 |       0 | USD        |           6039.33 |         7684.38 |    159.099  |     0.2987 | 1804.15   |     0 |         140.84   |
        """
        pnl_method = pnl_method.upper()

        if pnl_method not in ["FIFO", "LIFO", "AVERAGE"]:
            raise ValueError(
                "Please provide a valid method. This can be 'FIFO', 'LIFO', or 'AVERAGE'"
            )

        if not self._collect_data():
            return pd.DataFrame()

        try:
            new_columns = overview_model.create_transactions_overview(
                portfolio_volume=self._portfolio_dataset[self._volume_column],
                portfolio_price=self._portfolio_dataset[self._price_column],
                portfolio_costs=self._portfolio_dataset[self._costs_column],
                latest_returns=self._latest_price.loc[self._tickers],
                dividends=self._dividends_since_transactions(
                    self._cumulative_dividends()
                ),
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate transaction performance metrics for a specified period.

        This method calculates per asset and period how the units bought in that period performed
        up to the end of the period, next to the return the identical purchases would have made in
        the benchmark. The following columns are included:

        - Volume: the net volume traded in the period, i.e. buys minus sells.
        - Price: the volume-weighted average price paid for the units bought in the period.
        - Costs: the transaction costs of the period, of buys and sells.
        - Invested Amount: the value of the units bought plus the costs of those buys.
        - Realized Proceeds: the cash received from sales in the period.
        - Current Value: the units bought in the period at the closing price at the end of the period.
        - Dividends: the cash dividends those units received up to the end of the period.
        - Return: (Current Value + Dividends - Invested Amount) / Invested Amount, NaN in a period
        without purchases.
        - Benchmark Return: the same return for benchmark units bought with the same money on the
        same dates, dividends included.
        - Alpha: Return minus Benchmark Return.

        Sales are reported but left out of the return: the units sold in a period were usually bought
        in an earlier one, and their proceeds would credit this period's purchases with the gains of
        earlier ones. Units are valued at the traded closing price and receive their dividends as cash.

        Args:
            period (str | None): The time period for which transaction performance metrics
                should be calculated. This can be one of the following: 'yearly', 'quarterly',
                'monthly', 'weekly', or 'daily'. If None, the default is 'quarterly' if
                the 'quarterly' attribute is set to True, otherwise 'yearly'.
            exclude_sold_positions (bool): Whether to leave out the periods without an
                invested amount, those in which only units bought earlier were sold, and those whose only
                purchases cost nothing (a price of 0), which have no return to measure. Defaults
                to True.
            rounding (int | None): The number of decimal places to round the output to.
                If None, it defaults to the rounding specified in the configuration.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            - The end-of-period prices and dividends are taken from the daily data, in the currency of
              each ticker's transactions for the asset and the benchmark alike.
            - A transaction on a day without a price, such as a weekend, belongs to the period of the
              next trading day, the day it is priced on and receives dividends from; a transaction after
              the last date belongs to the period of the last date.
            - If no rounding is specified, the rounding precision from the configuration is used.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_transactions_performance(period='quarterly')
        ```

        Which returns:

        |                                      |   Volume |   Price |   Costs |   Invested Amount |   Realized Proceeds |   Current Value |   Dividends |   Return |   Benchmark Return |   Alpha |
        |:-------------------------------------|---------:|--------:|--------:|------------------:|--------------------:|----------------:|------------:|---------:|-------------------:|--------:|
        | (Period('2023Q4', 'Q-DEC'), 'MCHI')  |       15 |   43.29 |      -3 |            652.35 |                   0 |          611.1  |     14.912  |  -0.0404 |             0.0856 | -0.1259 |
        | (Period('2024Q1', 'Q-DEC'), 'BAC')   |        5 |   33.62 |      -3 |            171.1  |                   0 |          189.6  |      1.2    |   0.1151 |             0.028  |  0.0872 |
        | (Period('2024Q1', 'Q-DEC'), 'GOOGL') |       14 |  141.16 |       0 |           1976.24 |                   0 |         2113.02 |      0      |   0.0692 |             0.071  | -0.0018 |
        | (Period('2024Q1', 'Q-DEC'), 'MCHI')  |        7 |   39.77 |      -1 |            279.39 |                   0 |          278.11 |      0      |  -0.0046 |             0.0292 | -0.0338 |
        | (Period('2024Q1', 'Q-DEC'), 'MPWR')  |       11 |  736.55 |       0 |           8102.05 |                   0 |         7451.62 |     13.75   |  -0.0786 |             0.0153 | -0.0939 |
        | (Period('2024Q2', 'Q-DEC'), 'CAMT')  |        4 |   94.11 |       0 |            376.44 |                   0 |          500.96 |      0      |   0.3308 |             0.0407 |  0.29   |
        | (Period('2024Q2', 'Q-DEC'), 'META')  |        8 |  507.47 |       0 |           4059.76 |                   0 |         4033.76 |      4      |  -0.0054 |             0.0158 | -0.0213 |
        | (Period('2024Q2', 'Q-DEC'), 'MPWR')  |       14 |  852.56 |      -1 |          11936.8  |                   0 |        11503.5  |     17.5    |  -0.0348 |            -0.0049 | -0.0299 |
        | (Period('2024Q4', 'Q-DEC'), 'MCHI')  |        6 |   49.85 |       0 |            299.1  |                   0 |          281.16 |      5.3464 |  -0.0421 |             0.0127 | -0.0548 |
        | (Period('2024Q4', 'Q-DEC'), 'VOO')   |       11 |  549.03 |       0 |           6039.33 |                   0 |         5926.91 |     19.1235 |  -0.0154 |            -0.0173 |  0.0019 |
        """
        if not period:
            period = "quarterly" if self._quarterly else "yearly"

        if not self._collect_data():
            return pd.DataFrame()

        period_symbol = _PERIOD_SYMBOLS.get(period.lower())
        if period_symbol is None:
            raise ValueError(
                "Please provide a valid period. This can be 'yearly', 'quarterly', 'monthly', 'weekly', or 'daily'"
            )

        def end_of_period(daily_values: pd.DataFrame) -> pd.DataFrame:
            # The last value of every period, from the daily data so the asset and the benchmark are both in the currency of each ticker's transactions.  # noqa: E501
            return daily_values.groupby(
                to_period_index(daily_values.index).asfreq(period_symbol)
            ).last()

        cumulative_dividends = self._cumulative_dividends()

        # A transaction belongs to the period of the trading day it is priced on and its dividends count from, so a Saturday purchase at the end of a month is part of the next month, valued at that month's close.  # noqa: E501
        trading_dates = self._daily_historical_data.index[self._transaction_locations()]
        transactions = self._portfolio_dataset.set_axis(
            pd.MultiIndex.from_arrays(
                [
                    trading_dates,
                    self._portfolio_dataset.index.get_level_values(1),
                ],
                names=self._portfolio_dataset.index.names,
            )
        )

        try:
            self._transactions_performance = overview_model.create_transactions_performance(
                portfolio_dataset=transactions,
                ticker_column=self._ticker_column,
                date_column=self._date_column,
                volume_column=self._volume_column,
                price_column=self._price_column,
                costs_column=self._costs_column,
                period_prices=end_of_period(
                    self._daily_historical_data["Close"].reindex(columns=self._tickers)
                ),
                period_string=period_symbol,
                benchmark_specific_prices=self._benchmark_specific_prices,
                benchmark_period_prices=end_of_period(
                    self._benchmark_prices_per_ticker
                ),
                period_dividends=end_of_period(cumulative_dividends),
                transaction_dividends=self._values_at_transactions(
                    cumulative_dividends
                ),
                benchmark_period_dividends=end_of_period(
                    self._benchmark_dividends_per_ticker
                ),
                benchmark_transaction_dividends=self._benchmark_transaction_dividends,
            )
        except ValueError as error:
            raise ValueError(
                f"Failed to create transaction performance metrics: {error}"
            ) from error

        # The periods with purchases are kept: their return is measured on those units, also
        # when more was sold than bought in the same period.
        if exclude_sold_positions:
            self._transactions_performance = self._transactions_performance[
                self._transactions_performance["Invested Amount"] > 0
            ]

        self._transactions_performance = self._transactions_performance.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        return self._transactions_performance

    def get_money_weighted_return(
        self, rounding: int | None = None, show_columns: str | list[str] | None = None
    ) -> pd.DataFrame:
        """
        Calculate the money-weighted return of every position and of the whole portfolio:
        the annual rate at which the money put in grew into the money taken out plus the
        value still held (the internal rate of return, or XIRR, of the transactions).

        Unlike the time-weighted return, it depends on when money was added: buying more
        just before a rally raises it and buying at a peak lowers it. It therefore
        measures the investor's own result, including the timing of every purchase and
        sale, rather than the result of the assets alone.

        The cash flows are every purchase (money in, including costs), every sale (money
        out, after costs), every cash dividend received (money out, on its ex-dividend date)
        and the latest value of the positions still held, on the last date of the
        historical data.

        For more information, see the following book:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
          Wiley, 2nd edition, chapter 2.

        Also known as: internal rate of return, IRR, XIRR, dollar-weighted return.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Portfolio's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per position and for the Portfolio, the money Invested, the money
            Received from sales, the Dividends received, the Latest Value, the First
            Transaction date and the annualized Money-Weighted Return.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_money_weighted_return()
        ```

        Which returns:

        |           |   Invested |   Received |   Dividends |   Latest Value | First Transaction   |   Money-Weighted Return |
        |:----------|-----------:|-----------:|------------:|---------------:|:--------------------|------------------------:|
        | ASML      |   6348.35  |    736.149 |   8880.87   |      181476    | 2000-01-07          |                  0.2066 |
        | SKY       |   3339.23  |    261.99  |    901.062  |       11594.5  | 2000-01-10          |                  0.0718 |
        | AMZN      |   5042.91  |    192.851 |      0      |       29899    | 2000-01-28          |                  0.2079 |
        | FIX       |   1976.93  |     18.43  |   1416.06   |      129037    | 2000-02-23          |                  0.2456 |
        | MSFT      |   5529.99  |    147.7   |   3367.52   |       54388.9  | 2000-02-28          |                  0.1349 |
        | AMD       |   2384.36  |   1431.84  |      0      |       39333.8  | 2000-04-27          |                  0.1447 |
        | AAPL      |   5750.96  |    266.963 |   1283.5    |       48330.9  | 2000-05-08          |                  0.2783 |
        | FICO      |   4917.94  |    576.42  |    102.132  |       83204.5  | 2000-06-01          |                  0.1585 |
        | BAC       |   2401.58  |    303.8   |    888.08   |        4006.4  | 2000-11-16          |                  0.0665 |
        | ALGN      |   9918     |    324.77  |      0      |       13444.4  | 2001-09-18          |                  0.0419 |
        | CWST      |   1373.89  |     91.95  |      0      |        6465.06 | 2001-10-15          |                  0.1368 |
        | OXY       |   1994.56  |    424.898 |    876.061  |        1548.72 | 2001-11-16          |                  0.0352 |
        | WMT       |   2261.02  |    163.39  |    965.737  |       11813.7  | 2001-12-28          |                  0.132  |
        | CAMT      |    546.73  |    104.06  |     67.87   |        3916.5  | 2002-08-07          |                  0.2678 |
        | NVDA      |    201.303 |     29.941 |     49.6373 |       10773.7  | 2002-09-20          |                  0.3045 |
        | NFLX      |   1662.85  |    207.478 |      0      |        6925.5  | 2002-11-18          |                  0.1972 |
        | GOOGL     |   4668.1   |    359.26  |    304.08   |       41423.1  | 2004-08-24          |                  0.2094 |
        | MPWR      |  30633     |    446.4   |   3301.39   |      152424    | 2005-09-16          |                  0.2877 |
        | BLDR      |   1682.82  |    121.254 |      0      |        2887.2  | 2006-12-22          |                  0.1084 |
        | VSS       |  10364.5   |    995.66  |   3069.98   |       14199.2  | 2009-06-05          |                  0.0725 |
        | AVGO      |   1568.81  |    134.64  |    935.747  |       22004.4  | 2009-11-02          |                  0.375  |
        | MCHI      |   2965.43  |    139.96  |    327.316  |        2648.7  | 2011-04-26          |                  0.0088 |
        | VOO       |  20238.2   |    368.76  |   3925.88   |       52380    | 2011-12-20          |                  0.1475 |
        | META      |   4813.61  |      0     |     81.125  |        9396.15 | 2012-06-20          |                  0.1842 |
        | KHC       |    722.41  |    242.21  |    108.25   |         225.63 | 2017-03-07          |                 -0.0511 |
        | EMXC      |   1403.3   |    477.33  |    151.542  |        1373.76 | 2019-05-03          |                  0.079  |
        | Portfolio | 134711     |   8568.1   |  31003.8    |      935121    | 2000-01-07          |                  0.1728 |
        """
        # Imported here: the attribution model is only needed for this calculation.
        from financetoolkit.portfolio.construction.attribution import (  # noqa: PLC0415
            attribution_model,
        )

        if not self._collect_data():
            return pd.DataFrame()

        # Every transaction, sales included; the transactions overview leaves sold
        # positions out by default.
        transactions = self._portfolio_dataset.rename(
            columns={
                self._price_column: "Price",
                self._volume_column: "Volume",
                self._costs_column: "Costs",
            }
        )
        transactions.index = transactions.index.set_names(["Date", "Identifier"])

        if transactions.empty:
            raise ValueError("The portfolio has no transactions.")

        # From the investor's point of view: a purchase is money in (negative), a sale
        # money out (positive). Costs are always paid, whichever sign they are recorded
        # with, as the portfolio overview treats them.
        flows = (
            -transactions["Price"] * transactions["Volume"]
            - transactions["Costs"].abs()
        )
        trading_days = pd.to_datetime(
            self._daily_historical_data.index.get_level_values(0).astype(str)
        )
        last_date = trading_days.max()
        # A transaction on a day its ticker did not trade (a weekend or holiday, also when a cryptocurrency next to it did) flows on the ticker's next trading day, the day its position starts, as in the transactions performance, and one after the last date on the last close it is priced at.  # noqa: E501
        flow_dates = trading_days[self._transaction_locations()]

        # The dividends received per day follow from the cumulative dividends of the positions, dated on the ex-dividend date.  # noqa: E501
        positions_dividends = self.get_positions_overview()["Dividends"]
        daily_dividends = positions_dividends.diff().fillna(positions_dividends)
        daily_dividends.index = pd.to_datetime(
            to_period_index(daily_dividends.index).to_timestamp()
        )
        dividend_flows: dict[str, pd.Series] = {
            str(ticker): values[values != 0]
            for ticker, values in daily_dividends.items()
        }

        rows = {}
        latest_values: dict[str, float] = {}
        identifiers = transactions.index.get_level_values("Identifier")
        for identifier in list(dict.fromkeys(identifiers)) + ["Portfolio"]:
            selected = (
                flows if identifier == "Portfolio" else flows[identifiers == identifier]
            )
            dates = pd.to_datetime(selected.index.get_level_values("Date").astype(str))
            cash_flows = pd.Series(
                selected.to_numpy(),
                index=(
                    flow_dates
                    if identifier == "Portfolio"
                    else flow_dates[identifiers == identifier]
                ),
            )
            dividends = (
                pd.concat(list(dividend_flows.values()))
                if identifier == "Portfolio" and dividend_flows
                else dividend_flows.get(str(identifier), pd.Series(dtype=float))
            )
            # The value still held follows from the same transactions: their net volume at the latest close, unrounded, since a price below the rounding (such as 0.00001) would otherwise value the position at nothing.  # noqa: E501
            if identifier == "Portfolio":
                latest_value = float(sum(latest_values.values()))
            else:
                net_volume = float(
                    transactions["Volume"][identifiers == identifier].sum()
                )
                latest_value = net_volume * float(
                    self._latest_price.get(identifier, 0.0)
                )
                latest_values[identifier] = latest_value
            all_flows = pd.concat(
                [cash_flows, dividends, pd.Series([latest_value], index=[last_date])]
            )
            try:
                money_weighted = attribution_model.get_money_weighted_return(all_flows)
            except ValueError as error:
                logger.warning(
                    "No money-weighted return is available for %s: %s",
                    identifier,
                    error,
                )
                money_weighted = float("nan")

            rows[identifier] = {
                "Invested": float(-cash_flows[cash_flows < 0].sum()),
                "Received": float(cash_flows[cash_flows > 0].sum()),
                "Dividends": float(dividends.sum()),
                "Latest Value": latest_value,
                "First Transaction": dates.min().strftime("%Y-%m-%d"),
                "Money-Weighted Return": money_weighted,
            }

        result = pd.DataFrame(rows).T
        numeric = [
            "Invested",
            "Received",
            "Dividends",
            "Latest Value",
            "Money-Weighted Return",
        ]
        result[numeric] = result[numeric].astype(float)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def _get_position_values(self) -> pd.DataFrame:
        """The value of every position per day, in the currency of the portfolio."""
        positions = self.get_positions_overview(rounding=None)
        if positions.empty:
            raise ValueError("The Portfolio has no positions.")
        return positions["Current Value"].astype(float)

    def get_turnover(
        self,
        denominator: str = "gross_value",
        period: str = "daily",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the turnover of the portfolio, the value traded as a share of the value
        held: how much of the portfolio the transactions replace. A turnover of 0.1 on a
        day means that a tenth of the portfolio's value was bought or sold.

        - Traded Value = Σ |volume × price| of the day's transactions
        - Gross Value = Σ |value of every position|
        - Turnover = Traded Value / ((Gross Value_t + Gross Value_t-1) / 2)

        With denominator="net_value" the traded value is divided by the net value of the
        positions that day instead. Over weeks, months, quarters or years the traded
        values and the daily turnovers are added up and the value is that at the end of
        the period.

        For more information, see the following sources:

        - Bacon, C. R. (2008). "Practical Portfolio Performance Measurement and
          Attribution." 2nd edition, Wiley.
        - pyfolio: get_turnover (https://github.com/stefan-jansen/pyfolio-reloaded).

        Also known as: portfolio turnover, turnover rate, trading activity.

        Args:
            denominator (str, optional): "gross_value" (pyfolio's "AGB", the average of
                the gross value of the day and the day before) or "net_value" (pyfolio's
                "portfolio_value"). Defaults to "gross_value".
            period (str, optional): daily, weekly, monthly, quarterly or yearly. Defaults
                to "daily".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Portfolio's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per period the Traded Value, the Gross Value (or Net Value) and
            the Turnover.

        Notes:
        - Equals pyfolio's get_turnover on the daily positions; the positions hold no
          cash, so the net value is that of the positions, where pyfolio adds the cash.
        - The first purchase of the portfolio is a turnover of 2: the average gross value
          of that day and the empty day before is half the value bought.
        - A transaction on a day without prices counts on the next day that has one.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_turnover(period="yearly")
        ```

        Which returns:

        | Date   |   Traded Value |   Gross Value |   Turnover |
        |:-------|---------------:|--------------:|-----------:|
        | 2000   |       3657.08  |       2392.28 |     4.1243 |
        | 2001   |       1569.55  |       4798.28 |     0.4154 |
        | 2002   |       1693.63  |       5422.3  |     0.3135 |
        | 2003   |       1456.49  |       8204.56 |     0.2284 |
        | 2004   |       1518.38  |      10365.3  |     0.1683 |
        | 2005   |       1918.15  |      13539.3  |     0.1678 |
        | 2006   |       2377.66  |      16209    |     0.1559 |
        | 2007   |       3585.34  |      19071.4  |     0.2099 |
        | 2008   |       1123.91  |      12108.2  |     0.0745 |
        | 2009   |       3013.99  |      20985.5  |     0.1933 |
        | 2010   |       2621.67  |      25801.1  |     0.1206 |
        | 2011   |       3174.61  |      26933.1  |     0.1253 |
        | 2012   |       5141.61  |      34851.3  |     0.1546 |
        | 2013   |        522.848 |      49217.5  |     0.0121 |
        | 2014   |       3364.75  |      54772.7  |     0.0674 |
        | 2015   |       2003.66  |      61747.1  |     0.0349 |
        | 2016   |       2364.38  |      78378.2  |     0.0384 |
        | 2017   |       7271.19  |     117708    |     0.0743 |
        | 2018   |       6702.23  |     125233    |     0.0507 |
        | 2019   |       7866.05  |     206431    |     0.0445 |
        | 2020   |      15353.6   |     328089    |     0.0631 |
        | 2021   |       5121.04  |     435818    |     0.0137 |
        | 2022   |      18121.5   |     337393    |     0.0569 |
        | 2023   |       6215.22  |     547989    |     0.0142 |
        | 2024   |      35203.6   |     739785    |     0.051  |
        | 2025   |          0     |     908218    |     0      |
        | 2026   |          0     |     865290    |     0      |
        """
        if not self._collect_data():
            return pd.DataFrame()
        values = self._get_position_values()
        if self._transactions_overview.empty:
            self.get_transactions_overview()
        transactions = self._transactions_overview
        traded = (transactions["Volume"] * transactions["Price"]).abs().to_numpy()
        dates = to_period_index(transactions.index.get_level_values(0))
        # A transaction on a day without prices is booked on the next day with one.
        locations = values.index.get_indexer(dates, method="backfill")
        traded_values = (
            pd.Series(traded[locations >= 0])
            .groupby(values.index[locations[locations >= 0]])
            .sum()
        )
        table = overview_model.get_turnover(traded_values, values, denominator)
        if period != "daily":
            grouped = table.groupby(table.index.asfreq(self._period_frequency(period)))
            table = grouped.sum()
            table[table.columns[1]] = grouped[table.columns[1]].last()
        return apply_rounding(
            table, rounding if rounding is not None else self._rounding
        )

    def get_gross_leverage(
        self,
        period: str = "daily",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the gross leverage of the portfolio, the value of its long and short
        positions together as a multiple of its net value:

        - Gross Leverage = Σ |value of every position| / Σ value of every position

        A portfolio of long positions has a gross leverage of 1; short positions raise it,
        e.g. to 1.6 for a 130/30 portfolio. Over weeks, months, quarters or years the
        values at the end of the period are shown. The measure is defined in pyfolio
        (Quantopian) as gross_lev.

        For more information, see the following sources:

        - pyfolio: gross_lev (https://github.com/stefan-jansen/pyfolio-reloaded).

        Also known as: gross exposure, leverage ratio.

        Args:
            period (str, optional): daily, weekly, monthly, quarterly or yearly. Defaults
                to "daily".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Portfolio's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per period the Gross Value, Net Value and Gross Leverage, NaN when
            the net value is not positive.

        Notes:
        - Equals pyfolio's gross_lev without cash: the positions hold no cash, so the net
          value is that of the positions.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_gross_leverage(period="yearly")
        ```

        Which returns:

        | Date   |   Gross Value |   Net Value |   Gross Leverage |
        |:-------|--------------:|------------:|-----------------:|
        | 2000   |       2392.28 |     2392.28 |                1 |
        | 2001   |       4798.28 |     4798.28 |                1 |
        | 2002   |       5422.3  |     5422.3  |                1 |
        | 2003   |       8204.56 |     8204.56 |                1 |
        | 2004   |      10365.3  |    10365.3  |                1 |
        | 2005   |      13539.3  |    13539.3  |                1 |
        | 2006   |      16209    |    16209    |                1 |
        | 2007   |      19071.4  |    19071.4  |                1 |
        | 2008   |      12108.2  |    12108.2  |                1 |
        | 2009   |      20985.5  |    20985.5  |                1 |
        | 2010   |      25801.1  |    25801.1  |                1 |
        | 2011   |      26933.1  |    26933.1  |                1 |
        | 2012   |      34851.3  |    34851.3  |                1 |
        | 2013   |      49217.5  |    49217.5  |                1 |
        | 2014   |      54772.7  |    54772.7  |                1 |
        | 2015   |      61747.1  |    61747.1  |                1 |
        | 2016   |      78378.2  |    78378.2  |                1 |
        | 2017   |     117708    |   117708    |                1 |
        | 2018   |     125233    |   125233    |                1 |
        | 2019   |     206431    |   206431    |                1 |
        | 2020   |     328089    |   328089    |                1 |
        | 2021   |     435818    |   435818    |                1 |
        | 2022   |     337393    |   337393    |                1 |
        | 2023   |     547989    |   547989    |                1 |
        | 2024   |     739785    |   739785    |                1 |
        | 2025   |     908218    |   908218    |                1 |
        | 2026   |     865290    |   865290    |                1 |
        """
        if not self._collect_data():
            return pd.DataFrame()
        table = overview_model.get_gross_leverage(self._get_position_values())
        if period != "daily":
            table = table.groupby(
                table.index.asfreq(self._period_frequency(period))
            ).last()
        return apply_rounding(
            table, rounding if rounding is not None else self._rounding
        )

    def get_return_contributions(
        self,
        period: str = "monthly",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how much each position added to the portfolio's return in every period
        (week, month, quarter or year), so the contributions add up to the portfolio's
        return of that period. Every day a position contributes its weight in the
        portfolio at the previous close times its return, c_i,t = w_i,t-1 r_i,t, and over
        a longer period the daily contributions are carried by the growth of the portfolio
        since the period began, C_i = Σ_t c_i,t Π_(s<t) (1 + R_s), so they add up to the
        compounded return of the portfolio (Bacon, 2008; PerformanceAnalytics'
        to.period.contributions). It is `toolkit.portfolio.get_return_contributions` with
        the weights of the positions as they moved from day to day.

        The returns are the total returns of the assets (dividends reinvested) and the
        weights the Current Weight of the positions overview, so a purchase or a sale
        changes the weights from the next day on and is no return itself.

        For more information, see the following sources:

        - Bacon, C. R. (2008). "Practical Portfolio Performance Measurement and
          Attribution." 2nd edition, Wiley, chapter 5.
        - PerformanceAnalytics: to.period.contributions
          (https://github.com/braverock/PerformanceAnalytics).

        Also known as: contribution to return, return attribution by position.

        Args:
            period (str, optional): The period to add the contributions up over: daily,
                weekly, monthly, quarterly or yearly. Defaults to "monthly".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Portfolio's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per period the contribution of every position and the Portfolio
            return.

        As an example:

        ```python
        from financetoolkit import Portfolio

        portfolio = Portfolio(example=True, api_key="FINANCIAL_MODELING_PREP_KEY")

        portfolio.get_return_contributions(period="yearly")[
            ["ASML", "MSFT", "NVDA", "AMZN", "Portfolio"]
        ]
        ```

        Which returns:

        | Date   |    ASML |    MSFT |    NVDA |    AMZN |   Portfolio |
        |:-------|--------:|--------:|--------:|--------:|------------:|
        | 2000   |  0.0919 | -0.2261 |  0      | -0.0066 |     -0.3338 |
        | 2001   | -0.0367 |  0.1306 |  0      | -0.0005 |      0.4238 |
        | 2002   | -0.0308 | -0.0517 |  0      |  0.0004 |     -0.1472 |
        | 2003   |  0.1081 |  0.0148 |  0.0002 |  0.002  |      0.3674 |
        | 2004   | -0.0334 |  0.0131 |  0      |  0.0003 |      0.1162 |
        | 2005   |  0.0401 |  0      |  0.0002 |  0.0002 |      0.1154 |
        | 2006   |  0.0407 |  0.0222 |  0.0006 | -0.0003 |      0.0813 |
        | 2007   |  0.042  |  0.0235 |  0.0004 |  0.007  |      0.0048 |
        | 2008   | -0.0793 | -0.0579 | -0.0012 | -0.0049 |     -0.4002 |
        | 2009   |  0.1632 |  0.0634 |  0.0009 |  0.0144 |      0.4882 |
        | 2010   |  0.028  | -0.0073 | -0.0002 |  0.004  |      0.1288 |
        | 2011   |  0.0237 | -0.0038 | -0.0001 | -0.0004 |     -0.0446 |
        | 2012   |  0.097  |  0.0059 | -0.0001 |  0.0056 |      0.2122 |
        | 2013   |  0.1062 |  0.0274 |  0.0001 |  0.0091 |      0.4159 |
        | 2014   |  0.0363 |  0.018  |  0.0001 | -0.004  |      0.1042 |
        | 2015   | -0.0417 |  0.0158 |  0.0004 |  0.0202 |      0.1087 |
        | 2016   |  0.0488 |  0.0123 |  0.0018 |  0.0038 |      0.2379 |
        | 2017   |  0.0992 |  0.0287 |  0.0017 |  0.0199 |      0.4316 |
        | 2018   | -0.0159 |  0.0138 | -0.0006 |  0.0124 |      0.0286 |
        | 2019   |  0.1419 |  0.0426 |  0.0012 |  0.0127 |      0.592  |
        | 2020   |  0.1134 |  0.0316 |  0.0019 |  0.0353 |      0.513  |
        | 2021   |  0.1187 |  0.037  |  0.0034 |  0.0014 |      0.3203 |
        | 2022   | -0.0695 | -0.0226 | -0.0023 | -0.0218 |     -0.2619 |
        | 2023   |  0.0827 |  0.0431 |  0.0071 |  0.0233 |      0.6118 |
        | 2024   | -0.0108 |  0.0094 |  0.0106 |  0.014  |      0.2951 |
        | 2025   |  0.0672 |  0.0093 |  0.0049 |  0.0018 |      0.234  |
        | 2026   |  0.029  |  0.0005 | -0.0046 |  0.0021 |     -0.0438 |
        """
        if not self._collect_data():
            return pd.DataFrame()
        frequency = None if period == "daily" else self._period_frequency(period)
        positions = self.get_positions_overview(rounding=None)
        weights = positions["Current Weight"].astype(float)
        returns = (
            self._daily_historical_data["Return"]
            .reindex(index=weights.index, columns=weights.columns)
            .fillna(0)
        )
        # Every day a position earns its weight at the previous close; the first day has
        # no close before it.
        contributions = (weights.shift(1) * returns).iloc[1:]
        if frequency is None:
            table = contributions.assign(Portfolio=contributions.sum(axis=1))
        else:
            table = attribution_model.get_period_contributions(contributions, frequency)
        table.index.name = "Date"
        return apply_rounding(
            table, rounding if rounding is not None else self._rounding
        )

    @staticmethod
    def _period_frequency(period: str) -> str:
        """The pandas frequency of a period longer than a day."""
        frequencies = {"weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
        if period not in frequencies:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        return frequencies[period]
