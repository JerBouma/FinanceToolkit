"""Toolkit Module"""
__docformat__ = "google"


import re
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit.base.fundamentals_model import (
    get_analyst_estimates as _get_analyst_estimates,
    get_earnings_calendar as _get_earnings_calendar,
    get_financial_statements as _get_financial_statements,
    get_profile as _get_profile,
    get_quote as _get_quote,
    get_rating as _get_rating,
    get_revenue_segmentation as _get_revenue_segmentation,
)
from financetoolkit.base.helpers import calculate_growth as _calculate_growth
from financetoolkit.base.historical_model import (
    convert_daily_to_other_period as _convert_daily_to_other_period,
    get_historical_data as _get_historical_data,
)
from financetoolkit.base.models.models_controller import Models
from financetoolkit.base.normalization_model import (
    convert_date_label as _convert_date_label,
    convert_financial_statements as _convert_financial_statements,
    copy_normalization_files as _copy_normalization_files,
    read_normalization_file as _read_normalization_file,
)
from financetoolkit.base.performance.performance_controller import Performance
from financetoolkit.base.ratios.ratios_controller import Ratios
from financetoolkit.base.risk.risk_controller import Risk
from financetoolkit.base.technicals.technicals_controller import Technicals

# pylint: disable=too-many-instance-attributes,too-many-lines,line-too-long,too-many-locals,too-many-function-args
# ruff: noqa: E501


class Toolkit:
    """
    The Finance Toolkit is an open-source toolkit in which
    all 100+ financial ratios, indicators and performance measurements
    are written down in the most simplistic way allowing for complete transparency
    of the calculation method. This allows you to not have to rely on metrics
    from other providers and, given a financial statement, allow for efficient manual
    calculations. This leads to one uniform method of calculation being applied that
    is available and understood by everyone.
    """

    def __init__(
        self,
        tickers: list | str,
        api_key: str = "",
        start_date: str | None = None,
        end_date: str | None = None,
        quarterly: bool = False,
        risk_free_rate: str = "10y",
        benchmark_ticker: str | None = "^GSPC",
        custom_ratios: dict | None = None,
        historical: pd.DataFrame = pd.DataFrame(),
        balance: pd.DataFrame = pd.DataFrame(),
        income: pd.DataFrame = pd.DataFrame(),
        cash: pd.DataFrame = pd.DataFrame(),
        format_location: str = "",
        reverse_dates: bool = False,
        rounding: int | None = 4,
        remove_invalid_tickers: bool = True,
        sleep_timer: bool = False,
        progress_bar: bool = True,
    ):
        """
        Initializes an Toolkit object with a ticker or a list of tickers. The way the Toolkit is initialized
        will define how the data is collected. For example, if you enable the quarterly flag, you will
        be able to collect quarterly data. Next to that, you can define the start and end date to specify
        a specific range. Another option is to define the custom ratios you want to calculate.
        This can be done by passing a dictionary.

        See for more information on all of this, the following link: https://www.jeroenbouma.com/projects/financetoolkit

        Args:
        tickers (str or list): A string or a list of strings containing the company ticker(s). E.g. 'TSLA' or 'MSFT'
        Find the tickers on a variety of websites or via the FinanceDatabase: https://github.com/JerBouma/financedatabase
        api_key (str): An API key from FinancialModelingPrep. Obtain one here:
        https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/
        start_date (str): A string containing the start date of the data. This needs to be formatted as YYYY-MM-DD.
        end_date (str): A string containing the end date of the data. This needs to be formatted as YYYY-MM-DD.
        quarterly (bool): A boolean indicating whether to collect quarterly data. This defaults to False and thus
        collects yearly financial statements. Note that historical data can still be collected for
        any period and interval.
        risk_free_rate (str): A string containing the risk free rate. This can be 13w, 5y, 10y or 30y. This is
        based on the US Treasury Yields and is used to calculate various ratios and Excess Returns.
        benchmark_ticker (str): A string containing the benchmark ticker. Defaults to ^GSPC (S&P 500). This is
        meant to calculate ratios and indicators such as the CAPM and Jensen's Alpha but also serves as purpose to
        give insights in the performance of a stock compared to a benchmark.
        custom_ratios (dict): A dictionary containing custom ratios. This is meant to define your own ratios. See
        the following Notebook how to set this up: https://www.jeroenbouma.com/projects/financetoolkit/custom-ratios
        historical (pd.DataFrame): A DataFrame containing historical data. This is a custom dataset only relevant if
        you are looking to use custom data. See for more information the following Notebook:
        https://www.jeroenbouma.com/projects/financetoolkit/external-datasets
        balance (pd.DataFrame): A DataFrame containing balance sheet data. This is a custom dataset only
        relevant if you are looking to use custom data. See for more information the notebook as mentioned at historical.
        relevant if you are looking to use custom data. See for more information the notebook as mentioned at historical.
        cash (pd.DataFrame): A DataFrame containing cash flow statement data. This is a custom dataset only
        relevant if you are looking to use custom data. See for more information the notebook as mentioned at historical.
        format_location (str): A string containing the location of the normalization files.
        reverse_dates (bool): A boolean indicating whether to reverse the dates in the financial statements.
        rounding (int): An integer indicating the number of decimals to round the results to.
        remove_invalid_tickers (bool): A boolean indicating whether to remove invalid tickers.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to enable the progress bar when ticker amount is over 10. Defaults to True.

        As an example:

        ```python
        from financetoolkit import Toolkit

        # Simple example
        toolkit = Toolkit(["TSLA", "ASML"], api_key=FMP_KEY)

        # Obtaining quarterly data
        toolkit = Toolkit(["AAPL", "GOOGL"], quarterly=True, api_key=FMP_KEY)

        # Including a start and end date
        toolkit = Toolkit(["MSFT", "MU"], start_date="2020-01-01", end_date="2023-01-01", quarterly=True, api_key=FMP_KEY)

        # Changing the benchmark and risk free rate
        toolkit = Toolkit("AMZN", benchmark_ticker="^DJI", risk_free_rate="30y", api_key=FMP_KEY)
        ```
        """
        if isinstance(tickers, str):
            self._tickers = [tickers.upper()]
        elif isinstance(tickers, list):
            self._tickers = [ticker.upper() for ticker in tickers]
        else:
            raise TypeError("Tickers must be a string or a list of strings.")

        if start_date and re.match(r"^\d{4}-\d{2}-\d{2}$", start_date) is None:
            raise ValueError(
                "Please input a valid start date (%Y-%m-%d) like '2010-01-01'"
            )
        if end_date and re.match(r"^\d{4}-\d{2}-\d{2}$", end_date) is None:
            raise ValueError(
                "Please input a valid end date (%Y-%m-%d) like '2020-01-01'"
            )
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"Please ensure the start date {start_date} is before the end date {end_date}"
            )

        if risk_free_rate not in [
            "13w",
            "5y",
            "10y",
            "30y",
        ]:
            raise ValueError(
                "Please select a valid risk free rate (13w, 5y, 10y or 30y)"
            )

        self._api_key = api_key
        self._start_date = start_date
        self._end_date = end_date
        self._quarterly = quarterly
        self._risk_free_rate = risk_free_rate
        self._benchmark_ticker = benchmark_ticker
        self._rounding = rounding
        self._remove_invalid_tickers = remove_invalid_tickers
        self._invalid_tickers: list = []
        self._sleep_timer = sleep_timer
        self._progress_bar = progress_bar
        self._historical = historical

        if self._api_key:
            # Initialization of FinancialModelingPrep Variables
            self._profile: pd.DataFrame = pd.DataFrame()
            self._quote: pd.DataFrame = pd.DataFrame()
            self._rating: pd.DataFrame = pd.DataFrame()
            self._analyst_estimates: pd.DataFrame = pd.DataFrame()
            self._analyst_estimates_growth: pd.DataFrame = pd.DataFrame()
            self._earnings_calendar: pd.DataFrame = pd.DataFrame()
            self._revenue_geographic_segmentation: pd.DataFrame = pd.DataFrame()
            self._revenue_geographic_segmentation_growth: pd.DataFrame = pd.DataFrame()
            self._revenue_product_segmentation: pd.DataFrame = pd.DataFrame()
            self._revenue_product_segmentation_growth: pd.DataFrame = pd.DataFrame()

        # Initialization of Historical Variables
        self._daily_historical_data: pd.DataFrame = (
            historical if not historical.empty else pd.DataFrame()
        )
        self._weekly_historical_data: pd.DataFrame = (
            _convert_daily_to_other_period(
                "weekly",
                self._daily_historical_data,
                self._start_date,
                self._end_date,
            )
            if not historical.empty
            else pd.DataFrame()
        )
        self._monthly_historical_data: pd.DataFrame = (
            _convert_daily_to_other_period(
                "monthly",
                self._daily_historical_data,
                self._start_date,
                self._end_date,
            )
            if not historical.empty
            else pd.DataFrame()
        )
        self._quarterly_historical_data: pd.DataFrame = (
            _convert_daily_to_other_period(
                "quarterly",
                self._daily_historical_data,
                self._start_date,
                self._end_date,
            )
            if not historical.empty
            else pd.DataFrame()
        )
        self._yearly_historical_data: pd.DataFrame = (
            _convert_daily_to_other_period(
                "yearly", self._daily_historical_data, self._start_date, self._end_date
            )
            if not historical.empty
            else pd.DataFrame()
        )

        # Initialization of Risk Free Rate
        self._daily_risk_free_rate: pd.DataFrame = pd.DataFrame()
        self._weekly_risk_free_rate: pd.DataFrame = pd.DataFrame()
        self._monthly_risk_free_rate: pd.DataFrame = pd.DataFrame()
        self._quarterly_risk_free_rate: pd.DataFrame = pd.DataFrame()
        self._yearly_risk_free_rate: pd.DataFrame = pd.DataFrame()

        # Initialization of Treasury Variables
        self._daily_treasury_data: pd.DataFrame = pd.DataFrame()
        self._weekly_treasury_data: pd.DataFrame = pd.DataFrame()
        self._monthly_treasury_data: pd.DataFrame = pd.DataFrame()
        self._quarterly_treasury_data: pd.DataFrame = pd.DataFrame()
        self._yearly_treasury_data: pd.DataFrame = pd.DataFrame()

        # Initialization of Normalization Variables
        self._balance_sheet_statement_generic: pd.DataFrame = _read_normalization_file(
            "balance", format_location
        )
        self._income_statement_generic: pd.DataFrame = _read_normalization_file(
            "income", format_location
        )
        self._cash_flow_statement_generic: pd.DataFrame = _read_normalization_file(
            "cash", format_location
        )

        self._statistics_statement_generic: pd.DataFrame = _read_normalization_file(
            "statistics", format_location
        )

        # Initialization of Financial Statements
        self._balance_sheet_statement: pd.DataFrame = (
            _convert_date_label(
                _convert_financial_statements(
                    balance, self._balance_sheet_statement_generic, reverse_dates
                ),
                self._start_date,
                self._end_date,
                self._quarterly,
            )
            if not balance.empty
            else pd.DataFrame()
        )
        self._balance_sheet_statement_growth: pd.DataFrame = pd.DataFrame()

        self._income_statement: pd.DataFrame = (
            _convert_date_label(
                _convert_financial_statements(
                    income, self._income_statement_generic, reverse_dates
                ),
                self._start_date,
                self._end_date,
                self._quarterly,
            )
            if not income.empty
            else pd.DataFrame()
        )
        self._income_statement_growth: pd.DataFrame = pd.DataFrame()

        self._cash_flow_statement: pd.DataFrame = (
            _convert_date_label(
                _convert_financial_statements(
                    cash, self._cash_flow_statement_generic, reverse_dates
                ),
                self._start_date,
                self._end_date,
                self._quarterly,
            )
            if not cash.empty
            else pd.DataFrame()
        )
        self._cash_flow_statement_growth: pd.DataFrame = pd.DataFrame()

        self._statistics_statement: pd.DataFrame = pd.DataFrame()
        self._custom_ratios: dict | None = custom_ratios

        pd.set_option("display.float_format", str)

    @property
    def ratios(self) -> Ratios:
        """
        The Ratios Module contains over 50+ ratios that can be used to analyse companies. These ratios
        are divided into 5 categories which are efficiency, liquidity, profitability, solvency and
        valuation. Each ratio is calculated using the data from the Toolkit module.

        Some examples of ratios are the Current Ratio, Debt to Equity Ratio, Return on Assets (ROA),
        Return on Equity (ROE), Return on Invested Capital (ROIC), Return on Capital Employed (ROCE),
        Price to Earnings Ratio (P/E), Price to Book Ratio (P/B), Price to Sales Ratio (P/S), Price
        to Cash Flow Ratio (P/CF), Price to Free Cash Flow Ratio (P/FCF), Dividend Yield and
        Dividend Payout Ratio.

        Next to that, it is also possible to define custom ratios.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

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
        empty_data: list = []

        if not self._api_key and (
            self._balance_sheet_statement.empty
            or self._income_statement.empty
            or self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The ratios property requires manual addition of balance, income "
                "and cash flow statements or an API key from FinancialModelPrep "
                "within the Toolkit class."
            )
        if self._balance_sheet_statement.empty:
            empty_data.append("Balance Sheet Statement")
            self.get_balance_sheet_statement()
        if self._income_statement.empty:
            empty_data.append("Income Statement")
            self.get_income_statement()
        if self._cash_flow_statement.empty:
            empty_data.append("Cash Flow Statement")
            self.get_cash_flow_statement()

        if (
            self._balance_sheet_statement.empty
            and self._income_statement.empty
            and self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The datasets could not be populated and therefore the Ratios class cannot be initialized."
            )

        if not self._start_date:
            self._start_date = (
                f"{self._balance_sheet_statement.columns[0].year - 5}-01-01"
            )
        if not self._end_date:
            self._end_date = (
                f"{self._balance_sheet_statement.columns[-1].year + 5}-01-01"
            )

        if self._quarterly:
            self.get_historical_data(period="quarterly")
        else:
            self.get_historical_data(period="yearly")

        if empty_data:
            print(
                "The following data was not provided within the Toolkit class and "
                f"thus was retrieved from FinancialModelingPrep: {', '.join(empty_data)}."
            )

        return Ratios(
            tickers=self._tickers,
            historical=self._quarterly_historical_data
            if self._quarterly
            else self._yearly_historical_data,
            balance=self._balance_sheet_statement,
            income=self._income_statement,
            cash=self._cash_flow_statement,
            custom_ratios_dict=self._custom_ratios,
            quarterly=self._quarterly,
            rounding=self._rounding,
        )

    @property
    def models(self) -> Models:
        """
        Gives access to the Models module. The Models module is meant to execute well-known models
        such as DUPONT and the Discounted Cash Flow (DCF) model. These models are also directly
        related to the data retrieved from the Toolkit module.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/models

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AMZN"], api_key=FMP_KEY, quarterly=True, start_date='2022-12-31')

        dupont_analysis = toolkit.models.get_extended_dupont_analysis()

        dupont_analysis.loc['AMZN']
        ```

        Which returns:

        |                         |      2022Q2 |    2022Q3 |      2022Q4 |    2023Q1 |    2023Q2 |
        |:------------------------|------------:|----------:|------------:|----------:|----------:|
        | Interest Burden Ratio   |  -1.24465   | 0.858552  | -2.88409    | 1.20243   | 1.01681   |
        | Tax Burden Ratio        |  -0.611396  | 1.13743   |  0.101571   | 0.640291  | 0.878792  |
        | Operating Profit Margin |  -0.0219823 | 0.0231391 | -0.00636042 | 0.0323498 | 0.0562125 |
        | Asset Turnover          | nan         | 0.299735  |  0.3349     | 0.274759  | 0.285319  |
        | Equity Multiplier       | nan         | 3.15403   |  3.14263    | 3.08433   | 2.91521   |
        | Return on Equity        | nan         | 0.0213618 |  0.00196098 | 0.0211066 | 0.0417791 |
        """
        empty_data: list = []

        if not self._api_key and (
            self._balance_sheet_statement.empty
            or self._income_statement.empty
            or self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The model property requires manual addition of balance, income "
                "and cash flow statements or an API key from FinancialModelPrep "
                "within the Toolkit class."
            )

        if self._balance_sheet_statement.empty:
            empty_data.append("Balance Sheet Statement")
            self.get_balance_sheet_statement()
        if self._income_statement.empty:
            empty_data.append("Income Statement")
            self.get_income_statement()
        if self._cash_flow_statement.empty:
            empty_data.append("Cash Flow Statement")
            self.get_cash_flow_statement()

        if (
            self._balance_sheet_statement.empty
            and self._income_statement.empty
            and self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The datasets could not be populated and therefore the Models class cannot be initialized."
            )

        if not self._start_date:
            self._start_date = (
                f"{self._balance_sheet_statement.columns[0].year - 5}-01-01"
            )
        if not self._end_date:
            self._end_date = (
                f"{self._balance_sheet_statement.columns[-1].year + 5}-01-01"
            )

        if self._quarterly:
            self.get_historical_data(period="quarterly")
        else:
            self.get_historical_data(period="yearly")

        if empty_data:
            print(
                "The following data was not provided within the Toolkit class and "
                f"thus was retrieved from FinancialModelingPrep: {', '.join(empty_data)}."
            )

        return Models(
            tickers=self._tickers,
            daily_historical=self._daily_historical_data,
            period_historical=self._quarterly_historical_data
            if self._quarterly
            else self._yearly_historical_data,
            risk_free_rate=self._quarterly_risk_free_rate
            if self._quarterly
            else self._yearly_risk_free_rate,
            balance=self._balance_sheet_statement,
            income=self._income_statement,
            cash=self._cash_flow_statement,
            quarterly=self._quarterly,
            rounding=self._rounding,
        )

    @property
    def technicals(self) -> Technicals:
        """
        This gives access to the Technicals module. The Technicals Module contains
        nearly 50 Technical Indicators that can be used to analyse companies. These indicators are
        divided into 3 categories: breadth, overlap and volatility. Each indicator is calculated using
        the data from the Toolkit module.

        Some examples of technical indicators are the Average Directional Index (ADX), the
        Accumulation/Distribution Line (ADL), the Average True Range (ATR), the Bollinger Bands (BBANDS),
        the Commodity Channel Index (CCI), the Chaikin Oscillator (CHO), the Chaikin Money Flow (CMF),
        the Double Exponential Moving Average (DEMA), the Exponential Moving Average (EMA) and
        the Moving Average Convergence Divergence (MACD).

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/technicals

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        average_directional_index = toolkit.technical.get_average_directional_index()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |
        |:-----------|--------:|--------:|
        | 2023-08-21 | 62.8842 | 36.7468 |
        | 2023-08-22 | 65.7063 | 36.5525 |
        | 2023-08-23 | 67.3596 | 35.5149 |
        | 2023-08-24 | 66.4527 | 35.4399 |
        | 2023-08-25 | 63.4837 | 32.3323 |

        """
        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        if self._historical.empty:
            if self._daily_historical_data.empty:
                self.get_historical_data(period="daily")
            if self._weekly_historical_data.empty:
                self.get_historical_data(period="weekly")
            if self._monthly_historical_data.empty:
                self.get_historical_data(period="monthly")
            if self._quarterly_historical_data.empty:
                self.get_historical_data(period="quarterly")
            if self._yearly_historical_data.empty:
                self.get_historical_data(period="yearly")

        return Technicals(
            tickers=self._tickers,
            daily_historical=self._daily_historical_data.drop(
                "Benchmark", axis=1, level=1, errors="ignore"
            ),
            weekly_historical=self._weekly_historical_data.drop(
                "Benchmark", axis=1, level=1, errors="ignore"
            ),
            monthly_historical=self._monthly_historical_data.drop(
                "Benchmark", axis=1, level=1, errors="ignore"
            ),
            quarterly_historical=self._quarterly_historical_data.drop(
                "Benchmark", axis=1, level=1, errors="ignore"
            ),
            yearly_historical=self._yearly_historical_data.drop(
                "Benchmark", axis=1, level=1, errors="ignore"
            ),
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
        )

    @property
    def performance(self) -> Performance:
        """
        This gives access to the Performance module. The Performance Module is meant to calculate metrics related
        to the risk-return relationship. These are things such as Beta, Sharpe Ratio, Sortino Ratio, CAPM,
        Alpha and the Treynor Ratio.

        It gives insights in the performance a stock has to e.g. a benchmark that is not easily identified by
        looking at the raw data. This class is closely related to the Risk class which highlights things
        such as Value at Risk (VaR) and Maximum Drawdown.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/performance

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.performance.get_capital_asset_pricing_model(period='quarterly')
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2022Q3 | -0.0684 | -0.1047 |
        | 2022Q4 |  0.0857 |  0.0828 |
        | 2023Q1 |  0.075  |  0.1121 |
        | 2023Q2 |  0.0922 |  0.1342 |
        | 2023Q3 |  0.0052 | -0.0482 |
        """
        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        if self._historical.empty:
            if self._daily_historical_data.empty:
                self.get_historical_data(period="daily")
            if self._weekly_historical_data.empty:
                self.get_historical_data(period="weekly")
            if self._monthly_historical_data.empty:
                self.get_historical_data(period="monthly")
            if self._quarterly_historical_data.empty:
                self.get_historical_data(period="quarterly")
            if self._yearly_historical_data.empty:
                self.get_historical_data(period="yearly")

        historical_data = {
            "daily": self._daily_historical_data,
            "weekly": self._weekly_historical_data,
            "monthly": self._monthly_historical_data,
            "quarterly": self._quarterly_historical_data,
            "yearly": self._yearly_historical_data,
        }

        risk_free_rate_data = {
            "daily": self._daily_risk_free_rate["Adj Close"],
            "weekly": self._weekly_risk_free_rate["Adj Close"],
            "monthly": self._monthly_risk_free_rate["Adj Close"],
            "quarterly": self._quarterly_risk_free_rate["Adj Close"],
            "yearly": self._yearly_risk_free_rate["Adj Close"],
        }

        return Performance(
            tickers=self._tickers,
            historical_data=historical_data,
            risk_free_rate_data=risk_free_rate_data,
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
        )

    @property
    def risk(self) -> Risk:
        """
        This gives access to the Risk module. The Risk Module is meant to calculate metrics related to risk such
        as Value at Risk (VaR), Conditional Value at Risk (cVaR), EMWA/GARCH models and similar models.

        It gives insights in the risk a stock composes that is not perceived as easily by looking at the data.
        This class is closely related to the Performance class which highlights things such as Sharpe Ratio and
        Sortino Ratio.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/risk

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.risk.get_value_at_risk(period='yearly')
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2012   |  0      |  0      |
        | 2013   |  0.1754 |  4.96   |
        | 2014   |  1.7515 |  0.9481 |
        | 2015   | -0.1958 |  0.1454 |
        | 2016   |  0.4177 | -0.3437 |
        | 2017   |  2.6368 |  1.2225 |
        | 2018   | -0.2786 |  0.0718 |
        | 2019   |  3.2243 |  0.4707 |
        | 2020   |  1.729  |  8.3319 |
        | 2021   |  1.3179 |  0.8797 |
        | 2022   | -0.8026 | -1.0046 |
        | 2023   |  1.8549 |  1.8238 |
        """
        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        if self._historical.empty:
            if self._daily_historical_data.empty:
                self.get_historical_data(period="daily")
            if self._weekly_historical_data.empty:
                self.get_historical_data(period="weekly")
            if self._monthly_historical_data.empty:
                self.get_historical_data(period="monthly")
            if self._quarterly_historical_data.empty:
                self.get_historical_data(period="quarterly")
            if self._yearly_historical_data.empty:
                self.get_historical_data(period="yearly")

        return Risk(
            tickers=self._tickers,
            daily_historical=self._daily_historical_data,
            weekly_historical=self._weekly_historical_data,
            monthly_historical=self._monthly_historical_data,
            quarterly_historical=self._quarterly_historical_data,
            yearly_historical=self._yearly_historical_data,
            quarterly=self._quarterly,
            rounding=self._rounding,
        )

    def get_profile(self):
        """
        Returns a pandas dataframe containing the company profile information for the specified tickers.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The company profile information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL"], api_key=FMP_KEY)

        toolkit.get_profile()
        ```

        Which returns:

        |                       | MSFT                      | AAPL                  |
        |:----------------------|:--------------------------|:----------------------|
        | Symbol                | MSFT                      | AAPL                  |
        | Price                 | 316.48                    | 174.49                |
        | Beta                  | 0.903706                  | 1.286802              |
        | Average Volume        | 28153120                  | 57348456              |
        | Market Capitalization | 2353183809372             | 2744500935588         |
        | Last Dividend         | 2.7199999999999998        | 0.96                  |
        | Range                 | 213.43-366.78             | 124.17-198.23         |
        | Changes               | -0.4                      | 0.49                  |
        | Company Name          | Microsoft Corporation     | Apple Inc.            |
        | Currency              | USD                       | USD                   |
        | CIK                   | 789019                    | 320193                |
        | ISIN                  | US5949181045              | US0378331005          |
        | CUSIP                 | 594918104                 | 37833100              |
        | Exchange              | NASDAQ Global Select      | NASDAQ Global Select  |
        | Exchange Short Name   | NASDAQ                    | NASDAQ                |
        | Industry              | Software—Infrastructure   | Consumer Electronics  |
        | Website               | https://www.microsoft.com | https://www.apple.com |
        | CEO                   | Mr. Satya  Nadella        | Mr. Timothy D. Cook   |
        | Sector                | Technology                | Technology            |
        | Country               | US                        | US                    |
        | Full Time Employees   | 221000                    | 164000                |
        | Phone                 | 425 882 8080              | 408 996 1010          |
        | Address               | One Microsoft Way         | One Apple Park Way    |
        | City                  | Redmond                   | Cupertino             |
        | State                 | WA                        | CA                    |
        | ZIP Code              | 98052-6399                | 95014                 |
        | DCF Difference        | 4.56584                   | 4.15176               |
        | DCF                   | 243.594                   | 150.082               |
        | IPO Date              | 1986-03-13                | 1980-12-12            |
        """
        if not self._api_key:
            return print(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )

        if self._profile.empty:
            self._profile, self._invalid_tickers = _get_profile(
                tickers=self._tickers, api_key=self._api_key
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._profile

    def get_quote(self):
        """
        Returns a pandas dataframe containing the stock quote information for the specified tickers.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The stock quote information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AAPL"], api_key=FMP_KEY)

        toolkit.get_quote()
        ```

        Which returns:

        |                        | TSLA                         | AAPL                         |
        |:-----------------------|:-----------------------------|:-----------------------------|
        | Symbol                 | TSLA                         | AAPL                         |
        | Name                   | Tesla, Inc.                  | Apple Inc.                   |
        | Price                  | 215.49                       | 174.49                       |
        | Changes Percentage     | -1.7015                      | 0.2816                       |
        | Change                 | -3.73                        | 0.49                         |
        | Day Low                | 212.36                       | 171.96                       |
        | Day High               | 217.58                       | 175.1                        |
        | Year High              | 313.8                        | 198.23                       |
        | Year Low               | 101.81                       | 124.17                       |
        | Market Capitalization  | 682995534313                 | 2744500935588                |
        | Price Average 50 Days  | 258.915                      | 187.129                      |
        | Price Average 200 Days | 196.52345                    | 161.4698                     |
        | Exchange               | NASDAQ                       | NASDAQ                       |
        | Volume                 | 136276584                    | 61172150                     |
        | Average Volume         | 133110158                    | 57348456                     |
        | Open                   | 214.12                       | 172.3                        |
        | Previous Close         | 219.22                       | 174                          |
        | EPS                    | 3.08                         | 5.89                         |
        | PE                     | 69.96                        | 29.62                        |
        | Earnings Announcement  | 2023-10-17T20:00:00.000+0000 | 2023-10-25T10:59:00.000+0000 |
        | Shares Outstanding     | 3169499904                   | 15728700416                  |
        | Timestamp              | 2023-08-18 20:00:00          | 2023-08-18 20:00:01          |
        """
        if not self._api_key:
            return print(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )

        if self._quote.empty:
            self._quote, self._invalid_tickers = _get_quote(
                tickers=self._tickers, api_key=self._api_key
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._quote

    def get_rating(self):
        """
        Returns a pandas dataframe containing the stock rating information for the specified tickers.

        Args:
            limit (int): The number of results to return. Defaults to 100.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The stock rating information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

        rating = toolkit.get_rating()

        rating.loc['AMZN', 'Rating Recommendation'].tail()
        ```

        Which returns:

        | date                | Rating Recommendation   |
        |:--------------------|:------------------------|
        | 2023-08-01 00:00:00 | Strong Buy              |
        | 2023-08-02 00:00:00 | Strong Buy              |
        | 2023-08-03 00:00:00 | Strong Buy              |
        | 2023-08-04 00:00:00 | Strong Buy              |
        | 2023-08-07 00:00:00 | Strong Buy              |

        """
        if not self._api_key:
            return print(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )

        if self._rating.empty:
            self._rating, self._invalid_tickers = _get_rating(
                tickers=self._tickers, api_key=self._api_key
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._rating

    def get_analyst_estimates(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Obtain analyst estimates regarding revenues, EBITDA, EBIT, Net Income
        SGA Expenses and EPS. The number of analysts are also reported.

        Note that this information requires a Premium FMP subscription.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period. E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pandas.DataFrame: The analyst estimates for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key=FMP_KEY, start_date="2021-05-01", quarterly=False
        )

        analyst_estimates = toolkit.get_analyst_estimates()

        analyst_estimates.loc['AAPL']
        ```

        Which returns:

        |                               |         2021 |         2022 |         2023 |         2024 |
        |:------------------------------|-------------:|-------------:|-------------:|-------------:|
        | Estimated Revenue Low         |  2.98738e+11 |  3.07919e+11 |  3.3871e+11  |  2.93633e+11 |
        | Estimated Revenue High        |  4.48107e+11 |  4.61878e+11 |  5.08066e+11 |  4.4045e+11  |
        | Estimated Revenue Average     |  3.73422e+11 |  3.84898e+11 |  4.23388e+11 |  3.67042e+11 |
        | Estimated EBITDA Low          |  8.50991e+10 |  1.00742e+11 |  1.10816e+11 |  1.07415e+11 |
        | Estimated EBITDA High         |  1.27649e+11 |  1.51113e+11 |  1.66224e+11 |  1.61122e+11 |
        | Estimated EBITDA Average      |  1.06374e+11 |  1.25928e+11 |  1.3852e+11  |  1.34269e+11 |
        | Estimated EBIT Low            |  7.62213e+10 |  9.05428e+10 |  9.9597e+10  |  9.81566e+10 |
        | Estimated EBIT High           |  1.14332e+11 |  1.35814e+11 |  1.49396e+11 |  1.47235e+11 |
        | Estimated EBIT Average        |  9.52766e+10 |  1.13178e+11 |  1.24496e+11 |  1.22696e+11 |
        | Estimated Net Income Low      |  6.54258e+10 |  7.62265e+10 |  8.38492e+10 |  8.23371e+10 |
        | Estimated Net Income High     |  9.81387e+10 |  1.1434e+11  |  1.25774e+11 |  1.23506e+11 |
        | Estimated Net Income Average  |  8.17822e+10 |  9.52832e+10 |  1.04811e+11 |  1.02921e+11 |
        | Estimated SGA Expense Low     |  1.48491e+10 |  1.85317e+10 |  2.03848e+10 |  2.04857e+10 |
        | Estimated SGA Expense High    |  2.22737e+10 |  2.77975e+10 |  3.05772e+10 |  3.07286e+10 |
        | Estimated SGA Expense Average |  1.85614e+10 |  2.31646e+10 |  2.5481e+10  |  2.56072e+10 |
        | Estimated EPS Average         |  4.26        |  5.465       |  6.01        |  6.2612      |
        | Estimated EPS High            |  5.12        |  6.56        |  7.21        |  7.5135      |
        | Estimated EPS Low             |  3.4         |  4.37        |  4.81        |  5.009       |
        | Number of Analysts            | 14           | 16           | 12           | 10           |
        """
        if not self._api_key:
            return print(
                "The requested data requires the api_key parameter to be set, consider obtaining "
                "a key with the following link: https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThis functionality also requires a Premium subscription. You can get 15% off by "
                "using the above affiliate link which also supports the project."
            )
        if self._analyst_estimates.empty or overwrite:
            (
                self._analyst_estimates,
                self._invalid_tickers,
            ) = _get_analyst_estimates(
                tickers=self._tickers,
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                rounding=rounding if rounding else self._rounding,
                sleep_timer=self._sleep_timer,
                progress_bar=self._progress_bar,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if growth:
            self._analyst_estimates_growth = _calculate_growth(
                self._analyst_estimates,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        if len(self._tickers) == 1:
            return (
                self._analyst_estimates_growth.loc[self._tickers[0]]
                if growth
                else self._analyst_estimates.loc[self._tickers[0]]
            )

        return self._analyst_estimates_growth if growth else self._analyst_estimates

    def get_earnings_calendar(
        self,
        actual_dates: bool = True,
        overwrite: bool = False,
        rounding: int | None = None,
    ):
        """
        Obtain Earnings Calendars for any range of companies. You have the option to
        obtain the actual dates or to convert to the corresponding quarters.

        Note that this information requires a Premium FMP subscription.

        Args:
            actual_dates (bool): Defines whether to return the actual dates or the corresponding quarters.
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The earnings calendar for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key=FMP_KEY, start_date="2022-08-01", quarterly=False
        )

        earning_calendar = toolkit.get_earnings_calendar()

        earning_calendar.loc['AMZN']
        ```

        Which returns:

        | date        |    EPS |   Estimated EPS |       Revenue |   Estimated Revenue | Fiscal Date Ending   | Time   |
        |:------------|-------:|----------------:|--------------:|--------------------:|:---------------------|:-------|
        | 2022-10-27  |   0.17 |            0.22 |   1.27101e+11 |       nan           | 2022-09-30           | amc    |
        | 2023-02-02  |   0.25 |            0.18 |   1.49204e+11 |         1.5515e+11  | 2022-12-31           | amc    |
        | 2023-04-27  |   0.31 |            0.21 |   1.27358e+11 |         1.24551e+11 | 2023-03-31           | amc    |
        | 2023-08-03  |   0.65 |            0.35 |   1.34383e+11 |         1.19573e+11 | 2023-06-30           | amc    |
        | 2023-10-25  | nan    |            0.56 | nan           |         1.41407e+11 | 2023-09-30           | amc    |
        | 2024-01-31  | nan    |          nan    | nan           |       nan           | 2023-12-30           | amc    |
        | 2024-04-25  | nan    |          nan    | nan           |       nan           | 2024-03-30           | amc    |
        | 2024-08-01  | nan    |          nan    | nan           |       nan           | 2024-06-30           | amc    |
        """
        if not self._api_key:
            return print(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThis functionality also requires a Premium subscription. You can get 15% off by using "
                "the above affiliate link which also supports the project."
            )
        if self._earnings_calendar.empty or overwrite:
            (
                self._earnings_calendar,
                self._invalid_tickers,
            ) = _get_earnings_calendar(
                tickers=self._tickers,
                api_key=self._api_key,
                start_date=self._start_date,
                end_date=self._end_date,
                actual_dates=actual_dates,
                sleep_timer=self._sleep_timer,
                progress_bar=self._progress_bar,
            )

        earnings_calendar = self._earnings_calendar.round(
            rounding if rounding else self._rounding
        ).loc[self._start_date : self._end_date]

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1:
            return earnings_calendar.loc[self._tickers[0]]

        return earnings_calendar

    def get_revenue_geographic_segmentation(self, overwrite: bool = False):
        """
        Obtain revenue by geographic segmentation (e.g. United States, Europe, Asia).

        Note that this information requires a Premium FMP subscription.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The revenue by geographic segmentation for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key=FMP_KEY, start_date="2021-05-01", quarterly=False
        )

        geographic_segmentation = toolkit.get_revenue_geographic_segmentation()

        geographic_segmentation.loc['AAPL']
        ```

        Which returns:

        |              |       2020 |       2021 |       2022 |       2023 |
        |:-------------|-----------:|-----------:|-----------:|-----------:|
        | Americas     | 4.631e+10  | 5.1496e+10 | 4.9278e+10 | 3.5383e+10 |
        | Asia Pacific | 8.225e+09  | 9.81e+09   | 9.535e+09  | 5.63e+09   |
        | China        | 2.1313e+10 | 2.5783e+10 | 2.3905e+10 | 1.5758e+10 |
        | Europe       | 2.7306e+10 | 2.9749e+10 | 2.7681e+10 | 2.0205e+10 |
        | Japan        | 8.285e+09  | 7.107e+09  | 6.755e+09  | 4.821e+09  |

        """
        if not self._api_key:
            return print(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThis functionality also requires a Professional or Enterprise subscription. "
                "You can get 15% off by using the above affiliate link which also supports the project."
            )
        if self._revenue_geographic_segmentation.empty or overwrite:
            (
                self._revenue_geographic_segmentation,
                self._invalid_tickers,
            ) = _get_revenue_segmentation(
                tickers=self._tickers,
                method="geographic",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                sleep_timer=self._sleep_timer,
                progress_bar=self._progress_bar,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1:
            return self._revenue_geographic_segmentation.loc[self._tickers[0]]

        return self._revenue_geographic_segmentation

    def get_revenue_product_segmentation(self, overwrite: bool = False):
        """
        Obtain revenue by product segmentation (e.g. iPad, Advertisement, Windows).

        Note that this information requires a Premium FMP subscription.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The revenue by product segmentation for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key=FMP_KEY, start_date="2021-05-01", quarterly=False
        )

        product_segmentation = toolkit.get_revenue_product_segmentation()

        product_segmentation.loc['MSFT']
        ```

        Which returns:

        |                                    |     2022Q2 |     2022Q3 |     2022Q4 |     2023Q1 |      2023Q2 |
        |:-----------------------------------|-----------:|-----------:|-----------:|-----------:|------------:|
        | Devices                            | 1.581e+09  | 1.448e+09  | 1.43e+09   | 1.282e+09  |  1.361e+09  |
        | Enterprise Services                | 1.902e+09  | 1.876e+09  | 1.862e+09  | 2.007e+09  |  1.977e+09  |
        | Gaming                             | 3.455e+09  | 3.61e+09   | 4.758e+09  | 3.607e+09  |  3.491e+09  |
        | Linked In Corporation              | 3.712e+09  | 3.663e+09  | 3.876e+09  | 3.697e+09  |  3.909e+09  |
        | Office Products And Cloud Services | 1.1639e+10 | 1.1548e+10 | 1.1837e+10 | 1.2438e+10 |  1.2905e+10 |
        | Other Products And Services        | 1.403e+09  | 1.348e+09  | 1.359e+09  | 1.428e+09  | -3.924e+09  |
        | Search And News Advertising        | 2.926e+09  | 2.928e+09  | 3.223e+09  | 3.045e+09  |  3.012e+09  |
        | Server Products And Cloud Services | 1.8839e+10 | 1.8388e+10 | 1.9594e+10 | 2.0025e+10 |  2.1963e+10 |
        | Windows                            | 6.408e+09  | 5.313e+09  | 4.808e+09  | 5.328e+09  |  6.058e+09  |

        """
        if not self._api_key:
            return print(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThis functionality also requires a Professional or Enterprise subscription. You can get 15% off by using "
                "the above affiliate link which also supports the project."
            )
        if self._revenue_product_segmentation.empty or overwrite:
            (
                self._revenue_product_segmentation,
                self._invalid_tickers,
            ) = _get_revenue_segmentation(
                tickers=self._tickers,
                method="product",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                sleep_timer=self._sleep_timer,
                progress_bar=self._progress_bar,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1:
            return self._revenue_product_segmentation.loc[self._tickers[0]]

        return self._revenue_product_segmentation

    def get_historical_data(
        self,
        period: str = "daily",
        return_column: str = "Adj Close",
        fill_nan: bool = True,
        overwrite: bool = False,
        rounding: int | None = None,
    ):
        """
        Returns historical data for the specified tickers. This contains the following columns:
            - Open: The opening price for the period.
            - High: The highest price for the period.
            - Low: The lowest price for the period.
            - Close: The closing price for the period.
            - Adj Close: The adjusted closing price for the period.
            - Volume: The volume for the period.
            - Dividends: The dividends for the period.
            - Return: The return for the period.
            - Volatility: The volatility for the period.
            - Excess Return: The excess return for the period. This is defined as the return minus
            the a predefined risk free rate. Only calculated when excess_return is True.
            - Excess Volatility: The excess volatility for the period. This is defined as the volatility
            of the excess return. Only calculated when excess_return is True.
            - Cumulative Return: The cumulative return for the period.

        If a benchmark ticker is selected, it also calculates the benchmark ticker together with the results.
        By default this is set to "SPY" (S&P 500 ETF) but can be any ticker. This is relevant for calculations
        for models such as CAPM, Alpha and Beta.

        Args:
            start (str): The start date for the historical data. Defaults to None.
            end (str): The end date for the historical data. Defaults to None.
            period (str): The interval at which the historical data should be
            returned - daily, weekly, monthly, quarterly, or yearly.
            Defaults to "daily".
            fill_nan (bool): Defines whether to forward fill NaN values. This defaults
            to True to prevent holes in the dataset. This is especially relevant for
            technical indicators.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.

        Raises:
            ValueError: If an invalid value is specified for period.

        Returns:
            pandas.DataFrame: The historical data for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit("AAPL", api_key=FMP_KEY)

        toolkit.get_historical_data(period="yearly")
        ```

        Which returns:

        | Date   |     Open |     High |      Low |    Close |   Adj Close |      Volume |   Dividends |     Return |   Volatility |   Excess Return |   Excess Volatility |   Cumulative Return |
        |:-------|---------:|---------:|---------:|---------:|------------:|------------:|------------:|-----------:|-------------:|----------------:|--------------------:|--------------------:|
        | 2013   |  19.7918 |  20.0457 |  19.7857 |  20.0364 |     17.5889 | 2.23084e+08 |    0.108929 |  0         |     0.240641 |       0         |            0.244248 |             1       |
        | 2014   |  28.205  |  28.2825 |  27.5525 |  27.595  |     24.734  | 1.65614e+08 |    0.461429 |  0.406225  |     0.216574 |       0.384525  |            0.219536 |             1.40623 |
        | 2015   |  26.7525 |  26.7575 |  26.205  |  26.315  |     23.9886 | 1.63649e+08 |    0.5075   | -0.0301373 |     0.267373 |      -0.0528273 |            0.269845 |             1.36385 |
        | 2016   |  29.1625 |  29.3    |  28.8575 |  28.955  |     26.9824 | 1.22345e+08 |    0.5575   |  0.124804  |     0.233383 |       0.100344  |            0.240215 |             1.53406 |
        | 2017   |  42.63   |  42.6475 |  42.305  |  42.3075 |     40.0593 | 1.04e+08    |    0.615    |  0.484644  |     0.176058 |       0.460594  |            0.17468  |             2.27753 |
        | 2018   |  39.6325 |  39.84   |  39.12   |  39.435  |     37.9    | 1.40014e+08 |    0.705    | -0.0539019 |     0.287421 |      -0.0807619 |            0.289905 |             2.15477 |
        | 2019   |  72.4825 |  73.42   |  72.38   |  73.4125 |     71.615  | 1.00806e+08 |    0.76     |  0.889578  |     0.261384 |       0.870388  |            0.269945 |             4.0716  |
        | 2020   | 134.08   | 134.74   | 131.72   | 132.69   |    130.559  | 9.91166e+07 |    0.8075   |  0.823067  |     0.466497 |       0.813897  |            0.470743 |             7.4228  |
        | 2021   | 178.09   | 179.23   | 177.26   | 177.57   |    175.795  | 6.40623e+07 |    0.865    |  0.346482  |     0.251019 |       0.331362  |            0.251429 |             9.99467 |
        | 2022   | 128.41   | 129.95   | 127.43   | 129.93   |    129.378  | 7.70342e+07 |    0.91     | -0.264042  |     0.356964 |      -0.302832  |            0.377293 |             7.35566 |
        | 2023   | 187.84   | 188.51   | 187.68   | 188.108  |    188.108  | 4.72009e+06 |    0.71     |  0.453941  |     0.213359 |       0.412901  |            0.22327  |            10.6947  |
        """
        if self._daily_risk_free_rate.empty or overwrite:
            self.get_treasury_data()

        if self._daily_historical_data.empty or overwrite:
            self._daily_historical_data, self._invalid_tickers = _get_historical_data(
                tickers=self._tickers + [self._benchmark_ticker]
                if self._benchmark_ticker
                else self._tickers,
                start=self._start_date,
                end=self._end_date,
                interval="1d",
                return_column=return_column,
                risk_free_rate=self._daily_risk_free_rate,
                progress_bar=self._progress_bar,
                fill_nan=fill_nan,
                rounding=rounding if rounding else self._rounding,
            )

            # Change the benchmark ticker name to Benchmark
            self._daily_historical_data = self._daily_historical_data.rename(
                columns={self._benchmark_ticker: "Benchmark"}, level=1
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if period == "daily":
            historical_data = self._daily_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        if period == "weekly":
            if self._weekly_risk_free_rate.empty or overwrite:
                self.get_treasury_data(period="weekly")

            self._weekly_historical_data = _convert_daily_to_other_period(
                period="weekly",
                daily_historical_data=self._daily_historical_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=self._weekly_risk_free_rate,
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._weekly_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        if period == "monthly":
            if self._monthly_risk_free_rate.empty or overwrite:
                self.get_treasury_data(period="monthly")

            self._monthly_historical_data = _convert_daily_to_other_period(
                period="monthly",
                daily_historical_data=self._daily_historical_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=self._monthly_risk_free_rate,
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._monthly_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        if period == "quarterly":
            if self._quarterly_risk_free_rate.empty or overwrite:
                self.get_treasury_data(period="quarterly")

            self._quarterly_historical_data = _convert_daily_to_other_period(
                period="quarterly",
                daily_historical_data=self._daily_historical_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=self._quarterly_risk_free_rate,
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._quarterly_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        if period == "yearly":
            if self._yearly_risk_free_rate.empty or overwrite:
                self.get_treasury_data(period="yearly")

            self._yearly_historical_data = _convert_daily_to_other_period(
                period="yearly",
                daily_historical_data=self._daily_historical_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=self._yearly_risk_free_rate,
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._yearly_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        raise ValueError(
            "Please choose from daily, weekly, monthly, quarterly or yearly as period."
        )

    def get_treasury_data(
        self,
        period: str = "daily",
        fill_nan: bool = True,
        divide_ohlc_by: int | float | None = 100,
        rounding: int | None = None,
    ):
        """
        Retrieve daily, weekly, monthly, quarterly or yearly treasury data. This can be from FinancialModelingPrep
        or from YahooFinance. FinancialModelingPrep is by far a more extensive dataset containing daily data from
        1 month to 30 years. YahooFinance only contains daily data for 5, 10 and 30 years but is a free alternative.

        Args:
            period (str): The interval at which the treasury data should be returned - daily, weekly, monthly, quarterly, or yearly.
            fill_nan (bool): Defines whether to forward fill NaN values. This defaults
            to True to prevent holes in the dataset. This is especially relevant for
            technical indicators.

        Returns:
            pd.DataFrame: A DataFrame containing the treasury data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        companies = Toolkit(["AAPL", "MSFT"], api_key=FMP_KEY, start_date="2023-08-10")

        companies.get_treasury_data()
        ```

        Which returns:


        """
        risk_free_names = {
            "13w": "13 Week",
            "5y": "5 Year",
            "10y": "10 Year",
            "30y": "30 Year",
        }
        treasury_names = {
            "^IRX": "13 Week",
            "^FVX": "5 Year",
            "^TNX": "10 Year",
            "^TYX": "30 Year",
        }

        risk_free_rate = risk_free_names[self._risk_free_rate]

        if self._daily_treasury_data.empty:
            daily_treasury_data, _ = _get_historical_data(
                tickers=["^IRX", "^FVX", "^TNX", "^TYX"],
                start=self._start_date,
                end=self._end_date,
                progress_bar=False,
                divide_ohlc_by=divide_ohlc_by,
                rounding=rounding if rounding else self._rounding,
            )

            daily_treasury_data = daily_treasury_data.rename(
                treasury_names, axis=1, level=1
            )

            self._daily_treasury_data = daily_treasury_data
            self._daily_risk_free_rate = self._daily_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            if fill_nan:
                self._daily_treasury_data = self._daily_treasury_data.ffill()

        if period == "daily":
            return self._daily_treasury_data.loc[self._start_date : self._end_date, :]
        if period == "weekly":
            self._weekly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding else self._rounding,
            )

            self._weekly_risk_free_rate = self._weekly_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            return self._weekly_treasury_data.loc[self._start_date : self._end_date, :]
        if period == "monthly":
            self._monthly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding else self._rounding,
            )

            self._monthly_risk_free_rate = self._monthly_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            return self._monthly_treasury_data.loc[self._start_date : self._end_date, :]
        if period == "quarterly":
            self._quarterly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding else self._rounding,
            )

            self._quarterly_risk_free_rate = self._quarterly_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            return self._quarterly_treasury_data.loc[
                self._start_date : self._end_date, :
            ]
        if period == "yearly":
            self._yearly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding else self._rounding,
            )

            self._yearly_risk_free_rate = self._yearly_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            return self._yearly_treasury_data.loc[self._start_date : self._end_date, :]

        raise ValueError(
            "Please choose from daily, weekly, monthly, quarterly or yearly as period."
        )

    def get_balance_sheet_statement(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Retrieves the balance sheet statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved balance sheet statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "MU"], api_key=FMP_KEY, quarterly=True, start_date='2022-05-01')

        balance_sheet_statements = toolkit.get_balance_sheet_statement()

        balance_sheet_statements.loc['MU']
        ```

        Which returns:

        |                                          |      2022Q2 |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:-----------------------------------------|------------:|------------:|------------:|------------:|------------:|
        | Cash and Cash Equivalents                |  9.157e+09  |  8.262e+09  |  9.574e+09  |  9.798e+09  |  9.298e+09  |
        | Short Term Investments                   |  1.07e+09   |  1.069e+09  |  1.007e+09  |  1.02e+09   |  1.054e+09  |
        | Cash and Short Term Investments          |  1.0227e+10 |  9.331e+09  |  1.0581e+10 |  1.0818e+10 |  1.0352e+10 |
        | Accounts Receivable                      |  6.229e+09  |  5.13e+09   |  3.318e+09  |  2.278e+09  |  2.429e+09  |
        | Inventory                                |  5.629e+09  |  6.663e+09  |  8.359e+09  |  8.129e+09  |  8.238e+09  |
        | Other Current Assets                     |  6.08e+08   |  6.44e+08   |  6.63e+08   |  6.73e+08   |  7.15e+08   |
        | Total Current Assets                     |  2.2708e+10 |  2.1781e+10 |  2.2921e+10 |  2.1898e+10 |  2.1734e+10 |
        | Property, Plant and Equipment            |  3.7355e+10 |  3.9227e+10 |  4.0028e+10 |  3.9758e+10 |  3.9382e+10 |
        | Goodwill                                 |  1.228e+09  |  1.228e+09  |  1.228e+09  |  1.228e+09  |  1.252e+09  |
        | Intangible Assets                        |  4.15e+08   |  4.21e+08   |  4.28e+08   |  4.1e+08    |  4.1e+08    |
        | Long Term Investments                    |  1.646e+09  |  1.647e+09  |  1.426e+09  |  1.212e+09  |  9.73e+08   |
        | Tax Assets                               |  6.82e+08   |  7.02e+08   |  6.72e+08   |  6.97e+08   |  7.08e+08   |
        | Other Fixed Assets                       |  1.262e+09  |  1.277e+09  |  1.171e+09  |  1.317e+09  |  1.221e+09  |
        | Fixed Assets                             |  4.2588e+10 |  4.4502e+10 |  4.4953e+10 |  4.4622e+10 |  4.3946e+10 |
        | Other Assets                             |  0          |  0          |  0          |  0          |  0          |
        | Total Assets                             |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Accounts Payable                         |  2.019e+09  |  2.142e+09  |  1.789e+09  |  1.689e+09  |  1.64e+09   |
        | Short Term Debt                          |  1.07e+08   |  1.03e+08   |  1.71e+08   |  2.37e+08   |  2.59e+08   |
        | Tax Payables                             |  3.82e+08   |  4.2e+08    |  4.19e+08   |  2.41e+08   |  1.48e+08   |
        | Deferred Revenue                         |  0          |  0          |  0          |  0          | -1.64e+09   |
        | Other Current Liabilities                |  4.883e+09  |  5.294e+09  |  4.565e+09  |  3.329e+09  |  4.845e+09  |
        | Total Current Liabilities                |  7.009e+09  |  7.539e+09  |  6.525e+09  |  5.255e+09  |  5.104e+09  |
        | Long Term Debt                           |  7.485e+09  |  7.413e+09  |  1.0719e+10 |  1.2647e+10 |  1.3589e+10 |
        | Deferred Revenue Non Current             |  6.63e+08   |  5.89e+08   |  5.16e+08   |  5.29e+08   |  6.32e+08   |
        | Deferred Tax Liabilities                 |  0          |  0          |  0          |  0          |  0          |
        | Other Non Current Liabilities            |  8.58e+08   |  8.35e+08   |  8.08e+08   |  8.32e+08   |  9.5e+08    |
        | Total Non Current Liabilities            |  9.006e+09  |  8.837e+09  |  1.2043e+10 |  1.4008e+10 |  1.5171e+10 |
        | Other Liabilities                        |  0          |  0          |  0          |  0          |  0          |
        | Capital Lease Obligations                |  6.29e+08   |  6.1e+08    |  6.25e+08   |  6.1e+08    |  6.03e+08   |
        | Total Liabilities                        |  1.6015e+10 |  1.6376e+10 |  1.8568e+10 |  1.9263e+10 |  2.0275e+10 |
        | Preferred Stock                          |  0          |  0          |  0          |  0          |  0          |
        | Common Stock                             |  1.22e+08   |  1.23e+08   |  1.23e+08   |  1.23e+08   |  1.24e+08   |
        | Retained Earnings                        |  4.5916e+10 |  4.7274e+10 |  4.6873e+10 |  4.4426e+10 |  4.2391e+10 |
        | Accumulated Other Comprehensive Income   | -3.64e+08   | -5.6e+08    | -4.73e+08   | -3.73e+08   | -3.4e+08    |
        | Other Total Shareholder Equity           |  3.607e+09  |  3.07e+09   |  2.783e+09  |  3.081e+09  |  3.23e+09   |
        | Total Shareholder Equity                 |  4.9281e+10 |  4.9907e+10 |  4.9306e+10 |  4.7257e+10 |  4.5405e+10 |
        | Total Equity                             |  4.9281e+10 |  4.9907e+10 |  4.9306e+10 |  4.7257e+10 |  4.5405e+10 |
        | Total Liabilities and Shareholder Equity |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Minority Interest                        |  0          |  0          |  0          |  0          |  0          |
        | Total Liabilities and Equity             |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Total Investments                        |  2.716e+09  |  2.716e+09  |  2.433e+09  |  2.232e+09  |  2.027e+09  |
        | Total Debt                               |  7.592e+09  |  7.516e+09  |  1.089e+10  |  1.2884e+10 |  1.3848e+10 |
        | Net Debt                                 | -1.565e+09  | -7.46e+08   |  1.316e+09  |  3.086e+09  |  4.55e+09   |
        """
        if not self._api_key and self._balance_sheet_statement.empty:
            return print(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )

        if self._balance_sheet_statement.empty or overwrite:
            (
                self._balance_sheet_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                tickers=self._tickers,
                statement="balance",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                rounding=rounding if rounding else self._rounding,
                statement_format=self._balance_sheet_statement_generic,
                statistics_format=self._statistics_statement_generic,
                sleep_timer=self._sleep_timer,
                progress_bar=self._progress_bar,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        balance_sheet_statement = self._balance_sheet_statement

        if trailing:
            balance_sheet_statement = (
                self._balance_sheet_statement.T.rolling(trailing).sum().T
            )

        if growth:
            self._balance_sheet_statement_growth = _calculate_growth(
                balance_sheet_statement,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._balance_sheet_statement_growth.loc[self._tickers[0]]
                if growth
                else balance_sheet_statement.loc[self._tickers[0]]
            )

        return (
            self._balance_sheet_statement_growth if growth else balance_sheet_statement
        )

    def get_income_statement(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Retrieves the income statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved income statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "MU"], api_key=FMP_KEY, quarterly=True, start_date='2022-05-01')

        income_sheet_statements = toolkit.get_income_statement()

        income_sheet_statements.loc['TSLA']
        ```

        Which returns:

        |                                              |     2022Q2 |      2022Q3 |      2022Q4 |      2023Q1 |     2023Q2 |
        |:---------------------------------------------|-----------:|------------:|------------:|------------:|-----------:|
        | Revenue                                      | 1.6934e+10 |  2.1454e+10 |  2.4318e+10 |  2.3329e+10 | 2.4927e+10 |
        | Cost of Goods Sold                           | 1.27e+10   |  1.6072e+10 |  1.8541e+10 |  1.8818e+10 | 2.0394e+10 |
        | Gross Profit                                 | 4.234e+09  |  5.382e+09  |  5.777e+09  |  4.511e+09  | 4.533e+09  |
        | Gross Profit Ratio                           | 0.25003    |  0.250862   |  0.237561   |  0.193364   | 0.181851   |
        | Research and Development Expenses            | 6.67e+08   |  7.33e+08   |  8.1e+08    |  7.71e+08   | 9.43e+08   |
        | General and Administrative Expenses          | 0          |  0          |  0          |  0          | 0          |
        | Selling and Marketing Expenses               | 0          |  0          |  0          |  0          | 0          |
        | Selling, General and Administrative Expenses | 9.61e+08   |  9.61e+08   |  1.032e+09  |  1.076e+09  | 1.191e+09  |
        | Other Expenses                               | 2.8e+07    | -8.5e+07    | -4.2e+07    | -4.8e+07    | 3.28e+08   |
        | Operating Expenses                           | 1.628e+09  |  1.694e+09  |  1.842e+09  |  1.847e+09  | 2.134e+09  |
        | Cost and Expenses                            | 1.4328e+10 |  1.7766e+10 |  2.0383e+10 |  2.0665e+10 | 2.2528e+10 |
        | Interest Income                              | 2.6e+07    |  8.6e+07    |  1.57e+08   |  2.13e+08   | 2.38e+08   |
        | Interest Expense                             | 4.4e+07    |  5.3e+07    |  3.3e+07    |  2.9e+07    | 2.8e+07    |
        | Depreciation and Amortization                | 1.118e+09  |  9.57e+08   |  1.138e+09  |  1.211e+09  | 1.72e+09   |
        | EBITDA                                       | 3.582e+09  |  4.645e+09  |  5.039e+09  |  3.875e+09  | 4.119e+09  |
        | EBITDA Ratio                                 | 0.211527   |  0.21651    |  0.207213   |  0.166102   | 0.165243   |
        | Operating Income                             | 2.464e+09  |  3.688e+09  |  3.901e+09  |  2.664e+09  | 2.399e+09  |
        | Operating Income Ratio                       | 0.145506   |  0.171903   |  0.160416   |  0.114193   | 0.096241   |
        | Total Other Income                           | 1e+07      | -5.2e+07    |  8.2e+07    |  1.36e+08   | 5.38e+08   |
        | Income Before Tax                            | 2.474e+09  |  3.636e+09  |  3.983e+09  |  2.8e+09    | 2.937e+09  |
        | Income Before Tax Ratio                      | 0.146097   |  0.169479   |  0.163788   |  0.120022   | 0.117824   |
        | Income Tax Expense                           | 2.05e+08   |  3.05e+08   |  2.76e+08   |  2.61e+08   | 3.23e+08   |
        | Net Income                                   | 2.259e+09  |  3.292e+09  |  3.687e+09  |  2.513e+09  | 2.703e+09  |
        | Net Income Ratio                             | 0.1334     |  0.153445   |  0.151616   |  0.10772    | 0.108437   |
        | EPS                                          | 0.73       |  1.05       |  1.18       |  0.8        | 0.85       |
        | EPS Diluted                                  | 0.65       |  0.95       |  1.07       |  0.73       | 0.78       |
        | Weighted Average Shares                      | 3.111e+09  |  3.146e+09  |  3.16e+09   |  3.166e+09  | 3.171e+09  |
        | Weighted Average Shares Diluted              | 3.465e+09  |  3.468e+09  |  3.471e+09  |  3.468e+09  | 3.478e+09  |
        """
        if not self._api_key and self._income_statement.empty:
            return print(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )

        if self._income_statement.empty or overwrite:
            (
                self._income_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                tickers=self._tickers,
                statement="income",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                rounding=rounding if rounding else self._rounding,
                statement_format=self._income_statement_generic,
                statistics_format=self._statistics_statement_generic,
                sleep_timer=self._sleep_timer,
                progress_bar=self._progress_bar,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        income_statement = self._income_statement

        if trailing:
            income_statement = self._income_statement.T.rolling(trailing).sum().T

        if growth:
            self._income_statement_growth = _calculate_growth(
                income_statement,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._income_statement_growth.loc[self._tickers[0]]
                if growth
                else income_statement.loc[self._tickers[0]]
            )

        return self._income_statement_growth if growth else income_statement

    def get_cash_flow_statement(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MU", "AMZN"], api_key=FMP_KEY, quarterly=True, start_date='2022-09-01')

        cash_flow_statements = toolkit.get_cash_flow_statement()

        cash_flow_statements.loc['AMZN']
        ```

        Which returns:

        |                               |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:------------------------------|------------:|------------:|------------:|------------:|
        | Net Income                    |  2.872e+09  |  2.78e+08   |  3.172e+09  |  6.75e+09   |
        | Depreciation and Amortization |  1.0204e+10 |  1.2685e+10 |  1.1123e+10 |  1.1589e+10 |
        | Deferred Income Tax           | -8.25e+08   | -3.367e+09  | -4.72e+08   | -2.744e+09  |
        | Stock Based Compensation      |  5.556e+09  |  5.606e+09  |  4.748e+09  |  7.127e+09  |
        | Change in Working Capital     | -5.254e+09  |  1.0526e+10 | -1.4317e+10 | -6.293e+09  |
        | Accounts Receivables          | -4.794e+09  | -8.788e+09  |  1.521e+09  | -5.167e+09  |
        | Inventory                     |  7.32e+08   |  3.18e+09   |  3.71e+08   | -2.373e+09  |
        | Accounts Payables             | -1.226e+09  |  9.852e+09  | -1.1264e+10 |  3.029e+09  |
        | Other Working Capital         |  3.4e+07    |  6.282e+09  | -4.945e+09  | -1.782e+09  |
        | Other Non Cash Items          | -1.149e+09  |  3.445e+09  |  5.34e+08   |  4.7e+07    |
        | Cash Flow from Operations     |  1.1404e+10 |  2.9173e+10 |  4.788e+09  |  1.6476e+10 |
        | Property, Plant and Equipment | -1.6378e+10 | -1.6592e+10 | -1.4207e+10 | -1.1455e+10 |
        | Acquisitions                  | -8.85e+08   | -8.31e+08   | -3.513e+09  | -3.16e+08   |
        | Purchases of Investments      | -2.39e+08   | -2.33e+08   | -3.38e+08   | -4.96e+08   |
        | Sales of Investments          |  5.57e+08   |  5.683e+09  |  1.115e+09  |  1.551e+09  |
        | Other Investing Activities    |  1.337e+09  |  1.152e+09  |  1.137e+09  |  1.043e+09  |
        | Cash Flow from Investing      | -1.5608e+10 | -1.0821e+10 | -1.5806e+10 | -9.673e+09  |
        | Debt Repayment                | -9.429e+09  | -1.8756e+10 | -6.369e+09  | -1.0861e+10 |
        | Common Stock Issued           |  0          |  0          |  0          |  0          |
        | Common Stock Purchased        |  0          |  6e+09      |  0          |  0          |
        | Dividends Paid                |  0          |  0          |  0          |  0          |
        | Other Financing Activities    |  1.2445e+10 |  1.2842e+10 |  1.2723e+10 |  4.322e+09  |
        | Cash Flow from Financing      |  3.016e+09  |  8.6e+07    |  6.354e+09  | -6.539e+09  |
        | Forex Changes on Cash         | -1.334e+09  |  6.37e+08   |  1.45e+08   |  6.9e+07    |
        | Net Change in Cash            | -2.522e+09  |  1.9075e+10 | -4.519e+09  |  3.33e+08   |
        | Cash End of Period            |  3.5178e+10 |  5.4253e+10 |  4.9734e+10 |  5.0067e+10 |
        | Cash Beginning of Period      |  3.77e+10   |  3.5178e+10 |  5.4253e+10 |  4.9734e+10 |
        | Operating Cash Flow           |  1.1404e+10 |  2.9173e+10 |  4.788e+09  |  1.6476e+10 |
        | Capital Expenditure           | -1.6378e+10 | -1.6592e+10 | -1.4207e+10 | -1.1455e+10 |
        | Free Cash Flow                | -4.974e+09  |  1.2581e+10 | -9.419e+09  |  5.021e+09  |
        """
        if not self._api_key and self._cash_flow_statement.empty:
            return print(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )

        if self._cash_flow_statement.empty or overwrite:
            (
                self._cash_flow_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                tickers=self._tickers,
                statement="cashflow",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                rounding=rounding if rounding else self._rounding,
                statement_format=self._cash_flow_statement_generic,
                statistics_format=self._statistics_statement_generic,
                sleep_timer=self._sleep_timer,
                progress_bar=self._progress_bar,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        cash_flow_statement = self._cash_flow_statement

        if trailing:
            cash_flow_statement = self._cash_flow_statement.T.rolling(trailing).sum().T

        if growth:
            self._cash_flow_statement_growth = _calculate_growth(
                cash_flow_statement,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            return (
                self._cash_flow_statement_growth.loc[self._tickers[0]]
                if growth
                else cash_flow_statement.loc[self._tickers[0]]
            )

        return self._cash_flow_statement_growth if growth else cash_flow_statement

    def get_statistics_statement(
        self,
        overwrite: bool = False,
    ):
        """
        Retrieves the balance, cash and income statistics for the company(s) from the specified source.

        Note that this also obtains the balance sheet statement at the same time given that it's the same
        API call. This is done to reduce the number of API calls to FinancialModelingPrep.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved statistics statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit("TSLA", api_key=FMP_KEY, quarterly=True, start_date='2023-05-01')

        toolkit.get_statistics_statement()
        ```

        Which returns:

        |                   | 2023Q2                                                                                            |
        |:------------------|:--------------------------------------------------------------------------------------------------|
        | Reported Currency | USD                                                                                               |
        | CIK ID            | 1318605                                                                                           |
        | Filling Date      | 2023-07-24                                                                                        |
        | Accepted Date     | 2023-07-21 18:08:29                                                                               |
        | Calendar Year     | 2023                                                                                              |
        | Period            | Q2                                                                                                |
        | SEC Link          | https://www.sec.gov/Archives/edgar/data/1318605/000095017023033872/0000950170-23-033872-index.htm |
        | Document Link     | https://www.sec.gov/Archives/edgar/data/1318605/000095017023033872/tsla-20230630.htm              |

        """
        if not self._api_key and self._statistics_statement.empty:
            return print(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://financialmodelingprep.com/developer/docs/pricing/jeroen/"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )

        if self._statistics_statement.empty or overwrite:
            (
                self._balance_sheet_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                tickers=self._tickers,
                statement="balance",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                statement_format=self._balance_sheet_statement_generic,
                statistics_format=self._statistics_statement_generic,
                sleep_timer=self._sleep_timer,
                progress_bar=self._progress_bar,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1:
            return self._statistics_statement.loc[self._tickers[0]]

        return self._statistics_statement

    def get_normalization_files(self, path: str = ""):
        """
        Copies the normalization files to a folder based on path. By default, this is the path
        of the 'Downloads' folder.

        Args:
            path (str, optional): The path where to save the files to.

        Returns:
            Three csv files saved to the desired location.
        """
        if path:
            _copy_normalization_files(path)
        else:
            _copy_normalization_files()
