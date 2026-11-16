"""Models Module"""

__docformat__ = "google"

from collections.abc import Callable
from functools import cached_property

import pandas as pd

from financetoolkit.helpers import LazyDict
from financetoolkit.models import helpers
from financetoolkit.models.distress.distress_controller import DistressModels
from financetoolkit.models.profitability.profitability_controller import (
    ProfitabilityModels,
)
from financetoolkit.models.quality.quality_controller import QualityModels
from financetoolkit.models.valuation.valuation_controller import ValuationModels
from financetoolkit.ratios import profitability_model, valuation_model
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import concat_frames, filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

logger = get_logger()


@filter_shown_columns
@track_sources
class ModelsBase:
    """
    The data every model topic shares: the statements, prices, rates and the helpers several topics use.
    """

    def __init__(
        self,
        tickers: list[str] | Callable[[], list[str]],
        historical_data: dict[str, pd.DataFrame],
        risk_free_rate_data: dict[str, pd.DataFrame],
        balance: pd.DataFrame | Callable[[], pd.DataFrame],
        income: pd.DataFrame | Callable[[], pd.DataFrame],
        cash: pd.DataFrame | Callable[[], pd.DataFrame],
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
        peer_function: (
            Callable[[dict[str, list[str]] | None], pd.DataFrame] | None
        ) = None,
        country_risk_premiums: pd.Series | Callable[[], pd.Series] | None = None,
        period_end_dates: pd.DataFrame | Callable[[], pd.DataFrame] | None = None,
        sp500_function: (
            Callable[[str, str], tuple[pd.DataFrame, pd.Series]] | None
        ) = None,
    ):
        """
        Initializes the Models Controller Class.

        The tickers and statements are either the data itself or a function without arguments
        that retrieves it, and the period dictionaries can be a LazyDict (financetoolkit.helpers)
        that retrieves a period when it is first read. The Toolkit passes these, so a model
        retrieves only the data it uses, when it is calculated.

        Args:
            tickers (list[str] | Callable[[], list[str]]): The ticker(s) to use for the models.
            historical_data (dict[str, pd.DataFrame]): The historical data per period, and
                under "lookback" the daily data from a year before the start date when
                available.
            risk_free_rate_data (dict[str, pd.DataFrame]): The risk free rate data per period.
            balance (pd.DataFrame | Callable[[], pd.DataFrame]): The balance sheet data.
            income (pd.DataFrame | Callable[[], pd.DataFrame]): The income statement data.
            cash (pd.DataFrame | Callable[[], pd.DataFrame]): The cash flow statement data.
            quarterly (bool, optional): Whether to use quarterly or yearly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            peer_function (Callable | None, optional): Compares the companies with their peers,
                used by get_peer_comparison. Defaults to None.
            country_risk_premiums (pd.Series | Callable[[], pd.Series] | None, optional): The
                annual equity risk premium of each company's country (Damodaran's Total Equity
                Risk Premium), indexed by ticker, which the Cost of Equity uses unless a
                market_risk_premium is given. Defaults to None, which uses the benchmark's
                average return above the risk-free rate for every company.
            period_end_dates (pd.DataFrame | Callable[[], pd.DataFrame], optional): The last day
                of the fiscal period behind each statement column, with the tickers as index and
                the periods as columns (the filing days with point_in_time). get_factor_analysis
                applies each period's model values from that day on. Defaults to None, which
                applies them from the end of the calendar period the column is labelled with.
            sp500_function (Callable | None, optional): Retrieves the S&P 500 between two
                dates for get_campbell_hilscher_szilagyi_score: its daily price and total
                return index (the columns "Price Index" and "Total Return Index", with a
                DatetimeIndex) and the latest total market value of its companies (a Series
                with the date as index). Defaults to None, which leaves the excess returns
                and the relative size NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY",
                        end_date="2025-12-31",
        quarterly=True, start_date='2022-12-31')

        dupont_analysis = toolkit.models.get_extended_dupont_analysis()

        dupont_analysis.loc['AMZN']
        ```

        Which returns:

        |                         |   2022Q4 |   2023Q1 |   2023Q2 |
        |:------------------------|---------:|---------:|---------:|
        | Interest Burden Ratio   |  -0.3467 |   0.863  |   0.9835 |
        | Tax Burden Ratio        |  -0.2929 |   0.7699 |   0.8936 |
        | Operating Profit Margin |   0.0183 |   0.0375 |   0.0572 |
        | Asset Turnover          |   0.3349 |   0.2748 |   0.2853 |
        | Equity Multiplier       |   3.1426 |   3.0843 |   2.9152 |
        | Return on Equity        |   0.002  |   0.0211 |   0.0418 |
        """
        self._tickers_source = tickers
        self._benchmark_name = "Benchmark"
        self._balance_source = balance
        self._income_source = income
        self._cash_source = cash
        self._quarterly = quarterly
        self._rounding = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._peer_function = peer_function
        self._country_risk_premiums_source = country_risk_premiums
        self._period_end_dates_source = period_end_dates
        self._sp500_function = sp500_function

        # Historical Data
        self._historical_data = historical_data
        self._risk_free_rate_data = risk_free_rate_data

        # The daily returns within each period, built when a model first asks for a period.
        self._within_historical_data = LazyDict(
            lambda period: helpers.determine_within_historical_data(
                period, self._historical_data["daily"]
            )
        )

    @cached_property
    def _tickers(self) -> list[str]:
        """The tickers, retrieved when first needed."""
        return (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

    @cached_property
    def _peer_tickers(self) -> list[str]:
        """
        The tickers the scores that rank or compare the tickers with each other use: the
        Total of a basket is not a peer of the tickers it adds up.
        """
        return [ticker for ticker in self._tickers if ticker != "Total"]

    @cached_property
    def _period_end_dates(self) -> pd.DataFrame | None:
        """The fiscal period end dates, retrieved when get_factor_analysis first needs them."""
        source = self._period_end_dates_source
        return source() if callable(source) else source

    @cached_property
    def _country_risk_premiums(self) -> pd.Series:
        """The equity risk premium of each company's country, retrieved when first needed."""
        source = self._country_risk_premiums_source
        premiums = source() if callable(source) else source

        if premiums is None:
            return pd.Series(index=self._tickers, dtype=float)

        return premiums.reindex(self._tickers).astype(float)

    @cached_property
    def _balance_sheet_statement(self) -> pd.DataFrame:
        """The balance sheet statement, retrieved when a model first needs it."""
        return (
            self._balance_source()
            if callable(self._balance_source)
            else self._balance_source
        )

    @cached_property
    def _income_statement(self) -> pd.DataFrame:
        """The income statement, retrieved when a model first needs it."""
        return (
            self._income_source()
            if callable(self._income_source)
            else self._income_source
        )

    @cached_property
    def _cash_flow_statement(self) -> pd.DataFrame:
        """The cash flow statement, retrieved when a model first needs it."""
        return self._cash_source() if callable(self._cash_source) else self._cash_source

    def _earnings_before_interest_and_taxes(
        self, trailing: int | None = None
    ) -> pd.DataFrame:
        """
        The Earnings Before Interest and Taxes (EBIT) the models use, taken as the reported
        Operating Income, consistent with the EBIT based ratios of the Ratios module.

        Args:
            trailing (int | None, optional): The number of periods to sum. Defaults to None,
                which is a single period.

        Returns:
            pd.DataFrame: The EBIT per ticker (rows) and period (columns).
        """
        operating_income = self._income_statement.loc[:, "Operating Income", :]

        return (
            operating_income.T.rolling(trailing).sum().T
            if trailing
            else operating_income
        )

    def _return_on_equity_and_payout(
        self, trailing: int | None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        The Return on Equity on the equity at the beginning of the period (`trailing` periods
        back with `trailing` set, the flows then summed over `trailing` periods) and the
        dividend payout ratio, per ticker and period, as the sustainable growth rate and the
        justified multiples use them.
        """
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        # Higgins measures the Return on Equity on the equity at the beginning of the period.
        beginning_total_equity = self._balance_sheet_statement.loc[
            :, "Total Equity", :
        ].shift(trailing if trailing else 1, axis=1)
        dividends = (
            self._cash_flow_statement.loc[:, "Dividends Paid", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Dividends Paid", :]
        )

        return_on_equity = profitability_model.get_return_on_equity(
            net_income=net_income,
            average_total_equity=beginning_total_equity,
        )
        dividend_payout_ratio = valuation_model.get_dividend_payout_ratio(
            dividends=dividends,
            net_income=net_income,
        )
        return return_on_equity, dividend_payout_ratio

    def _rows(self, statement: pd.DataFrame, row: str) -> pd.DataFrame:
        """A statement row per ticker (rows) and period (columns); NaN when not reported."""
        if row not in statement.index.get_level_values(1):
            return pd.DataFrame(
                float("nan"), index=self._tickers, columns=statement.columns
            )
        return (
            statement.loc[:, row, :]
            .reindex(self._tickers)
            .apply(pd.to_numeric, errors="coerce")
        )

    def _period_prices(self) -> pd.DataFrame:
        """
        The share price at the end of every statement period, per ticker, as traded (adjusted
        for splits, not for dividends): a market value is the shares outstanding at the
        price paid then, which the dividend-adjusted price understates for past periods.
        """
        columns = self._balance_sheet_statement.columns
        prices = self._historical_data["quarterly" if self._quarterly else "yearly"][
            "Close"
        ]
        prices = prices.reindex(columns=self._tickers)
        prices.index = prices.index.asfreq(columns.freq)
        return prices.reindex(columns).T

    def _combine(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Per-period results as ticker and metric (rows) by period (columns)."""
        combined = concat_frames(results).swaplevel(0, 1)
        order = {name: position for position, name in enumerate(results)}
        combined = combined.sort_index(
            level=1, key=lambda names: names.map(order), sort_remaining=False
        )
        return combined.reindex(self._tickers, level=0)

    def _calendar(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        A ticker by period frame on every period from the first to the last, so that a
        period no statement reports is a NaN column and positions and rolling windows count
        calendar periods rather than reported ones.
        """
        columns = frame.columns
        if not isinstance(columns, pd.PeriodIndex) or columns.empty:
            return frame
        return frame.reindex(
            columns=pd.period_range(columns.min(), columns.max(), freq=columns.freq)
        )

    def _finish_score(
        self,
        results: dict[str, pd.DataFrame],
        rounding: int | None,
        first_columns: int = 1,
        name: str | None = None,
    ) -> pd.DataFrame:
        """
        Combine score components, drop the periods without a comparison and round. When the
        statements hold no more periods than the comparison needs, the result is empty, and
        a warning says how many periods `start_date` must cover.
        """
        combined = self._combine(results)
        if name and len(combined.columns) <= first_columns:
            logger.warning(
                "%s needs at least %s %s of statements and %s were available, so it could "
                "not be calculated. Set an earlier start_date to obtain it.",
                name,
                first_columns + 1,
                "quarters" if self._quarterly else "years",
                len(combined.columns),
            )
        combined = combined[combined.columns[first_columns:]].dropna(axis=1, how="all")
        combined = apply_rounding(
            combined, rounding if rounding is not None else self._rounding
        )
        return combined.loc[:, self._start_date : self._end_date]


# ruff: noqa: E501

# The models per collection that get_factor_analysis compares across tickers: those with one
# value per ticker and statement period. The valuation models (intrinsic value, dividend
# discount, residual income) give a currency amount whose level reflects the share price, and
# the analyses (DuPont, WACC, enterprise value breakdown) a set of components without one
# score, so they are not in the collections.
FACTOR_COLLECTIONS = {
    "scores": [
        "get_piotroski_score",
        "get_altman_z_score",
        "get_beneish_m_score",
        "get_ohlson_o_score",
        "get_zmijewski_score",
        "get_springate_score",
        "get_grover_score",
        "get_fulmer_h_score",
        "get_taffler_z_score",
        "get_campbell_hilscher_szilagyi_score",
        "get_dechow_f_score",
        "get_montier_c_score",
        "get_mohanram_g_score",
    ],
}

FACTOR_COLLECTIONS["all"] = FACTOR_COLLECTIONS["scores"] + [
    "get_accruals_ratio",
    "get_tobins_q_ratio",
    "get_sustainable_growth_rate",
    "get_internal_growth_rate",
    "get_magic_formula",
    "get_merton_distance_to_default",
]


@filter_shown_columns
@track_sources
class Models(
    ValuationModels, QualityModels, DistressModels, ProfitabilityModels, ModelsBase
):
    """
    The Models module is meant to execute well-known models such
    as DUPONT and the Discounted Cash Flow (DCF) model. These models
    are also directly related to the data retrieved from the Toolkit module.

    """

    @handle_errors
    def get_factor_analysis(
        self,
        metrics: str | list[str] = "scores",
        period: str = "monthly",
        horizons: list[int] | int | None = None,
        quantiles: int = 5,
        periods_apart: int = 1,
        detail: str | None = None,
        column: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Factor analysis (predictive power) of the models: whether the tickers that score high
        on a model, such as the Piotroski F-Score or the Altman Z-Score, go on to return more
        than those that score low. Per period, the information coefficient (IC) is the
        Spearman rank correlation between the model score of the tickers at the end of the
        period and their return over the following horizon: 1 means the score ranks the later
        returns perfectly, -1 in reverse and 0 that it says nothing about them. The IC measures
        skill in Grinold and Kahn's fundamental law of active management,
        IR ≈ IC × √breadth: a small but consistent IC across many tickers and periods gives a
        high information ratio.

        The summary has a row per model and horizon with the mean IC, its standard deviation,
        the IC information ratio (mean divided by standard deviation), the t-statistic of the
        mean with its p-value, the share of periods with a positive IC, the number of periods,
        the top minus bottom spread (the mean forward return of the highest quantile minus
        that of the lowest), the mean turnover of the top and bottom quantiles and the mean
        rank autocorrelation of the score from one period to the next. With detail, it returns
        one of those parts per period (or per quantile) for a single model.

        The models are calculated per statement period and each period's score applies from
        the end of the fiscal period, or with point_in_time from the day it was filed, never
        earlier, so no day uses figures that were not known yet. The returns use the dividend
        and split adjusted close and start at the close of the period the score belongs to.
        Each ticker keeps its own history: a ticker without a score or a return in a period is
        left out of that period only.

        Args:
            metrics (str | list[str], optional): A model method name with one value per
                ticker and period, e.g. "get_piotroski_score", a list of them, or a
                collection: "scores" or "all". Defaults to "scores", the Piotroski, Altman Z,
                Beneish M, Ohlson O, Zmijewski, Springate, Grover, Fulmer H, Taffler z,
                Dechow F, Montier C and Mohanram G scores. "all" adds the accruals ratio, Tobin's Q, the
                sustainable and internal growth rates, the magic formula and the Merton
                distance to default. The valuation models and the analyses with several
                components and no score are not in the collections.
            period (str, optional): The frequency at which the models are evaluated: daily,
                weekly, monthly, quarterly or yearly. Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            quantiles (int, optional): The number of quantiles the tickers are sorted into per
                period. Defaults to 5.
            periods_apart (int, optional): The number of periods between the dates the
                quantile turnover and the rank autocorrelation compare. Defaults to 1.
            detail (str | None, optional): Returns one part of the analysis of a single model
                instead of the summary: "information_coefficient" (per period and horizon),
                "quantile_returns" (the mean forward return per quantile and horizon),
                "quantile_turnover" (per period and quantile) or "rank_autocorrelation" (per
                period). Defaults to None.
            column (str | None, optional): The component to use of a model, e.g.
                "Working Capital to Total Assets" of the Altman Z-Score. Defaults to None,
                which uses the model's score (its last row).
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            **kwargs: Parameters passed to the models that take them, e.g. trailing=4 or
                diluted=False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The summary per model and horizon, or with detail that part of the
            analysis of the model.

        Notes:
            - The IC is reported as measured: for a distress score where a higher value means
              more risk (Ohlson O, Zmijewski, Beneish M), a negative IC means the riskier
              companies went on to return less. Read the sign as the direction of the
              relationship.
            - Scores that take few values (the Piotroski F-Score from 0 to 9) tie many
              tickers; ties are split over adjacent quantiles by the order of the tickers.
            - The cross-section is the tickers of the Toolkit, so use at least ten and
              preferably twenty or more for a meaningful rank correlation and quantiles.
            - A horizon longer than one period makes consecutive returns overlap, which
              inflates the t-statistic; judge those horizons by their mean IC and IC
              information ratio.
            - Based on alphalens (Quantopian) and on Grinold and Kahn (2000), Active Portfolio
              Management.

        Also known as: predictive power, information coefficient, factor testing, alpha signal

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ",
             "XOM", "PG", "KO", "WMT", "V", "UNH", "HD", "CVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_factor_analysis(
            ["get_piotroski_score", "get_altman_z_score"], horizons=12
        )
        ```

        Which returns:
        |                                      |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   p-Value |   Positive Share |   Observations |   Top Minus Bottom |   Quantile Turnover |   Rank Autocorrelation |
        |:-------------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|-----------------:|---------------:|-------------------:|--------------------:|-----------------------:|
        | ('get_piotroski_score', '12 Months') |   -0.1352 |                  0.3685 |                -0.3669 |       -3.6693 |    0.0004 |             0.31 |            100 |            -0.1482 |              0.0787 |                 0.93   |
        | ('get_altman_z_score', '12 Months')  |    0.1209 |                  0.3923 |                 0.3082 |        3.0824 |    0.0027 |             0.7  |            100 |             0.1264 |              0.0285 |                 0.9941 |
        """
        factors = factor_model.get_daily_factors(
            self,
            factor_model.select_metrics(metrics, FACTOR_COLLECTIONS),
            kwargs,
            {},
            self._tickers,
            column,
            self._period_end_dates,
        )
        result = factor_model.get_factor_analysis(
            factors,
            self._historical_data["daily"]["Adj Close"],
            period,
            horizons,
            quantiles,
            periods_apart,
            detail,
            self._start_date,
            self._end_date,
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
