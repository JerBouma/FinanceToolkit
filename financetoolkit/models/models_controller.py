"""Models Module"""

__docformat__ = "google"

from collections.abc import Callable
from functools import cached_property

import pandas as pd

from financetoolkit.helpers import LazyDict, check_trailing, trailing_sum
from financetoolkit.models import helpers
from financetoolkit.models.constraints.constraints_controller import ConstraintsModels
from financetoolkit.models.distress.distress_controller import DistressModels
from financetoolkit.models.profitability.profitability_controller import (
    ProfitabilityModels,
)
from financetoolkit.models.quality.quality_controller import QualityModels
from financetoolkit.models.valuation import eva_model, wacc_model
from financetoolkit.models.valuation.valuation_controller import ValuationModels
from financetoolkit.ratios import profitability_model, valuation_model
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import concat_frames, filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding, finalize_dataset

logger = get_logger()


@check_trailing
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
        analyst_estimates: pd.DataFrame | Callable[[], pd.DataFrame] | None = None,
        profile: pd.DataFrame | Callable[[], pd.DataFrame] | None = None,
        corporate_bond_yield_function: Callable[[str, str], pd.Series] | None = None,
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
                the periods as columns (the filing days with point_in_time). The market values
                (share prices, equity volatility, risk free rate) of each period are taken on
                that day and get_factor_analysis applies each period's model values from it on.
                Defaults to None, which uses the end of the calendar period the column is
                labelled with.
            sp500_function (Callable | None, optional): Retrieves the S&P 500 between two
                dates for get_campbell_hilscher_szilagyi_score: its daily price and total
                return index (the columns "Price Index" and "Total Return Index", with a
                DatetimeIndex) and the latest total market value of its companies (a Series
                with the date as index). Defaults to None, which leaves the excess returns
                and the relative size NaN.
            corporate_bond_yield_function (Callable | None, optional): Retrieves the daily
                yield of Moody's seasoned Aaa corporate bonds between two dates, as decimals
                with a DatetimeIndex or PeriodIndex, the bond yield of
                get_graham_intrinsic_value. Defaults to None, which requires that method to
                be given a bond_yield.
            profile (pd.DataFrame | Callable | None, optional): The company profiles, with
                the tickers as columns and the "IPO Date" and "Industry" rows that
                get_size_age_index and get_whited_wu_index use, or a function that retrieves
                them when a model first needs them. Defaults to None, which leaves the firm
                age and the industry averages NaN.
            analyst_estimates (pd.DataFrame | Callable[[], pd.DataFrame] | None, optional): The
                analyst consensus estimates per ticker and estimate row (index) and fiscal period
                (columns), as the forward ratios of the Ratios module use them, for
                get_implied_cost_of_capital. Defaults to None, which leaves it NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY",
                        end_date="2025-12-31",
        quarterly=True, start_date='2022-12-31')

        dupont_analysis = toolkit.models.get_dupont_analysis(method="extended")

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
        self._analyst_estimates_source = analyst_estimates
        self._profile_source = profile
        self._corporate_bond_yield_function = corporate_bond_yield_function

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
        """The fiscal period end dates, retrieved when a model first needs them."""
        source = self._period_end_dates_source
        return source() if callable(source) else source

    @cached_property
    def _analyst_estimates(self) -> pd.DataFrame:
        """The analyst consensus estimates, retrieved when a model first needs them."""
        source = self._analyst_estimates_source
        estimates = source() if callable(source) else source

        return pd.DataFrame() if estimates is None else estimates

    @cached_property
    def _profile(self) -> pd.DataFrame | None:
        """The company profiles, retrieved when a model first needs them."""
        source = self._profile_source
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
            trailing (int | None, optional): The number of periods to sum. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to
                None, which is a single period.

        Returns:
            pd.DataFrame: The EBIT per ticker (rows) and period (columns).
        """
        operating_income = self._income_statement.loc[:, "Operating Income", :]

        return trailing_sum(operating_income, trailing)

    def _net_operating_profit_after_taxes(
        self, trailing: int | None = None
    ) -> pd.DataFrame:
        """
        The Net Operating Profit After Taxes (NOPAT) the models use: EBIT (the reported
        Operating Income) times one minus the effective tax rate, the rate kept within 0 and
        1 and 0 without a pre-tax profit, as in the Economic Value Added and the WACC.

        Args:
            trailing (int | None, optional): The number of periods to sum. Defaults to None,
                which is a single period.

        Returns:
            pd.DataFrame: The NOPAT per ticker (rows) and period (columns).
        """
        effective_tax_rate = wacc_model.get_corporate_tax_rate(
            income_tax_expense=trailing_sum(
                self._income_statement.loc[:, "Income Tax Expense", :], trailing
            ),
            income_before_tax=trailing_sum(
                self._income_statement.loc[:, "Income Before Tax", :], trailing
            ),
        )
        return eva_model.get_net_operating_profit_after_taxes(
            ebit=self._earnings_before_interest_and_taxes(trailing),
            effective_tax_rate=effective_tax_rate,
        )

    def _annual_trailing(self, trailing: int | None) -> tuple[int | None, int]:
        """
        The trailing window of a measure that is annual by nature and the number of periods
        in a year: a trailing year (4) on quarterly data unless `trailing` says otherwise, so
        that comparisons with a year earlier are year over year.
        """
        periods_per_year = 4 if self._quarterly else 1
        if trailing is None and self._quarterly:
            trailing = 4
        return trailing, periods_per_year

    def _return_on_equity_and_payout(
        self, trailing: int | None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        The Return on Equity on the equity at the beginning of the period (`trailing` periods
        back with `trailing` set, the flows then summed over `trailing` periods) and the
        dividend payout ratio, per ticker and period, as the sustainable growth rate and the
        justified multiples use them.
        """
        net_income = trailing_sum(
            self._income_statement.loc[:, "Net Income", :], trailing
        )
        # Higgins measures the Return on Equity on the equity at the beginning of the period.
        # The shareholders' equity (without minority interests), as the Net Income is the income
        # attributable to the shareholders.
        beginning_total_equity = self._balance_sheet_statement.loc[
            :, "Total Shareholder Equity", :
        ].shift(trailing or 1, axis=1)
        dividends = trailing_sum(
            self._cash_flow_statement.loc[:, "Dividends Paid", :], trailing
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

    def _market_dates(self) -> pd.DataFrame:
        """
        The day the market data of every statement period is taken on, per ticker (rows) and
        period (columns): the end of the fiscal period (the filing day with point_in_time),
        and where that is not known the end of the calendar period the column is labelled
        with. A fiscal year ending in January through May is labelled with the calendar year
        before, so the label's end lies before the end of the fiscal year and a market value
        taken there would mix the balance sheet with a price from before its date.
        """
        columns = self._balance_sheet_statement.columns
        label_ends = pd.Series(
            columns.to_timestamp(how="end").normalize(), index=columns
        )
        dates = self._period_end_dates
        if dates is None or dates.empty:
            dates = pd.DataFrame(index=self._tickers, columns=columns)

        return (
            dates.reindex(index=self._tickers, columns=columns)
            .apply(pd.to_datetime)
            .fillna(label_ends)
        )

    def _values_at_market_dates(self, daily: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """
        Daily values on (or on the last day before) each ticker's market date of every
        statement period (see _market_dates), per ticker (rows) and period (columns).

        Args:
            daily (pd.DataFrame | pd.Series): The daily values with a PeriodIndex or
                DatetimeIndex, per ticker (columns), or one Series that holds for every
                ticker (such as the risk free rate).

        Returns:
            pd.DataFrame: The values per ticker (rows) and period (columns), NaN where no
            value lies within 10 days before the date.
        """
        if isinstance(daily, pd.Series):
            daily = pd.DataFrame({ticker: daily for ticker in self._tickers})
        daily = daily.reindex(columns=self._tickers)
        if isinstance(daily.index, pd.PeriodIndex):
            daily.index = daily.index.to_timestamp()

        return helpers.values_on_or_before(daily, self._market_dates())

    def _period_prices(self) -> pd.DataFrame:
        """
        The share price at the end of every statement period, per ticker, as traded (adjusted
        for splits, not for dividends): a market value is the shares outstanding at the
        price paid then, which the dividend-adjusted price understates for past periods.

        The price is the close on the last trading day up to the end of the fiscal period
        (the filing day with point_in_time, see _market_dates), the date of the balance sheet
        it is combined with. Where the daily prices do not reach that day, the close of the
        calendar period the column is labelled with is used.
        """
        columns = self._balance_sheet_statement.columns
        prices = self._historical_data["quarterly" if self._quarterly else "yearly"][
            "Close"
        ]
        prices = prices.reindex(columns=self._tickers)
        prices.index = prices.index.asfreq(columns.freq)
        label_prices = prices.reindex(columns).T

        fiscal_prices = self._values_at_market_dates(
            self._historical_data["daily"]["Close"]
        )

        return fiscal_prices.combine_first(label_prices).reindex(
            index=self._tickers, columns=columns
        )

    def _finalize_components(
        self,
        components: dict[str, pd.DataFrame],
        rounding: int | None,
        growth: bool,
        lag: int | list[int],
        standardize: bool,
    ) -> pd.DataFrame:
        """
        A model's components per ticker and period as ticker and component (rows) by period
        (columns), without the periods none of them has, finalized as every model result.
        """
        combined = (
            concat_frames(components)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )
        return finalize_dataset(
            dataset=combined,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
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
        a warning says how many periods `start_date` must cover. With a name, a result that
        is empty because none of its windows is complete (a window longer than the history)
        is also reported.
        """
        # Ticker and metric (rows) by period (columns), the metrics in the order of `results`.
        order = {metric: position for position, metric in enumerate(results)}
        combined = (
            concat_frames(results)
            .swaplevel(0, 1)
            .sort_index(
                level=1, key=lambda names: names.map(order), sort_remaining=False
            )
            .reindex(self._tickers, level=0)
        )
        unit = "quarters" if self._quarterly else "years"
        available = len(combined.columns)
        if name and available <= first_columns:
            logger.warning(
                "%s needs at least %s %s of statements and %s were available, so it could "
                "not be calculated. Set an earlier start_date to obtain it.",
                name,
                first_columns + 1,
                unit,
                available,
            )
        combined = combined[combined.columns[first_columns:]].dropna(axis=1, how="all")
        if name and available > first_columns and combined.empty:
            logger.warning(
                "%s could not be calculated from the %s %s of statements available, as none "
                "of its windows is complete. Set an earlier start_date (or a smaller window) "
                "to obtain it.",
                name,
                available,
                unit,
            )
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
        "get_shumway_score",
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


@check_trailing
@filter_shown_columns
@track_sources
class Models(
    ValuationModels,
    QualityModels,
    DistressModels,
    ProfitabilityModels,
    ConstraintsModels,
    ModelsBase,
):
    """
    The Models module is meant to execute well-known models such as DUPONT and the
    Discounted Cash Flow (DCF) model.

    These models are also directly related to the data retrieved from the Toolkit module.

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
        group: str | dict | None = None,
        demeaned: bool = False,
        binning: str = "quantile",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Factor analysis (predictive power) of the models.

        Whether the tickers that score high on a model, such as the Piotroski F-Score or the
        Altman Z-Score, go on to return more than those that score low. Per period, the
        information coefficient (IC) is the Spearman rank correlation between the model score of
        the tickers at the end of the period and their return over the following horizon: 1
        means the score ranks the later returns perfectly, -1 in reverse and 0 that it says
        nothing about them. The IC measures skill in Grinold and Kahn's fundamental law of
        active management, IR ≈ IC × √breadth: a small but consistent IC across many tickers and
        periods gives a high information ratio.

        The summary has a row per model and horizon with the mean IC, its standard deviation, the IC
        information ratio (mean divided by standard deviation), the t-statistic of the mean with its
        p-value, the skewness and excess kurtosis of the IC, the share of periods with a positive
        IC, the number of periods, the top minus bottom spread (the mean forward return of the
        highest quantile minus that of the lowest), the annualized alpha and the beta of the
        factor-weighted long-short portfolio against the equally weighted tickers, the mean turnover
        of the top and bottom quantiles and the mean rank autocorrelation of the score from one
        period to the next. With detail, it returns one part of the analysis per period, quantile or
        group for a single model.

        The formula is as follows:

        - IC(t) = Spearman rank correlation of f(i, t) and R(i, t, t + h) over the tickers i,
          with f the factor and R the forward return over the horizon h
        - Demeaned return: R(i, t) − mean R(j, t) over the tickers j (of the group of i)
        - Portfolio weight: w(i, t) = (f(i, t) − mean f(t)) / Σ_j |f(j, t) − mean f(t)|
        - Portfolio return: R_F(t) = Σ_i w(i, t) × R(i, t, t + h)
        - Alpha and beta: R_F(t) = α + β × R_U(t) + ε(t), with R_U(t) = mean R(i, t, t + h)
        - Annualized Alpha = (1 + α) ^ (periods per year / h) − 1
        - Per period: (1 + mean return) ^ (1 / h) − 1 and standard error / √h
        - Standard error of a quantile = standard deviation over the periods / √periods

        The models are calculated per statement period and each period's score applies from
        the end of the fiscal period, or with point_in_time from the day it was filed, never
        earlier, so no day uses figures that were not known yet. The returns use the dividend
        and split adjusted close and start at the close of the period the score belongs to.
        Each ticker keeps its own history: a ticker without a score or a return in a period is
        left out of that period only.

        Shares similar logic:
        - alphalens, factor_information_coefficient: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L28
        - alphalens, mean_return_by_quantile: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L461
        - alphalens, factor_weights: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L131
        - alphalens, factor_returns: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L210
        - alphalens, quantile_turnover: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L575
        - alphalens, factor_rank_autocorrelation: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L620
        - alphalens, average_cumulative_return_by_quantile (event returns): https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L758

        Args:
            metrics (str | list[str], optional): A model method name with one value per
                ticker and period, e.g. "get_piotroski_score", a list of them, or a
                collection: "scores" or "all". Defaults to "scores", the Piotroski, Altman Z,
                Beneish M, Ohlson O, Zmijewski, Springate, Grover, Fulmer H, Taffler z,
                Campbell-Hilscher-Szilagyi, Shumway, Dechow F, Montier C and Mohanram G
                scores. "all" adds the accruals ratio, Tobin's Q, the sustainable and internal
                growth rates, the magic formula and the Merton distance to default. The
                valuation models and the analyses with several components and no score are not
                in the collections.
            period (str, optional): The frequency at which the models are evaluated. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            quantiles (int, optional): The number of quantiles the tickers are sorted into per
                period. Defaults to 5.
            periods_apart (int, optional): The number of periods between the dates the
                quantile turnover and the rank autocorrelation compare. Defaults to 1.
            detail (str | None, optional): Returns one part of the analysis of a single model instead of the
                summary. Options are: "information_coefficient" (per period and horizon),
                "information_coefficient_by_month" (the mean IC per calendar month),
                "information_coefficient_by_year" (the mean IC per calendar year), "group_information_coefficient"
                (the IC statistics within each group, needs group), "quantile_returns" (the mean forward return per
                quantile and horizon), "quantile_returns_by_date" (per period), "quantile_return_statistics" (the
                mean, its standard error and both per single period), "quantile_statistics" (the range, mean and
                count of the factor values in each quantile), "quantile_turnover" (per period and quantile),
                "rank_autocorrelation" (per period), "factor_weights" (the weights of the factor-weighted long-short
                portfolio per period and ticker), "factor_returns" (its return per period and horizon),
                "cumulative_returns" (the growth of 1 in it and in each quantile), "event_returns" (the mean price
                path of each quantile from the longest horizon before to the longest horizon after the periods it is
                formed). Defaults to None.
            column (str | None, optional): The component to use of a model, e.g.
                "Working Capital to Total Assets" of the Altman Z-Score. Defaults to None,
                which uses the model's score (its last row).
            group (str | dict | None, optional): The group of each ticker for the group options:
                "sector" or "industry" (or another company profile item) from the company profiles
                of Financial Modeling Prep, or a dictionary of tickers and their groups. Defaults to
                None.
            demeaned (bool, optional): Whether to compare the forward returns with the mean of
                the tickers in each period, or with group with the mean of the ticker's group:
                a group-neutral analysis, in which the IC is group-adjusted and the
                factor-weighted portfolio neutral within each group. Without group the IC
                stays the same, as subtracting one mean from every ticker keeps their ranks.
                Defaults to False.
            binning (str, optional): How the tickers are sorted into quantiles per period. Options are: "quantile"
                (equally sized), "equal_width" (bins of equal width between the lowest and highest value),
                "zero_aware" (half of the quantiles for the negative and half for the zero and positive values, for
                a signal centred on zero; needs an even number of quantiles), "group" (equally sized within each
                group, needs group). Defaults to "quantile".
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
            - The groups from the company profiles are today's sector or industry, applied
              to every period.
            - The factor-weighted portfolio is long-short: its weights sum to zero and their
              absolute values to one. A horizon longer than one period makes its returns
              overlap too, so its alpha and beta are averages rather than tests.
            - The cumulative returns split the capital over as many portfolios as the horizon
              has periods, started on consecutive periods and each rebalanced once per
              horizon; a value shows when its holding period ends, so the series starts at 1.
            - The IC is that of Grinold, R.C. and Kahn, R.N. (2000). "Active Portfolio
              Management." 2nd ed. McGraw-Hill; the factor-weighted portfolio, the demeaned
              and group-neutral returns that of Qian, E., Hua, R. and Sorensen, E. (2007).
              "Quantitative Equity Portfolio Management." Chapman & Hall/CRC; the alpha that
              of Jensen, M.C. (1968). "The Performance of Mutual Funds in the Period
              1945-1964." Journal of Finance 23(2); the event returns follow MacKinlay, A.C.
              (1997). "Event Studies in Economics and Finance." Journal of Economic Literature
              35(1).
            - Unlike alphalens, whose top minus bottom spread is half of the one here (a gross
              exposure of one rather than two), each horizon here uses its own returns where
              alphalens drops a period and ticker without every horizon's return, and the event
              returns are the price paths rebased to the period itself where alphalens averages
              the single-period returns at each offset.

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

        |                                      |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   P-Value |   IC Skewness |   IC Kurtosis |   Positive Share |   Observations |   Top Minus Bottom |   Annualized Alpha |    Beta |   Quantile Turnover |   Rank Autocorrelation |
        |:-------------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|--------------:|--------------:|-----------------:|---------------:|-------------------:|-------------------:|--------:|--------------------:|-----------------------:|
        | ('get_piotroski_score', '12 Months') |   -0.1353 |                  0.3685 |                -0.3672 |       -3.6724 |    0.0004 |        0.3723 |       -0.1763 |             0.31 |            100 |            -0.1482 |             0.0279 | -0.3828 |              0.0787 |                 0.93   |
        | ('get_altman_z_score', '12 Months')  |    0.1179 |                  0.3915 |                 0.3011 |        3.0112 |    0.0033 |       -0.657  |       -0.2792 |             0.66 |            100 |             0.1564 |            -0.1619 |  1.2913 |              0.0258 |                 0.9939 |
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
            returns=self._historical_data["daily"].get("Return"),
            binning=binning,
            demeaned=demeaned,
            groups=factor_model.get_groups(group, self._profile_source, self._tickers),
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
