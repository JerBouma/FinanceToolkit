"""Portfolio Construction Module"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.portfolio.construction.analysis.analysis_controller import (
    AnalysisConstruction,
)
from financetoolkit.portfolio.construction.attribution.attribution_controller import (
    AttributionConstruction,
)
from financetoolkit.portfolio.construction.optimization import optimization_model
from financetoolkit.portfolio.construction.optimization.optimization_controller import (
    OptimizationConstruction,
)
from financetoolkit.portfolio.helpers import BENCHMARK_TICKER, PORTFOLIO_TICKER
from financetoolkit.risk.helpers import trades_on_weekends
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import check_period
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR, apply_rounding

logger = get_logger()


PERIODS = ["daily", "weekly", "monthly", "quarterly", "yearly"]


# How far weights may sum from one before they are reported as scaled.
WEIGHT_SUM_TOLERANCE = 1e-6


@filter_shown_columns
@track_sources
class PortfolioConstructionBase:
    """
    The data every portfolio construction topic shares: the assets, returns, risk-free rate, benchmark, current weights, constraints and the weights table.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: dict[str, pd.DataFrame] | None = None,
        risk_free_rate_data: dict[str, pd.Series] | None = None,
        benchmark_name: str | None = None,
        portfolio_weights: dict | None = None,
        price_function: Callable[[list[str], str, str], pd.DataFrame] | None = None,
        profile_function: Callable[[], pd.DataFrame] | None = None,
        fund_function: Callable[[str, list[str]], pd.DataFrame] | None = None,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Portfolio Construction Controller Class.

        Args:
            tickers (str | list[str]): The tickers of the Toolkit instance.
            historical_data (dict[str, pd.DataFrame] | None, optional): The historical data
                per period. Defaults to None.
            risk_free_rate_data (dict[str, pd.Series] | None, optional): The annualized
                risk free rate per period. Defaults to None.
            benchmark_name (str | None, optional): The ticker of the benchmark, e.g. "SPY".
                Defaults to None.
            portfolio_weights (dict | None, optional): The weights of a Portfolio over time,
                per period, used when no weights are given. Defaults to None.
            price_function (Callable | None, optional): Retrieves the daily prices of other
                tickers over a date range, for stress tests and sector returns. Defaults
                to None.
            profile_function (Callable | None, optional): Retrieves the company profiles
                (sector, country, market capitalization). Defaults to None.
            fund_function (Callable | None, optional): Retrieves the sector, country or
                holding weights of funds. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to 4.
            start_date (str | None, optional): The start date of the data. Defaults to None.
            end_date (str | None, optional): The end date of the data. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_maximum_sharpe_weights()
        ```
        """
        self._tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        self._historical_data = historical_data if historical_data is not None else {}
        self._risk_free_rate_data = (
            risk_free_rate_data if risk_free_rate_data is not None else {}
        )
        self._benchmark_name = benchmark_name
        self._portfolio_weights = portfolio_weights
        self._price_function = price_function
        self._profile_function = profile_function
        self._fund_function = fund_function
        self._rounding: int | None = rounding
        self._start_date = start_date
        self._end_date = end_date
        self._profile: pd.DataFrame | None = None
        self._factor_datasets: dict[str, pd.DataFrame] = {}

    def _get_assets(self, include_benchmark: bool = False) -> list[str]:
        """The assets to build the portfolio from, without the Portfolio pseudo-ticker."""
        assets = [
            ticker
            for ticker in self._tickers
            if ticker not in (PORTFOLIO_TICKER, BENCHMARK_TICKER)
        ]
        if include_benchmark:
            assets.append(BENCHMARK_TICKER)
        return assets

    def _get_loaded_assets(self, period: str, include_benchmark: bool) -> list[str]:
        """The assets with returns in the historical data of the period."""
        check_period(period, PERIODS)
        columns = self._historical_data[period]["Return"].columns
        return [
            asset for asset in self._get_assets(include_benchmark) if asset in columns
        ]

    def _get_returns(
        self, period: str, include_benchmark: bool, assets: list[str] | None = None
    ) -> pd.DataFrame:
        """
        The returns of the assets over the periods in which every one of them has a return.
        An optimizer chooses among all loaded assets, so all of them decide the periods; the
        weights of a given portfolio pass only the assets it holds, so an asset it leaves out
        (with a shorter history) does not shorten its sample.
        """
        if assets is None:
            assets = self._get_loaded_assets(period, include_benchmark)
            if len(assets) < 2:  # noqa: PLR2004
                raise ValueError(
                    "A portfolio needs the returns of at least two assets."
                )
        # Historical data supplied to the Toolkit can reach beyond its date range.
        returns = (
            self._historical_data[period]["Return"]
            .loc[self._start_date : self._end_date, assets]
            .dropna(how="any")
        )
        if len(returns) < len(assets) + 1:
            raise ValueError(
                f"There are only {len(returns)} periods in which every asset has a "
                "return; use a longer date range or a higher frequency."
            )
        return returns

    def _get_held_weights(
        self,
        weights: dict[str, float] | pd.Series | None,
        period: str,
        include_benchmark: bool,
    ) -> pd.Series:
        """The weights of the loaded assets the portfolio holds (a weight other than zero)."""
        resolved = self._resolve_weights(
            weights, self._get_loaded_assets(period, include_benchmark)
        )
        return resolved[resolved != 0]

    def _get_weights_over_time(
        self,
        weights: dict[str, float] | pd.Series | None,
        period: str,
        assets: list[str],
        index: pd.Index,
    ) -> pd.DataFrame:
        """
        The weight of every asset at the start of each period: the weights as given,
        rebalanced to them every period, or, without weights, those of the Portfolio the
        Toolkit came from at the end of the period before (the weights held while the
        period's return was earned), and otherwise equal weights.
        """
        portfolio_weights = (self._portfolio_weights or {}).get(period)
        if (
            weights is None
            and portfolio_weights is not None
            and not portfolio_weights.empty
        ):
            held = (
                portfolio_weights.reindex(columns=assets)
                .shift(1)
                .reindex(index)
                .astype(float)
            )
            return held.fillna(0)
        resolved = self._resolve_weights(weights, assets)
        return pd.DataFrame(
            np.tile(resolved.to_numpy(), (len(index), 1)), index=index, columns=assets
        )

    @staticmethod
    def _get_periods_per_year(period: str, index: pd.Index) -> int:
        """
        The number of periods in a year that annualizes returns of the period: 365 for
        daily returns on calendar days, which have returns on Saturdays and Sundays (such
        as those of cryptocurrencies, see trades_on_weekends), and otherwise that of
        PERIODS_PER_YEAR, 252 trading days for daily returns, as the risk module counts.
        """
        if period == "daily" and trades_on_weekends(index):
            return 365
        return PERIODS_PER_YEAR[period]

    def _get_risk_free_rate(self, period: str, index: pd.Index) -> float:
        """The average annualized risk free rate over the periods, 0 when unknown."""
        risk_free = self._risk_free_rate_data.get(period)
        risk_free_rate = (
            float(risk_free.reindex(index).mean())
            if risk_free is not None and not risk_free.empty
            else 0.0
        )
        return 0.0 if np.isnan(risk_free_rate) else risk_free_rate

    def _get_benchmark_returns(self, period: str) -> pd.Series | None:
        """The returns of the benchmark per period, None when there is no benchmark."""
        returns = self._historical_data[period]["Return"]
        return returns.get(BENCHMARK_TICKER)

    def _get_factor_returns(
        self,
        period: str,
        factors: pd.DataFrame | None = None,
        region: str = "United States",
    ) -> pd.DataFrame:
        """
        The factor returns per period: the given factors, or the Fama and French five
        factors of the region (Ken French's daily file without the risk free rate),
        compounded within each period, (Π (1 + f_d)) - 1, for the periods other than daily.
        Given factors keep the periods they share with the returns: their dates are read
        as the period they fall in.
        """
        check_period(period, PERIODS)
        index = self._historical_data[period]["Return"].index
        if factors is not None:
            table = pd.DataFrame(factors).astype(float)
            dates = (
                table.index
                if isinstance(table.index, pd.PeriodIndex)
                else pd.PeriodIndex(pd.to_datetime(table.index), freq=index.freq)
            )
            table.index = dates.asfreq(index.freq)
            return table[~table.index.duplicated(keep="last")].dropna(how="any")

        # Lazily imported: the factor files are the Performance module's data source.
        from financetoolkit.performance import performance_model  # noqa: PLC0415

        if region not in self._factor_datasets:
            self._factor_datasets[region] = (
                performance_model.obtain_fama_and_french_dataset(region=region)
            )
        daily = self._factor_datasets[region].drop(columns="RF", errors="ignore")
        if period == "daily":
            return daily.dropna(how="any")
        return (
            (1 + daily)
            .groupby(daily.index.asfreq(index.freq))
            .prod(min_count=1)
            .sub(1)
            .dropna(how="any")
        )

    def _align_factors(
        self, returns: pd.DataFrame, factors: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        The returns and the factor returns on the periods both have. Ken French publishes
        the factors about a month after the month ends, so the latest periods of the
        returns have no factors yet and are left out.
        """
        periods = returns.index.intersection(factors.index)
        if len(periods) <= factors.shape[1] + 2:  # noqa: PLR2004
            raise ValueError(
                f"The returns and the factors share only {len(periods)} periods, too few "
                f"to estimate a model on {factors.shape[1]} factors; use a longer date "
                "range or a higher frequency."
            )
        return returns.loc[periods], factors.loc[periods]

    def _get_inputs(
        self,
        period: str,
        include_benchmark: bool,
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        assets: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, float]:
        """
        The returns, annualized expected returns, covariance and risk free rate of the
        assets, by default all loaded assets (see _get_returns).
        """
        returns = self._get_returns(period, include_benchmark, assets)
        risk_free_rate = self._get_risk_free_rate(period, returns.index)
        expected, covariance_matrix = (
            optimization_model.get_expected_returns_and_covariance(
                returns,
                self._get_periods_per_year(period, returns.index),
                covariance,
                covariance_span,
                expected_returns,
                expected_returns_span,
                (
                    self._get_benchmark_returns(period)
                    if expected_returns == "capm"
                    or covariance == "ledoit_wolf_single_factor"
                    else None
                ),
                risk_free_rate,
                self._get_factor_returns(period) if covariance == "factor" else None,
            )
        )
        return returns, expected, covariance_matrix, risk_free_rate

    def _resolve_weights(
        self, weights: dict[str, float] | pd.Series | None, assets: list[str]
    ) -> pd.Series:
        """
        The weights to examine: as given, otherwise the latest weights of the Portfolio the
        Toolkit came from, otherwise equal weights. They are scaled to sum to one.
        """
        if weights is not None:
            resolved = pd.Series(weights, dtype=float)
            unknown = [asset for asset in resolved.index if asset not in assets]
            if unknown:
                raise ValueError(
                    f"The weights name assets that are not loaded: {', '.join(unknown)}."
                )
            resolved = resolved.reindex(assets).fillna(0)
        elif self._portfolio_weights:
            latest = self._portfolio_weights.get("daily", pd.DataFrame())
            latest = latest.iloc[-1] if not latest.empty else pd.Series(dtype=float)
            # A holding without a price leaves the value of the portfolio, and every weight, unknown.
            if not latest.empty and latest.isna().all():
                raise ValueError(
                    "The latest weights of the Portfolio are unknown because one of its holdings "
                    "has no price; pass the weights to use instead."
                )
            resolved = latest.reindex(assets).fillna(0).astype(float)
        else:
            # Equal weights that already sum to one, so they are not reported as rescaled.
            resolved = pd.Series(1 / max(len(assets), 1), index=assets)

        if resolved.sum() == 0:
            raise ValueError("The weights sum to zero.")
        # Scaling by a negative sum would flip every sign and turn a short portfolio long.
        if resolved.sum() < 0:
            raise ValueError(
                f"The weights sum to {resolved.sum():.4f}; a portfolio needs a positive net "
                "weight to be scaled to one."
            )
        if abs(resolved.sum() - 1) > WEIGHT_SUM_TOLERANCE:
            logger.info(
                "The weights sum to %.4f and are scaled to one.", resolved.sum()
            )
        return resolved / resolved.sum()

    def _round(self, table: pd.DataFrame, rounding: int | None) -> pd.DataFrame:
        """The table rounded to the given decimals, or to the Toolkit's rounding when None."""
        return apply_rounding(
            table, rounding if rounding is not None else self._rounding
        )

    def _get_profile(self, alternative: str = "") -> pd.DataFrame:
        """
        The company profiles, retrieved once. Without them (no API key, or none returned)
        a ValueError names what needs them and, as the alternative, the argument that
        supplies the same information; nothing is cached, so a later call tries again.
        """
        if self._profile is None:
            profile = (
                self._profile_function() if self._profile_function is not None else None
            )
            if profile is None or profile.empty:
                raise ValueError(
                    "Company profiles (market capitalizations, sectors and countries) are "
                    "not available: they need a Financial Modeling Prep API key."
                    + (f" Alternatively, {alternative}." if alternative else "")
                )
            self._profile = profile
        return self._profile

    def _get_classification(
        self, assets: list[str], row: str, alternative: str = ""
    ) -> pd.Series:
        """
        The sector or country of each asset from the company profiles, with countries
        named in full ("United States" rather than "US"); empty for an asset without one.
        """
        profile = self._get_profile(alternative)
        values = (
            profile.loc[row].reindex(assets)
            if row in profile.index
            else pd.Series(index=assets, dtype=object)
        )
        values = values.where(
            values.map(lambda value: isinstance(value, str) and value != "")
        )
        if row == "Country":
            # Lazily imported: the country names live with the Economics data.
            from financetoolkit.economics.helpers import COUNTRY_CODES  # noqa: PLC0415

            values = values.map(lambda value: COUNTRY_CODES.get(value, value))
        return values


# ruff: noqa: E501


@filter_shown_columns
@track_sources
class PortfolioConstruction(
    OptimizationConstruction,
    AnalysisConstruction,
    AttributionConstruction,
    PortfolioConstructionBase,
):
    """
    The Portfolio Construction module builds and examines portfolios from the assets of a
    Toolkit.

    Optimal weights (minimum variance, maximum Sharpe ratio, mean-variance, risk parity,
    Hierarchical Risk Parity, Hierarchical Equal Risk Contribution, Nested Clustered
    Optimization, Black-Litterman, minimum CVaR, minimum drawdown, maximum Omega ratio,
    maximum CVaR ratio, maximum diversification, minimum tracking error, growth optimal,
    resampled efficient), the efficient frontier, a walk-forward backtest of the optimizers
    against equal weights, risk contributions to volatility, expected shortfall,
    semideviation, mean absolute deviation, drawdowns and the other risk measures of the
    Risk module, per asset and per factor, the risk metrics of given weights, a whole-share
    allocation of the weights, diversification metrics, historical stress tests,
    look-through exposure, a Brinson-Fachler attribution against the benchmark, a factor
    attribution and the return contributions per period, and the estimates behind the
    weights: expected returns, covariance and codependence matrices, factor loadings, asset
    clusters, coskewness and cokurtosis, their bootstrapped uncertainty and Entropy Pooling
    of views. Black-Litterman takes views on assets and, augmented or Bayesian, on factors.
    The optimizers take weight bounds per asset, sector and country bounds from the company
    profiles, for long-short portfolios a net and gross exposure and, from current weights,
    a maximum turnover.
    """
