"""PortfolioConstruction Base Module"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.portfolio.construction.optimization import optimization_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR, apply_rounding

# ruff: noqa: E501

logger = get_logger()

BENCHMARK_TICKER = "Benchmark"

PORTFOLIO_TICKER = "Portfolio"

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

    def _get_returns(self, period: str, include_benchmark: bool) -> pd.DataFrame:
        """The returns of the assets over the periods in which every asset has a return."""
        if period not in PERIODS:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        # Historical data supplied to the Toolkit can reach beyond its date range.
        returns = self._historical_data[period]["Return"].loc[
            self._start_date : self._end_date
        ]
        assets = [
            asset
            for asset in self._get_assets(include_benchmark)
            if asset in returns.columns
        ]
        if len(assets) < 2:  # noqa: PLR2004
            raise ValueError("A portfolio needs the returns of at least two assets.")

        returns = returns[assets].dropna(how="any")
        if len(returns) < len(assets) + 1:
            raise ValueError(
                f"There are only {len(returns)} periods in which every asset has a "
                "return; use a longer date range or a higher frequency."
            )
        return returns

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

    def _get_inputs(
        self,
        period: str,
        include_benchmark: bool,
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, float]:
        """The returns, annualized expected returns, covariance and risk free rate."""
        returns = self._get_returns(period, include_benchmark)
        risk_free_rate = self._get_risk_free_rate(period, returns.index)
        expected, covariance_matrix = (
            optimization_model.get_expected_returns_and_covariance(
                returns,
                PERIODS_PER_YEAR[period],
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
            resolved = latest.reindex(assets).fillna(0).astype(float)
        else:
            resolved = pd.Series(1.0, index=assets)

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

    def _get_profile(self) -> pd.DataFrame:
        """The company profiles, retrieved once."""
        if self._profile is None:
            if self._profile_function is None:
                raise ValueError("Company profiles are not available for this Toolkit.")
            self._profile = self._profile_function()
        return self._profile

    def _get_classification(self, assets: list[str], row: str) -> pd.Series:
        """
        The sector or country of each asset from the company profiles, with countries
        named in full ("United States" rather than "US"); empty for an asset without one.
        """
        profile = self._get_profile()
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
