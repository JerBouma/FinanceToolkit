"""Portfolio Construction Module"""

__docformat__ = "google"

from financetoolkit.portfolio.construction.analysis.analysis_controller import (
    AnalysisConstruction,
)
from financetoolkit.portfolio.construction.attribution.attribution_controller import (
    AttributionConstruction,
)
from financetoolkit.portfolio.construction.optimization.optimization_controller import (
    OptimizationConstruction,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class PortfolioConstruction(
    OptimizationConstruction, AnalysisConstruction, AttributionConstruction
):
    """
    The Portfolio Construction module builds and examines portfolios from the assets of a
    Toolkit: optimal weights (minimum variance, maximum Sharpe ratio, mean-variance,
    risk parity, Hierarchical Risk Parity, Hierarchical Equal Risk Contribution, Nested
    Clustered Optimization, Black-Litterman, minimum CVaR, minimum drawdown, maximum
    Omega ratio, maximum CVaR ratio, maximum diversification, minimum tracking error,
    growth optimal, resampled efficient), the efficient frontier, a walk-forward
    backtest of the optimizers against equal weights, risk contributions to volatility,
    expected shortfall, semideviation, mean absolute deviation and drawdowns, the risk
    metrics of given weights, a whole-share allocation of the weights, diversification
    metrics, historical stress tests,
    look-through exposure and a Brinson-Fachler attribution against the benchmark. The
    optimizers take weight bounds per asset, sector and country bounds from the company
    profiles, for long-short portfolios a net and gross exposure and, from current
    weights, a maximum turnover.
    """
