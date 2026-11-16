"""Econometrics Module"""

__docformat__ = "google"

from financetoolkit.econometrics.causal.causal_controller import CausalEconometrics
from financetoolkit.econometrics.diagnostics.diagnostics_controller import (
    DiagnosticsEconometrics,
)
from financetoolkit.econometrics.dynamics.dynamics_controller import (
    DynamicsEconometrics,
)
from financetoolkit.econometrics.forecasting.forecasting_controller import (
    ForecastingEconometrics,
)
from financetoolkit.econometrics.regression.regression_controller import (
    RegressionEconometrics,
)
from financetoolkit.econometrics.significance.significance_controller import (
    SignificanceEconometrics,
)
from financetoolkit.econometrics.stationarity.stationarity_controller import (
    StationarityEconometrics,
)
from financetoolkit.econometrics.testing.testing_controller import TestingEconometrics
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class Econometrics(
    DiagnosticsEconometrics,
    StationarityEconometrics,
    RegressionEconometrics,
    TestingEconometrics,
    CausalEconometrics,
    ForecastingEconometrics,
    DynamicsEconometrics,
    SignificanceEconometrics,
):
    """
    The Econometrics module is meant to test the statistical properties of a price
    or return series -- stationarity (unit root tests), long-run equilibrium
    relationships between series (cointegration), predictive lead-lag relationships
    (Granger causality), and general model/residual diagnostics (ARCH effects,
    normality, autocorrelation, structural stability). These are the standard
    foundational tests behind approaches such as pairs trading, spread modeling,
    and choosing an appropriate volatility or return model in the first place.

    Through the Toolkit (`toolkit.econometrics`) it analyzes the prices and returns of the
    Toolkit's tickers. Standalone, `Econometrics(data=...)` analyzes any metric: ratios,
    economic indicators, yield curves or a dataset of your own.
    """
