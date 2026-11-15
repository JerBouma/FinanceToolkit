"""Options Module"""

__docformat__ = "google"

from financetoolkit.options.greeks.greeks_controller import GreeksOptions
from financetoolkit.options.market.market_controller import MarketOptions
from financetoolkit.options.pricing.pricing_controller import PricingOptions
from financetoolkit.options.volatility.volatility_controller import VolatilityOptions
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class Options(PricingOptions, GreeksOptions, VolatilityOptions, MarketOptions):
    """
    The Options module is meant to calculate important options metrics such as the
    First, Second and Third Order Greeks, the Black Scholes Model and the Option Chains as well as
    Implied Volatilities, Breeden—Litzenberger and more.
    """
