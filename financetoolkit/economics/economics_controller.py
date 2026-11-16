"""Economics Module"""

__docformat__ = "google"

from financetoolkit.economics.activity.activity_controller import ActivityEconomics
from financetoolkit.economics.commodities.commodities_controller import (
    CommoditiesEconomics,
)
from financetoolkit.economics.environment.environment_controller import (
    EnvironmentEconomics,
)
from financetoolkit.economics.government.government_controller import (
    GovernmentEconomics,
)
from financetoolkit.economics.history.history_controller import HistoryEconomics
from financetoolkit.economics.jobs.jobs_controller import JobsEconomics
from financetoolkit.economics.output.output_controller import OutputEconomics
from financetoolkit.economics.prices.prices_controller import PricesEconomics
from financetoolkit.economics.rates.rates_controller import RatesEconomics
from financetoolkit.economics.stability.stability_controller import StabilityEconomics
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.sources_model import (
    track_sources,
)

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class Economics(
    OutputEconomics,
    GovernmentEconomics,
    HistoryEconomics,
    PricesEconomics,
    RatesEconomics,
    JobsEconomics,
    ActivityEconomics,
    StabilityEconomics,
    EnvironmentEconomics,
    CommoditiesEconomics,
):
    """
    The Economics module contains methods to retrieve economic data from the OECD, the
    Global Macro Database, central banks, statistical offices and other official
    publishers. These can be anything ranging from Gross Domestic Product (GDP) to
    Inflation to Consumer Price Index (CPI) and more.
    """
