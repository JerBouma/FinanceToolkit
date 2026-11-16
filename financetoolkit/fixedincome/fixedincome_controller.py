"""Fixed Income Module"""

__docformat__ = "google"

from financetoolkit.fixedincome.bonds.bonds_controller import BondsFixedIncome
from financetoolkit.fixedincome.credit.credit_controller import CreditFixedIncome
from financetoolkit.fixedincome.curves.curves_controller import CurvesFixedIncome
from financetoolkit.fixedincome.loans.loans_controller import LoansFixedIncome
from financetoolkit.fixedincome.markets.markets_controller import MarketsFixedIncome
from financetoolkit.fixedincome.policy.policy_controller import PolicyFixedIncome
from financetoolkit.fixedincome.short_rates.short_rates_controller import (
    ShortRatesFixedIncome,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class FixedIncome(
    BondsFixedIncome,
    MarketsFixedIncome,
    ShortRatesFixedIncome,
    CurvesFixedIncome,
    CreditFixedIncome,
    PolicyFixedIncome,
    LoansFixedIncome,
):
    """
    The Fixed income module contains methods to obtain data related to Central Banks, yield curves, credit
    spreads and individual bonds, and calculates bond measures from their market prices. The formulas for
    terms of your own (bond pricing, swaptions with the Black and Bachelier models, the option-adjusted
    spread of a callable bond) are in bond_model, yieldcurve_model, derivative_model and short_rate_model.
    """
