"""Fixed Income Module"""

__docformat__ = "google"

import os
from datetime import datetime, timedelta

from financetoolkit.cache.cache_controller import Cache, resolve_cache, set_active_cache
from financetoolkit.economics.sources import (
    boe_model,
    ecb_model as economics_ecb_model,
    mof_model,
    treasury_model,
)
from financetoolkit.fixedincome.bonds.bonds_controller import BondsFixedIncome
from financetoolkit.fixedincome.credit.credit_controller import CreditFixedIncome
from financetoolkit.fixedincome.curves.curves_controller import CurvesFixedIncome
from financetoolkit.fixedincome.inflation.inflation_controller import (
    InflationFixedIncome,
)
from financetoolkit.fixedincome.loans.loans_controller import LoansFixedIncome
from financetoolkit.fixedincome.markets.markets_controller import MarketsFixedIncome
from financetoolkit.fixedincome.policy.policy_controller import PolicyFixedIncome
from financetoolkit.fixedincome.short_rates.short_rates_controller import (
    ShortRatesFixedIncome,
)
from financetoolkit.fixedincome.sources import (
    boc_model,
    bundesbank_model,
    norgesbank_model,
    rba_model,
    riksbank_model,
)
from financetoolkit.fixedincome.swaps.swaps_controller import SwapsFixedIncome
from financetoolkit.utilities import logger_model, validation_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.sources_model import track_sources

logger = logger_model.get_logger()


FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")


@filter_shown_columns
@track_sources
class FixedIncomeBase:
    """
    The data every fixed income topic shares: the bond selection, the yield curves and the zero curves.
    """

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        quarterly: bool | None = None,
        rounding: int | None = 4,
        fred_api_key: str = FRED_API_KEY,
        api_key: str = "",
        cache: Cache | None = None,
        use_cached_data: bool | str | None = None,
    ):
        """
        Initializes the Fixed Income Controller Class.

        Args:
            start_date (str | None, optional): The start date to retrieve data from. Defaults to None.
            end_date (str | None, optional): The end date to retrieve data from. Defaults to None.
            quarterly (bool | None, optional): Whether to return quarterly data where a method is
                not given a period. Defaults to None, which returns the highest frequency available.
            rounding (int | None, optional): The number of decimals to round the results to, or None to skip rounding. Defaults to 4.
            fred_api_key (str, optional): A FRED API key used to retrieve ICE BofA bond index data
                (option-adjusted spread, effective yield, total return, yield to worst). Obtain a free key at
                https://fred.stlouisfed.org/docs/api/api_key.html. Can also be set via the FRED_API_KEY
                environment variable. Defaults to the value of FRED_API_KEY if set, otherwise an empty string.
            api_key (str, optional): A FinancialModelingPrep API key used to retrieve the Treasury par yield
                curve rates. Obtain one at https://www.jeroenbouma.com/fmp and pass it here. Defaults to an empty string.
            cache (Cache | None, optional): A cache to use for the data this module retrieves, which takes
                precedence over use_cached_data. Defaults to None.
            use_cached_data (bool | str | None, optional): Whether to cache the data retrieved from external
                sources. None or True uses the shared cache database in the user configuration directory, False
                retrieves everything every time and a string is the path to a dedicated cache folder or database
                file. Defaults to None, which caches unless the FINANCE_TOOLKIT_CACHE_ENABLED environment variable is 0.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
            fred_api_key='your_fred_api_key',
        )

        fixedincome.get_ice_bofa_effective_yield(maturity=False)
        ```

        Which returns:

        | Date       |      AAA |       AA |        A |      BBB |       BB |        B |      CCC |
        |:-----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
        | 2024-01-01 | nan      | nan      | nan      | nan      | nan      | nan      | nan      |
        | 2024-01-02 |   0.0459 |   0.0473 |   0.0509 |   0.0543 |   0.0622 |   0.0763 |   0.1333 |
        | 2024-01-03 |   0.0459 |   0.0474 |   0.051  |   0.0544 |   0.0634 |   0.0779 |   0.1358 |
        | 2024-01-04 |   0.0466 |   0.0481 |   0.0518 |   0.0551 |   0.0639 |   0.0784 |   0.1367 |
        | 2024-01-05 |   0.047  |   0.0485 |   0.0521 |   0.0554 |   0.0641 |   0.0787 |   0.137  |
        | 2024-01-08 |   0.0465 |   0.0481 |   0.0517 |   0.055  |   0.0633 |   0.0776 |   0.1365 |
        | 2024-01-09 |   0.0464 |   0.048  |   0.0516 |   0.0548 |   0.0629 |   0.0771 |   0.1359 |
        | 2024-01-10 |   0.0464 |   0.048  |   0.0515 |   0.0547 |   0.0622 |   0.0762 |   0.1351 |
        | 2024-01-11 |   0.0456 |   0.0472 |   0.0507 |   0.054  |   0.0619 |   0.076  |   0.1344 |
        | 2024-01-12 |   0.0451 |   0.0467 |   0.0502 |   0.0534 |   0.0613 |   0.0753 |   0.1338 |
        | 2024-01-15 |   0.0451 |   0.0467 |   0.0501 |   0.0533 |   0.0611 |   0.0751 |   0.1328 |
        """
        if start_date and not validation_model.is_valid_date(start_date):
            raise ValueError(
                f"Please input a valid start date (%Y-%m-%d) like '2010-01-01', not '{start_date}'"
            )
        if end_date and not validation_model.is_valid_date(end_date):
            raise ValueError(
                f"Please input a valid end date (%Y-%m-%d) like '2020-01-01', not '{end_date}'"
            )
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"Please ensure the start date {start_date} is before the end date {end_date}"
            )

        self._start_date = (
            start_date
            if start_date
            else (datetime.now() - timedelta(days=365 * 100)).strftime("%Y-%m-%d")
        )
        self._end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._fred_api_key = validation_model.resolve_fred_api_key(fred_api_key)
        # A copied documentation example passes the placeholder key, treated as no key at all.
        self._api_key = validation_model.resolve_api_key(api_key)
        # The data retrieved from external sources is cached, see use_cached_data.
        resolved_cache = resolve_cache(use_cached_data, cache)
        self._cache = resolved_cache if resolved_cache.enabled else None

        # Published once here so the FRED, ECB and Fed free functions read it back.
        set_active_cache(self._cache)

    def _get_country_yield_curves(self, start_date: str) -> dict:
        """
        Returns, per country, the function that retrieves its daily government bond yield
        curve from the official source, so that only the requested countries are fetched.

        Args:
            start_date (str): The start date to retrieve from (YYYY-MM-DD).

        Returns:
            dict: The retrieval function per country.
        """
        end_date = self._end_date

        return {
            "United States": lambda: treasury_model.get_labelled_yield_curve(
                start_date, end_date
            ),
            "Euro Area": lambda: economics_ecb_model.get_yield_curve(
                start_date, end_date
            ),
            "Germany": lambda: bundesbank_model.get_yield_curve(start_date, end_date),
            "United Kingdom": lambda: boe_model.get_yield_curve(start_date, end_date),
            "Japan": lambda: mof_model.get_government_bond_yields(start_date),
            "Canada": lambda: boc_model.get_yield_curve(start_date, end_date),
            "Sweden": lambda: riksbank_model.get_yield_curve(start_date, end_date),
            "Norway": lambda: norgesbank_model.get_yield_curve(start_date, end_date),
            "Australia": lambda: rba_model.get_yield_curve(start_date),
        }


# ruff: noqa: E501


@filter_shown_columns
@track_sources
class FixedIncome(
    BondsFixedIncome,
    SwapsFixedIncome,
    InflationFixedIncome,
    MarketsFixedIncome,
    ShortRatesFixedIncome,
    CurvesFixedIncome,
    CreditFixedIncome,
    PolicyFixedIncome,
    LoansFixedIncome,
    FixedIncomeBase,
):
    """
    The Fixed income module contains methods to obtain data related to Central Banks, yield curves, credit
    spreads and individual bonds, and calculates bond measures from their market prices. The formulas for
    terms of your own (bond pricing, swaptions, caps, floors, collars and year-on-year inflation caps with
    the Black, shifted Black and Bachelier models and their implied volatilities, bond options, caplets,
    swaptions and CIR++ shifts in the Vasicek, Hull-White, CIR and CIR++ models, the option-adjusted
    spread of a callable or puttable bond, Hull-White, Black-Karasinski and Black-Derman-Toy trees with
    callable and puttable bonds, Bermudan swaptions and American bond options, swaps and forwards, CDS
    options, inflation swaps and the forward inflation index) are in bond_model, yieldcurve_model,
    derivative_model, short_rate_model, trinomial_tree_model, swap_model, forward_model, cds_model and
    inflation_swap_model.
    """
