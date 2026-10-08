"""Fixed Income Module"""

__docformat__ = "google"


import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from financetoolkit import helpers
from financetoolkit.cache.cache_controller import Cache, set_active_cache
from financetoolkit.economics import (
    boe_model,
    ecb_model as economics_ecb_model,
    frb_model,
    mof_model,
    oecd_model,
    treasury_model,
)
from financetoolkit.economics.helpers import (
    buffered_start_date,
    resample_to_period,
    validate_period,
)
from financetoolkit.fixedincome import (
    boc_model,
    bond_model,
    bundesbank_model,
    derivative_model,
    ecb_model,
    eiopa_model,
    esma_model,
    euribor_model,
    fed_model,
    fmp_model,
    fred_model,
    norgesbank_model,
    rba_model,
    riksbank_model,
    yieldcurve_model,
)
from financetoolkit.utilities import logger_model, validation_model
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.statistics_model import apply_rounding, finalize_dataset

logger = logger_model.get_logger()

# The Treasury's names for the maturities of the US par yield curve, in the labels the
# government bond yield curve uses for every country.
TREASURY_MATURITIES = {
    "1 Mo": "1M",
    "1.5 Month": "1.5M",
    "2 Mo": "2M",
    "3 Mo": "3M",
    "4 Mo": "4M",
    "6 Mo": "6M",
    "1 Yr": "1Y",
    "2 Yr": "2Y",
    "3 Yr": "3Y",
    "5 Yr": "5Y",
    "7 Yr": "7Y",
    "10 Yr": "10Y",
    "20 Yr": "20Y",
    "30 Yr": "30Y",
}


def _maturity_in_months(label: str) -> float:
    """
    Converts a maturity label such as "3M" or "10Y" into months, for sorting a curve.

    Args:
        label (str): The maturity label.

    Returns:
        float: The maturity in months.
    """
    return float(label[:-1]) * (12 if label.endswith("Y") else 1)


# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,
# pylint: disable=too-many-locals,line-too-long,too-many-public-methods
# ruff: noqa: E501

FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# A sample nominal spot curve by maturity, the default for the yield-curve methods.
DEFAULT_SPOT_CURVE: dict[float, float] = {
    1: 0.03,
    2: 0.032,
    3: 0.034,
    5: 0.038,
    7: 0.041,
    10: 0.044,
    20: 0.048,
    30: 0.05,
}

# A sample real spot curve, the default for get_breakeven_inflation_rate.
DEFAULT_REAL_SPOT_CURVE: dict[float, float] = {
    1: 0.008,
    2: 0.009,
    3: 0.01,
    5: 0.012,
    7: 0.014,
    10: 0.016,
    20: 0.018,
    30: 0.02,
}


class FixedIncome:
    """
    The Fixed income module contains methods to obtain data related to Central Banks, Option Adjusted Spreads,
    Valuation Models such as Black Model and Bond Pricing.
    """

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        quarterly: bool = True,
        rounding: int | None = 4,
        fred_api_key: str = FRED_API_KEY,
        api_key: str = "",
        cache: Cache | None = None,
    ):
        """
        Initializes the Fixed Income Controller Class.

        Args:
            start_date (str | None, optional): The start date to retrieve data from. Defaults to None.
            end_date (str | None, optional): The end date to retrieve data from. Defaults to None.
            quarterly (bool, optional): Whether to return the data quarterly. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            fred_api_key (str, optional): A FRED API key used to retrieve ICE BofA bond index data
                (option-adjusted spread, effective yield, total return, yield to worst). Obtain a free key at
                https://fred.stlouisfed.org/docs/api/api_key.html. Can also be set via the FRED_API_KEY
                environment variable. Defaults to the value of FRED_API_KEY if set, otherwise an empty string.
            api_key (str, optional): A FinancialModelingPrep API key used to retrieve the Treasury par yield
                curve rates. Obtain one at https://www.jeroenbouma.com/fmp and pass it here. Defaults to an empty string.
            cache (Cache | None, optional): The incremental cache used for the FRED, ECB and Federal
                Reserve requests this module makes. Defaults to None, which disables caching.

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
        self._fred_api_key = fred_api_key
        # A copied documentation example passes the placeholder key, treated as no key at all.
        self._api_key = validation_model.resolve_api_key(api_key)
        self._cache = cache

        # Published once here so the FRED, ECB and Fed free functions read it back.
        set_active_cache(cache)

    def _require_fred_api_key(self) -> None:
        if not self._fred_api_key:
            logger.warning(
                "No FRED API key found. ICE BofA bond index data is sourced from FRED "
                "(Federal Reserve Economic Data) and requires a key to access — "
                "registration is entirely free and takes about a minute at "
                "https://fred.stlouisfed.org/docs/api/api_key.html. Once you have one, "
                "pass it via the fred_api_key argument or set the FRED_API_KEY "
                "environment variable."
            )
            raise ValueError(
                "A FRED API key is required to retrieve ICE BofA data. Obtain a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and pass it via the "
                "fred_api_key argument or set the FRED_API_KEY environment variable."
            )

    def _require_api_key(self) -> None:
        if not self._api_key:
            logger.warning(
                "No FinancialModelingPrep API key found. Treasury par yield curve rates "
                "require a key to access, obtain one (with 15% off) at "
                "https://www.jeroenbouma.com/fmp and pass it via the api_key argument."
            )
            raise ValueError(
                "A FinancialModelingPrep API key is required to retrieve Treasury rates. "
                "Obtain one at https://www.jeroenbouma.com/fmp and pass it via the "
                "api_key argument."
            )

    def collect_bond_statistics(
        self,
        par_value: float = 100,
        coupon_rate: float = 0.05,
        years_to_maturity: int = 5,
        yield_to_maturity: float = 0.08,
        frequency: int = 1,
        show_input_info: bool = True,
    ):
        """
        Collect the bond statistics for a given bond which includes the following fields:

            - Par Value: The face value of the bond.
            - Coupon Rate: The annual coupon rate (in decimal).
            - Years to Maturity: The number of years until the bond matures.
            - Yield to Maturity: The yield to maturity of the bond (in decimal).
            - Frequency: The number of coupon payments per year.
            - Present Value: The present value of the bond.
            - Current Yield: The annual coupon payment divided by the bond price.
            - Effective Yield: The annualised yield that accounts for the compounding of the coupon
                payments made within the year.
            - Macaulay's Duration: The weighted average time to receive the bond's cash flows.
            - Modified Duration: The Macaulay's duration divided by 1 plus the per-period yield
                (yield to maturity divided by the frequency).
            - Effective Duration: The percentage price change per unit change in yield, obtained by
                repricing the bond symmetrically 1% above and 1% below the current yield.
            - Dollar Duration: The modified duration multiplied by the bond price, divided by 100.
            - DV01: The currency change in the bond's price, per par value of face, for a one basis
                point (0.01%) change in the yield to maturity.
            - Convexity: The second derivative of the bond price with respect to the yield to maturity.

        These statistics can be used to evaluate the bond's performance as opposed to other bonds or to estimate the bond's
        sensitivity to changes in interest rates to be able to apply a hedging strategy.

        Also known as: bond data, fixed income statistics.

        Args:
            par_value (float): The face value of the bond. Defaults to 100.
            coupon_rate (float): The annual coupon rate (in decimal). Defaults to 0.05.
            years_to_maturity (int): The number of years until the bond matures. Defaults to 5.
            yield_to_maturity (float): The yield to maturity of the bond (in decimal). Defaults to 0.08.
            frequency (int): The number of coupon payments per year. Defaults to 1.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pd.Series: A pandas Series containing the bond statistics.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        # This is one example and below a collection of different bonds is shown with different characteristics
        fixedincome.collect_bond_statistics(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            yield_to_maturity=0.08,
            frequency=1,
        )
        ```

        Which returns:

        |                     |        0 |
        |:--------------------|---------:|
        | Par Value           | 100      |
        | Coupon Rate         |   0.05   |
        | Years to Maturity   |   5      |
        | Yield to Maturity   |   0.08   |
        | Frequency           |   1      |
        | Present Value       |  88.0219 |
        | Current Yield       |   0.0568 |
        | Effective Yield     |   0.05   |
        | Macaulay's Duration |   4.5116 |
        | Modified Duration   |   4.1774 |
        | Effective Duration  |   4.1798 |
        | Dollar Duration     |   3.677  |
        | DV01                |   0.0368 |
        | Convexity           |  22.4017 |

        Note how the effective duration sits just above the modified duration for every
        bond: the two measure the same sensitivity, and their small difference is exactly
        the convexity picked up by repricing over a 100 basis point shift rather than
        differentiating at a point.
        """
        bond_statistics = {
            "Par Value": par_value,
            "Coupon Rate": coupon_rate,
            "Years to Maturity": years_to_maturity,
            "Yield to Maturity": yield_to_maturity,
            "Frequency": frequency,
        }

        bond_statistics["Present Value"] = bond_model.get_bond_price(
            par_value=par_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=yield_to_maturity,
            frequency=frequency,
        )

        bond_statistics["Current Yield"] = bond_model.get_current_yield(
            par_value=par_value,
            coupon_rate=coupon_rate,
            bond_price=bond_statistics["Present Value"],
        )

        bond_statistics["Effective Yield"] = bond_model.get_effective_yield(
            coupon_rate=coupon_rate, frequency=frequency
        )

        bond_statistics["Macaulay's Duration"] = bond_model.get_macaulays_duration(
            par_value=par_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=yield_to_maturity,
            frequency=frequency,
        )

        bond_statistics["Modified Duration"] = bond_model.get_modified_duration(
            par_value=par_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=yield_to_maturity,
            frequency=frequency,
        )

        bond_statistics["Effective Duration"] = bond_model.get_effective_duration(
            par_value=par_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=yield_to_maturity,
            frequency=frequency,
        )

        bond_statistics["Dollar Duration"] = bond_model.get_dollar_duration(
            par_value=par_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=yield_to_maturity,
            frequency=frequency,
        )

        bond_statistics["DV01"] = bond_model.get_dv01(
            par_value=par_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=yield_to_maturity,
            frequency=frequency,
        )

        bond_statistics["Convexity"] = bond_model.get_convexity(
            par_value=par_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=yield_to_maturity,
            frequency=frequency,
        )

        if show_input_info:
            logger.info(
                "Par Value: %s, Coupon Rate: %s%%, Years to Maturity: %s, Yield to Maturity: %s%%, Frequency: %s",
                f"{par_value:,}",
                f"{coupon_rate * 100}",
                years_to_maturity,
                f"{yield_to_maturity * 100}",
                frequency,
            )

        return apply_rounding(pd.Series(bond_statistics), self._rounding)

    def get_present_value(
        self,
        par_value: float = 100,
        coupon_rate: float | range | np.ndarray | list | None = None,
        years_to_maturity: float | range | list | None = None,
        yield_to_maturity: float = 0.08,
        frequency: int = 1,
        show_input_info: bool = True,
    ):
        """
        Calculates the bond prices for different coupon rates and years to maturity. The bond price is the present value of the bond's
        future cash flows, which includes the coupon payments and the par value of the bond at maturity. The bond price is calculated
        using the following formula:

        - Bond Price = (C / r) * (1 — (1 + r)^-n) + F / (1 + r)^n

        where:

        - C = Coupon payment per period
        - r = Yield to maturity per period
        - n = Number of periods
        - F = Face value of the bond

        The bond price is used to determine the fair value of the bond and to compare the bond's price to its market price to determine
        if the bond is overvalued or undervalued.

        Also known as: PV, bond pricing, discounted cash flows.

        Args:
            par_value (float): The par value (face value) of the bond.
            coupon_rate (float, optional): The coupon rate of the bond. If not provided, a range of coupon rates will be used.
            years_to_maturity (float, optional): The years to maturity of the bond in years. If not provided, a range of years to maturity will be used.
            yield_to_maturity (float, optional): The yield to maturity of the bond. Defaults to 0.08.
            frequency (int, optional): The frequency of coupon payments per year. Defaults to 1.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the bond prices for different coupon rates and years to maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_present_value(
            coupon_rate=[0.03, 0.05, 0.07],
            years_to_maturity=[5, 10, 15],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Coupon Rate |     5 |    10 |    15 |
        |--------------:|------:|------:|------:|
        |          0.03 | 80.04 | 66.45 | 57.2  |
        |          0.05 | 88.02 | 79.87 | 74.32 |
        |          0.07 | 96.01 | 93.29 | 91.44 |
        """
        coupon_rate = (
            np.round(
                np.arange(max(0.05 - 0.005 * 20, 0.005), 0.05 + 0.005 * 20, 0.005), 10
            )
            if coupon_rate is None
            else coupon_rate
        )

        # A list of maturities has to be flattened into the column labels themselves; wrapping it in another list makes pandas read it as a one-level MultiIndex and label every column with a one-element tuple.
        years_to_maturity_dates = (
            [
                pd.to_datetime(self._end_date) + pd.Timedelta(days=365 * interval)
                for interval in range(1, 11)
            ]
            if years_to_maturity is None
            else list(
                [years_to_maturity]
                if isinstance(years_to_maturity, int | float)
                else years_to_maturity
            )
        )
        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(coupon_rate, int | float):
            coupon_rate = [coupon_rate]
        if isinstance(years_to_maturity, int | float):
            years_to_maturity = [years_to_maturity]

        bond_prices: dict[int, dict[float, dict[float, float]]] = {}

        for coupon in coupon_rate:
            bond_prices[coupon] = {}
            for maturity in years_to_maturity:
                bond_prices[coupon][maturity] = bond_model.get_bond_price(
                    par_value=par_value,
                    coupon_rate=float(coupon),
                    years_to_maturity=maturity,
                    yield_to_maturity=yield_to_maturity,
                    frequency=frequency,
                )

        bond_prices_df = pd.DataFrame.from_dict(bond_prices, orient="index")
        bond_prices_df.columns = years_to_maturity_dates

        bond_prices_df.index.name = "Coupon Rate"

        if show_input_info:
            logger.info(
                "Par Value: %s, Yield to Maturity: %s%%, Frequency: %s",
                f"{par_value:,}",
                f"{yield_to_maturity * 100}",
                frequency,
            )

        return bond_prices_df.round(2)

    def get_duration(
        self,
        duration_type: str = "modified",
        par_value: float = 100,
        coupon_rate: float | np.ndarray | list | None = None,
        years_to_maturity: float | range | list | None = None,
        yield_to_maturity: float = 0.08,
        frequency: int = 1,
        show_input_info: bool = True,
    ):
        """
        Calculates the bond duration for different coupon rates and years to maturity. It has the option to calculate the following
        type of bond durations:

        - Macaulay's Duration: The weighted average time to receive the bond's cash flows.
        - Modified Duration: The Macaulay's duration divided by 1 plus the per-period yield (yield to maturity divided by the frequency).
        - Effective Duration: The percentage change in the bond price for a 1% change in the yield to maturity.
        - Dollar Duration: The modified duration multiplied by the bond price, divided by 100.

        These duration measures can be used to estimate the sensitivity of a bond's price to changes in interest rates as well as
        to compare the risk of different bonds. The modified duration is particularly useful for estimating the percentage change
        in the bond price for a 1% change in the yield to maturity. Note that it is a percentage sensitivity and therefore not the
        same as the dollar duration, the price value of a basis point (PVBP) or the dollar value of a 0.01% change (DV01), which are
        all expressed as a currency amount instead. The dollar duration is available through this method via `duration_type='dollar'`
        and the DV01 is calculated separately, see `collect_bond_statistics`.

        Also known as: Macaulay duration, modified duration, bond price sensitivity.

        Args:
            duration_type (str, optional): The type of duration to calculate. Defaults to 'modified' but can also
                be 'macaulay', 'effective' or 'dollar'.
            par_value (float, optional): The par value (face value) of the bond. Defaults to 100.
            coupon_rate (float, optional): The coupon rate of the bond. If not provided, a range of coupon
                rates will be used. Defaults to None.
            years_to_maturity (float, optional): The years to maturity of the bond in years. If not provided, a range of years
                to maturity will be used. Defaults to None.
            yield_to_maturity (float, optional): The yield to maturity of the bond. Defaults to 0.08.
            frequency (int, optional): The frequency of coupon payments per year. Defaults to 1.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the bond duration for different coupon rates and years to maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_duration(
            duration_type='modified',
            coupon_rate=[0.03, 0.05, 0.07],
            years_to_maturity=[5, 10, 15],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Coupon Rate |    5 |   10 |    15 |
        |--------------:|-----:|-----:|------:|
        |          0.03 | 4.33 | 7.82 | 10.4  |
        |          0.05 | 4.18 | 7.26 |  9.41 |
        |          0.07 | 4.05 | 6.87 |  8.79 |
        """
        duration_type_lower = duration_type.lower()

        coupon_rate = (
            np.round(
                np.arange(max(0.05 - 0.005 * 20, 0.005), 0.05 + 0.005 * 20, 0.005), 10
            )
            if coupon_rate is None
            else coupon_rate
        )

        # A list of maturities has to be flattened into the column labels themselves; wrapping it in another list makes pandas read it as a one-level MultiIndex and label every column with a one-element tuple.
        years_to_maturity_dates = (
            [
                pd.to_datetime(self._end_date) + pd.Timedelta(days=365 * interval)
                for interval in range(1, 11)
            ]
            if years_to_maturity is None
            else list(
                [years_to_maturity]
                if isinstance(years_to_maturity, int | float)
                else years_to_maturity
            )
        )
        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(coupon_rate, int | float):
            coupon_rate = [coupon_rate]
        if isinstance(years_to_maturity, int | float):
            years_to_maturity = [years_to_maturity]

        bond_prices: dict[str, dict[float, dict[float, float]]] = {}

        for coupon in coupon_rate:
            bond_prices[coupon] = {}
            for maturity in years_to_maturity:
                if duration_type_lower == "modified":
                    bond_prices[coupon][maturity] = bond_model.get_modified_duration(
                        par_value=par_value,
                        coupon_rate=coupon,
                        years_to_maturity=maturity,
                        yield_to_maturity=yield_to_maturity,
                        frequency=frequency,
                    )
                elif duration_type_lower == "macaulay":
                    bond_prices[coupon][maturity] = bond_model.get_macaulays_duration(
                        par_value=par_value,
                        coupon_rate=coupon,
                        years_to_maturity=maturity,
                        yield_to_maturity=yield_to_maturity,
                        frequency=frequency,
                    )
                elif duration_type_lower == "effective":
                    bond_prices[coupon][maturity] = bond_model.get_effective_duration(
                        par_value=par_value,
                        coupon_rate=coupon,
                        years_to_maturity=maturity,
                        yield_to_maturity=yield_to_maturity,
                        frequency=frequency,
                    )
                elif duration_type_lower == "dollar":
                    bond_prices[coupon][maturity] = bond_model.get_dollar_duration(
                        par_value=par_value,
                        coupon_rate=coupon,
                        years_to_maturity=maturity,
                        yield_to_maturity=yield_to_maturity,
                        frequency=frequency,
                    )
                else:
                    raise ValueError(
                        "Please input a valid duration type ('macaulay', 'modified', 'effective' or 'dollar')"
                    )

        bond_prices_df = pd.DataFrame.from_dict(bond_prices, orient="index")
        bond_prices_df.columns = years_to_maturity_dates

        bond_prices_df.index.name = "Coupon Rate"

        if show_input_info:
            logger.info(
                "Par Value: %s, Yield to Maturity: %s%%, Frequency: %s, Type: %s Duration",
                f"{par_value:,}",
                f"{yield_to_maturity * 100}",
                frequency,
                duration_type_lower.title(),
            )

        return bond_prices_df.round(2)

    def get_yield_to_maturity(
        self,
        par_value: float = 100,
        coupon_rate: float = 0.05,
        years_to_maturity: float | range | list | None = None,
        bond_price: float | list | None = None,
        frequency: int = 1,
        guess: float = 0.05,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        show_input_info: bool = True,
    ):
        """
        Calculates the yield to maturity for a bond. The yield to maturity is the internal rate of return of the bond, which is the
        discount rate that equates the present value of the bond's cash flows to its market price. The yield to maturity is used to
        estimate the bond's return and to compare the bond's return to other investments.

        The yield to maturity is calculated using the following formula:

        - Bond Price = (C / r) * (1 — (1 + r)^-n) + F / (1 + r)^n

        where:

        - C = Coupon payment per period
        - r = Yield to maturity per period
        - n = Number of periods
        - F = Face value of the bond

        The goal is to find the yield to maturity that satisfies the equation above. This is done using the secant method
        which is an iterative method that converges to the root of a function.

        Also known as: YTM, bond return to maturity.

        Args:
            par_value (float): The par value (face value) of the bond. This is the original price when it was issued by the issuer.
            coupon_rate (float, optional): The coupon rate of the bond. Defaults to 0.05.
            years_to_maturity (float, optional): The years to maturity of the bond in years. Defaults to None.
            bond_price (float, optional): The price of the bond. Defaults to None.
            frequency (int, optional): The number of coupon payments per year. Defaults to 1.
            guess (float, optional): The initial guess for the yield to maturity. Defaults to 0.05.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations for convergence. Defaults to 100.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the yield to maturity for different bond prices and years to maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_yield_to_maturity(
            coupon_rate=0.05,
            years_to_maturity=[5, 10, 15],
            bond_price=[95, 100, 105],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Bond Price |      5 |     10 |     15 |
        |-------------:|-------:|-------:|-------:|
        |           95 | 0.0619 | 0.0567 | 0.055  |
        |          100 | 0.05   | 0.05   | 0.05   |
        |          105 | 0.0388 | 0.0437 | 0.0453 |
        """
        if bond_price is None:
            # Determine the step size based on the input number
            step_size = par_value / 10

            # Generate the list of numbers
            bond_price = [
                int(par_value - i * step_size)
                for i in range(21)
                if int(par_value - i * step_size) > 0
            ][::-1]
            bond_price.extend(
                int(par_value + i * step_size)
                for i in range(1, 21)
                if int(par_value - i * step_size) > 0
            )

        # A list of maturities has to be flattened into the column labels themselves; wrapping it in another list makes pandas read it as a one-level MultiIndex and label every column with a one-element tuple.
        years_to_maturity_dates = (
            [
                pd.to_datetime(self._end_date) + pd.Timedelta(days=365 * interval)
                for interval in range(1, 11)
            ]
            if years_to_maturity is None
            else list(
                [years_to_maturity]
                if isinstance(years_to_maturity, int | float)
                else years_to_maturity
            )
        )
        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(bond_price, int | float):
            bond_price = [bond_price]
        if isinstance(years_to_maturity, int | float):
            years_to_maturity = [years_to_maturity]

        yield_to_maturities: dict[int, dict[float, dict[float, float]]] = {}

        for price in bond_price:
            yield_to_maturities[price] = {}
            for maturity in years_to_maturity:
                yield_to_maturities[price][maturity] = bond_model.get_yield_to_maturity(
                    par_value=par_value,
                    coupon_rate=coupon_rate,
                    years_to_maturity=maturity,
                    bond_price=price,
                    frequency=frequency,
                    guess=guess,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )

        yield_to_maturities_df = pd.DataFrame.from_dict(
            yield_to_maturities, orient="index"
        )
        yield_to_maturities_df.columns = years_to_maturity_dates

        yield_to_maturities_df.index.name = "Bond Price"

        if show_input_info:
            logger.info(
                "Par Value: %s, Coupon Rate: %s%%, Frequency: %s",
                f"{par_value:,}",
                f"{coupon_rate * 100}",
                frequency,
            )

        return apply_rounding(yield_to_maturities_df, self._rounding)

    def get_forward_rate(
        self,
        spot_rates: pd.Series | dict | None = None,
        near_maturity: float | range | list | None = None,
        far_maturity: float | range | list | None = None,
        show_input_info: bool = True,
    ):
        """
        Calculates the implied forward rate between pairs of points on a zero-coupon
        (spot) yield curve. The forward rate is the interest rate, implied by today's
        yield curve, for a loan that starts at a future date — it is derived purely
        from no-arbitrage pricing rather than a forecast of future rates.

        The rate for each maturity is obtained by linearly interpolating the supplied
        spot curve, so `near_maturity` and `far_maturity` do not need to coincide
        exactly with a maturity present in `spot_rates`.

        The forward rate is calculated using the following formula:

        - Forward Rate = ((1 + r2)^t2 / (1 + r1)^t1)^(1 / (t2 - t1)) - 1

        where:

        - r1 = Spot rate at the near maturity
        - t1 = Near maturity, in years
        - r2 = Spot rate at the far maturity
        - t2 = Far maturity, in years

        Also known as: implied forward rate, forward-forward rate.

        Args:
            spot_rates (pd.Series | dict, optional): The zero-coupon (spot) yield curve,
                indexed by maturity in years (in decimal). Defaults to a sample curve.
            near_maturity (float | list, optional): The nearer maturity (or maturities),
                in years. If not provided, a range of near maturities will be used.
            far_maturity (float | list, optional): The further maturity (or maturities),
                in years. If not provided, a range of far maturities will be used.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the forward rate for each combination
            of near and far maturity. Combinations where the far maturity is not greater
            than the near maturity are returned as NaN.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_forward_rate(
            near_maturity=[1, 2, 3],
            far_maturity=[5, 10],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Near Maturity |     5 |     10 |
        |----------------:|------:|-------:|
        |               1 | 0.04  | 0.0456 |
        |               2 | 0.042 | 0.047  |
        |               3 | 0.044 | 0.0483 |
        """
        spot_rates_series = (
            pd.Series(DEFAULT_SPOT_CURVE)
            if spot_rates is None
            else pd.Series(spot_rates)
        ).sort_index()

        near_maturity = range(1, 6) if near_maturity is None else near_maturity
        far_maturity = range(5, 11) if far_maturity is None else far_maturity

        if isinstance(near_maturity, int | float):
            near_maturity = [near_maturity]
        if isinstance(far_maturity, int | float):
            far_maturity = [far_maturity]

        forward_rates: dict[float, dict[float, float]] = {}

        for near in near_maturity:
            forward_rates[near] = {}
            near_rate = float(
                np.interp(near, spot_rates_series.index, spot_rates_series.to_numpy())
            )
            for far in far_maturity:
                if far <= near:
                    forward_rates[near][far] = np.nan
                    continue

                far_rate = float(
                    np.interp(
                        far, spot_rates_series.index, spot_rates_series.to_numpy()
                    )
                )

                forward_rates[near][far] = yieldcurve_model.get_forward_rate(
                    near_rate=near_rate,
                    far_rate=far_rate,
                    near_maturity=near,
                    far_maturity=far,
                )

        forward_rates_df = pd.DataFrame.from_dict(forward_rates, orient="index")
        forward_rates_df.index.name = "Near Maturity"
        forward_rates_df.columns.name = "Far Maturity"

        if show_input_info:
            logger.info(
                "Spot Curve: %s",
                {k: round(v, 4) for k, v in spot_rates_series.items()},
            )

        return apply_rounding(forward_rates_df, self._rounding)

    def get_par_yield(
        self,
        spot_rates: pd.Series | dict | None = None,
        years_to_maturity: float | range | list | None = None,
        frequency: int = 1,
        par_value: float = 100,
        show_input_info: bool = True,
    ):
        """
        Calculates the par yield curve implied by a zero-coupon (spot) yield curve. The
        par yield for a given maturity is the coupon rate that would need to be attached
        to a newly-issued bond of that maturity so that, once its cash flows are
        discounted with the spot curve, its price equals its par value exactly.

        This is the curve that is typically quoted for on-the-run government bonds, as
        opposed to the theoretical spot curve which is usually bootstrapped rather than
        directly observed.

        The par yield is calculated using the following formula:

        - Par Yield = frequency * (1 - DF(n)) / SUM(DF(k))

        where DF(k) = 1 / (1 + spot_rate(k / frequency) / frequency)^k is the discount
        factor for the cash flow at period k, spot_rate(t) is obtained by interpolating
        the spot curve at time t (in years), and n = years_to_maturity * frequency is
        the number of coupon periods.

        Also known as: par rate, par coupon rate.

        Args:
            spot_rates (pd.Series | dict, optional): The zero-coupon (spot) yield curve,
                indexed by maturity in years (in decimal). Defaults to a sample curve.
            years_to_maturity (float | list, optional): The maturity (or maturities), in
                years, to calculate the par yield for. If not provided, a range of years
                to maturity will be used.
            frequency (int, optional): The number of coupon payments per year. Defaults to 1.
            par_value (float, optional): The face value of the bond. Defaults to 100.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.Series: A Series containing the par yield for each requested maturity,
            i.e. the par yield curve.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_par_yield(
            years_to_maturity=[1, 2, 3, 5, 10],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Years to Maturity |   Par Yield |
        |--------------------:|------------:|
        |                   1 |      0.03   |
        |                   2 |      0.032  |
        |                   3 |      0.0339 |
        |                   5 |      0.0377 |
        |                  10 |      0.0431 |
        """
        spot_rates_series = (
            pd.Series(DEFAULT_SPOT_CURVE)
            if spot_rates is None
            else pd.Series(spot_rates)
        ).sort_index()

        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(years_to_maturity, int | float):
            years_to_maturity = [years_to_maturity]

        par_yields: dict[float, float] = {}

        for maturity in years_to_maturity:
            par_yields[maturity] = yieldcurve_model.get_par_yield(
                spot_rates=spot_rates_series,
                years_to_maturity=maturity,
                frequency=frequency,
                par_value=par_value,
            )

        par_yields_series = pd.Series(par_yields)
        par_yields_series.index.name = "Years to Maturity"
        par_yields_series.name = "Par Yield"

        if show_input_info:
            logger.info(
                "Frequency: %s, Par Value: %s, Spot Curve: %s",
                frequency,
                f"{par_value:,}",
                {k: round(v, 4) for k, v in spot_rates_series.items()},
            )

        return apply_rounding(par_yields_series, self._rounding)

    def get_yield_curve_spread(
        self,
        spot_rates: pd.Series | dict | None = None,
        long_maturity: float | range | list | None = None,
        short_maturity: float | range | list | None = None,
        show_input_info: bool = True,
    ):
        """
        Calculates the spread between pairs of points on a yield curve, e.g. the widely
        followed 10-year minus 2-year Treasury spread. A positive spread indicates a
        "normal" upward-sloping curve, while a negative spread ("inversion") has
        historically been used as a leading indicator of an economic slowdown.

        The rate for each maturity is obtained by linearly interpolating the supplied
        curve, so `long_maturity` and `short_maturity` do not need to coincide exactly
        with a maturity present in `spot_rates`.

        The yield curve spread is calculated using the following formula:

        - Yield Curve Spread = Long-Term Yield - Short-Term Yield

        Also known as: term spread, yield curve slope.

        Args:
            spot_rates (pd.Series | dict, optional): The yield curve, indexed by
                maturity in years (in decimal). Defaults to a sample curve.
            long_maturity (float | list, optional): The longer maturity (or maturities),
                in years. If not provided, a range of long maturities will be used.
            short_maturity (float | list, optional): The shorter maturity (or
                maturities), in years. If not provided, a range of short maturities
                will be used.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the yield curve spread for each
            combination of long and short maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_yield_curve_spread(
            long_maturity=[10, 30],
            short_maturity=[1, 2],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Long Maturity |     1 |     2 |
        |----------------:|------:|------:|
        |              10 | 0.014 | 0.012 |
        |              30 | 0.02  | 0.018 |
        """
        spot_rates_series = (
            pd.Series(DEFAULT_SPOT_CURVE)
            if spot_rates is None
            else pd.Series(spot_rates)
        ).sort_index()

        long_maturity = range(5, 11) if long_maturity is None else long_maturity
        short_maturity = range(1, 5) if short_maturity is None else short_maturity

        if isinstance(long_maturity, int | float):
            long_maturity = [long_maturity]
        if isinstance(short_maturity, int | float):
            short_maturity = [short_maturity]

        yield_curve_spreads: dict[float, dict[float, float]] = {}

        for long in long_maturity:
            yield_curve_spreads[long] = {}
            long_yield = float(
                np.interp(long, spot_rates_series.index, spot_rates_series.to_numpy())
            )
            for short in short_maturity:
                short_yield = float(
                    np.interp(
                        short, spot_rates_series.index, spot_rates_series.to_numpy()
                    )
                )

                yield_curve_spreads[long][short] = (
                    yieldcurve_model.get_yield_curve_spread(
                        long_yield=long_yield, short_yield=short_yield
                    )
                )

        yield_curve_spreads_df = pd.DataFrame.from_dict(
            yield_curve_spreads, orient="index"
        )
        yield_curve_spreads_df.index.name = "Long Maturity"
        yield_curve_spreads_df.columns.name = "Short Maturity"

        if show_input_info:
            logger.info(
                "Spot Curve: %s",
                {k: round(v, 4) for k, v in spot_rates_series.items()},
            )

        return apply_rounding(yield_curve_spreads_df, self._rounding)

    def get_breakeven_inflation_rate(
        self,
        nominal_rates: pd.Series | dict | None = None,
        real_rates: pd.Series | dict | None = None,
        maturity: float | list | None = None,
        show_input_info: bool = True,
    ):
        """
        Calculates the breakeven inflation rate implied by a nominal and a real
        (inflation-protected) yield curve, e.g. the U.S. Treasury nominal curve versus
        the TIPS (Treasury Inflation-Protected Securities) curve. It is the rate of
        inflation that would make an investor indifferent between holding a nominal
        bond and an inflation-protected bond of the same maturity, and is widely used
        as a market-implied measure of expected inflation.

        The rate for each maturity is obtained by linearly interpolating the supplied
        curves, so `maturity` does not need to coincide exactly with a maturity present
        in `nominal_rates` or `real_rates`.

        The breakeven inflation rate is calculated using the following formula:

        - Breakeven Inflation Rate = Nominal Yield - Real Yield

        Also known as: TIPS breakeven spread, inflation breakeven.

        Args:
            nominal_rates (pd.Series | dict, optional): The nominal (non-inflation-protected)
                yield curve, indexed by maturity in years (in decimal). Defaults to a sample curve.
            real_rates (pd.Series | dict, optional): The real (inflation-protected) yield
                curve, indexed by maturity in years (in decimal). Defaults to a sample curve.
            maturity (float | list, optional): The maturity (or maturities), in years,
                to calculate the breakeven inflation rate for. If not provided, a range
                of maturities will be used.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.Series: A Series containing the breakeven inflation rate for each
            requested maturity, i.e. the breakeven inflation curve.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_breakeven_inflation_rate(
            maturity=[1, 5, 10, 30],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Maturity |   Breakeven Inflation Rate |
        |-----------:|---------------------------:|
        |          1 |                      0.022 |
        |          5 |                      0.026 |
        |         10 |                      0.028 |
        |         30 |                      0.03  |
        """
        nominal_rates_series = (
            pd.Series(DEFAULT_SPOT_CURVE)
            if nominal_rates is None
            else pd.Series(nominal_rates)
        ).sort_index()
        real_rates_series = (
            pd.Series(DEFAULT_REAL_SPOT_CURVE)
            if real_rates is None
            else pd.Series(real_rates)
        ).sort_index()

        maturity = [1, 2, 3, 5, 7, 10, 20, 30] if maturity is None else maturity

        if isinstance(maturity, int | float):
            maturity = [maturity]

        breakeven_inflation_rates: dict[float, float] = {}

        for single_maturity in maturity:
            nominal_yield = float(
                np.interp(
                    single_maturity,
                    nominal_rates_series.index,
                    nominal_rates_series.to_numpy(),
                )
            )
            real_yield = float(
                np.interp(
                    single_maturity,
                    real_rates_series.index,
                    real_rates_series.to_numpy(),
                )
            )

            breakeven_inflation_rates[single_maturity] = (
                yieldcurve_model.get_breakeven_inflation_rate(
                    nominal_yield=nominal_yield, real_yield=real_yield
                )
            )

        breakeven_inflation_rates_series = pd.Series(breakeven_inflation_rates)
        breakeven_inflation_rates_series.index.name = "Maturity"
        breakeven_inflation_rates_series.name = "Breakeven Inflation Rate"

        if show_input_info:
            logger.info(
                "Nominal Curve: %s, Real Curve: %s",
                {k: round(v, 4) for k, v in nominal_rates_series.items()},
                {k: round(v, 4) for k, v in real_rates_series.items()},
            )

        return apply_rounding(breakeven_inflation_rates_series, self._rounding)

    def get_z_spread(
        self,
        par_value: float = 100,
        coupon_rate: float = 0.05,
        years_to_maturity: float | range | list | None = None,
        bond_price: float | list | None = None,
        spot_rates: pd.Series | dict | None = None,
        frequency: int = 1,
        guess: float = 0.01,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        show_input_info: bool = True,
    ):
        """
        Calculates the zero-volatility spread (Z-spread) for a bond given a benchmark
        zero-coupon (spot) yield curve. The Z-spread is the constant spread that, when
        added uniformly to every point of the benchmark curve, makes the present value
        of the bond's discounted cash flows equal to its observed market price.

        Unlike a simple yield spread (the bond's yield to maturity minus a benchmark
        yield of the same maturity), the Z-spread is measured against the entire curve
        rather than a single point, which makes it a more accurate measure of the
        compensation an investor receives for a bond's credit and liquidity risk.

        The Z-spread is found iteratively using the secant method, in the same way that
        `get_yield_to_maturity` solves for the yield to maturity.

        Also known as: zero-volatility spread, static spread.

        Args:
            par_value (float): The par value (face value) of the bond.
            coupon_rate (float, optional): The coupon rate of the bond. Defaults to 0.05.
            years_to_maturity (float, optional): The years to maturity of the bond in years. Defaults to None.
            bond_price (float, optional): The price of the bond. Defaults to None.
            spot_rates (pd.Series | dict, optional): The benchmark zero-coupon (spot)
                yield curve, indexed by maturity in years (in decimal). Defaults to a sample curve.
            frequency (int, optional): The number of coupon payments per year. Defaults to 1.
            guess (float, optional): The initial guess for the Z-spread. Defaults to 0.01.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations for convergence. Defaults to 100.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the Z-spread for different bond prices and years to maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_z_spread(
            coupon_rate=0.05,
            years_to_maturity=[5, 10, 15],
            bond_price=[95, 100, 105],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Bond Price |      5 |     10 |     15 |
        |-------------:|-------:|-------:|-------:|
        |           95 | 0.0243 | 0.0137 | 0.0103 |
        |          100 | 0.0124 | 0.007  | 0.0053 |
        |          105 | 0.0012 | 0.0007 | 0.0005 |
        """
        spot_rates_series = (
            pd.Series(DEFAULT_SPOT_CURVE)
            if spot_rates is None
            else pd.Series(spot_rates)
        ).sort_index()

        if bond_price is None:
            step_size = par_value / 10

            bond_price = [
                int(par_value - i * step_size)
                for i in range(21)
                if int(par_value - i * step_size) > 0
            ][::-1]
            bond_price.extend(
                int(par_value + i * step_size)
                for i in range(1, 21)
                if int(par_value - i * step_size) > 0
            )

        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(bond_price, int | float):
            bond_price = [bond_price]
        if isinstance(years_to_maturity, int | float):
            years_to_maturity = [years_to_maturity]

        z_spreads: dict[float, dict[float, float]] = {}

        for price in bond_price:
            z_spreads[price] = {}
            for maturity in years_to_maturity:
                z_spreads[price][maturity] = bond_model.get_z_spread(
                    par_value=par_value,
                    coupon_rate=coupon_rate,
                    years_to_maturity=maturity,
                    bond_price=price,
                    spot_rates=spot_rates_series,
                    frequency=frequency,
                    guess=guess,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )

        z_spreads_df = pd.DataFrame.from_dict(z_spreads, orient="index")
        z_spreads_df.columns = list(years_to_maturity)

        z_spreads_df.index.name = "Bond Price"

        if show_input_info:
            logger.info(
                "Par Value: %s, Coupon Rate: %s%%, Frequency: %s, Spot Curve: %s",
                f"{par_value:,}",
                f"{coupon_rate * 100}",
                frequency,
                {k: round(v, 4) for k, v in spot_rates_series.items()},
            )

        return apply_rounding(z_spreads_df, self._rounding)

    def get_bond_equivalent_yield(
        self,
        discount_yield: float | list | np.ndarray | None = None,
        days_to_maturity: float | list | None = None,
        show_input_info: bool = True,
    ):
        """
        Converts a money-market discount yield (e.g. quoted for Treasury bills) into a
        bond-equivalent yield (BEY). Money-market instruments are often quoted on a
        discount-yield basis, which understates the actual return an investor earns
        because it is computed on face value rather than the (lower) purchase price,
        and uses a 360-day rather than a 365-day year. The bond-equivalent yield
        restates the discount yield on a basis that is comparable to coupon-bearing
        bonds and notes.

        The bond-equivalent yield is calculated using the following formula:

        - BEY = 365 * Discount Yield / (360 - Days to Maturity * Discount Yield)

        for a bill with half a year or less remaining. Beyond that an equivalent coupon-bearing
        note would have paid a coupon at the six month point, so the U.S. Treasury's semi-annually
        compounded solution (31 CFR 356, Appendix B) is used instead — see
        `bond_model.get_bond_equivalent_yield`. Applying the simple formula to a 52-week bill
        instead overstates its yield by roughly seven basis points.

        Also known as: BEY, coupon-equivalent yield, investment rate.

        Args:
            discount_yield (float | list, optional): The money-market discount yield of
                the instrument (in decimal). If not provided, a range of discount yields
                will be used.
            days_to_maturity (float | list, optional): The number of days until the
                instrument matures. If not provided, a range of typical T-bill maturities
                will be used.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the bond-equivalent yield for
            different discount yields and days to maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_bond_equivalent_yield(
            discount_yield=[0.03, 0.05, 0.07],
            days_to_maturity=[90, 180, 360],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Discount Yield |     90 |    180 |    360 |
        |-----------------:|-------:|-------:|-------:|
        |             0.03 | 0.0306 | 0.0309 | 0.0311 |
        |             0.05 | 0.0513 | 0.052  | 0.0527 |
        |             0.07 | 0.0722 | 0.0735 | 0.0749 |

        The 360-day column is computed with the Treasury's semi-annually compounded formula
        rather than the simple one, because a bill of that length would have paid a coupon
        halfway through if it were a note.
        """
        discount_yield = (
            np.round(np.arange(0.01, 0.105, 0.005), 10)
            if discount_yield is None
            else discount_yield
        )
        days_to_maturity = (
            [30, 60, 90, 180, 270, 360]
            if days_to_maturity is None
            else days_to_maturity
        )

        if isinstance(discount_yield, int | float):
            discount_yield = [discount_yield]
        if isinstance(days_to_maturity, int | float):
            days_to_maturity = [days_to_maturity]

        bond_equivalent_yields: dict[float, dict[float, float]] = {}

        for yield_value in discount_yield:
            bond_equivalent_yields[yield_value] = {}
            for days in days_to_maturity:
                bond_equivalent_yields[yield_value][days] = (
                    bond_model.get_bond_equivalent_yield(
                        discount_yield=float(yield_value), days_to_maturity=days
                    )
                )

        bond_equivalent_yields_df = pd.DataFrame.from_dict(
            bond_equivalent_yields, orient="index"
        )
        bond_equivalent_yields_df.index.name = "Discount Yield"
        bond_equivalent_yields_df.columns.name = "Days to Maturity"

        if show_input_info:
            logger.info(
                "Number of Discount Yields: %s, Days to Maturity: %s",
                len(discount_yield),
                list(days_to_maturity),
            )

        return apply_rounding(bond_equivalent_yields_df, self._rounding)

    def get_key_rate_duration(
        self,
        par_value: float = 100,
        coupon_rate: float = 0.05,
        years_to_maturity: float | range | list | None = None,
        spot_rates: pd.Series | dict | None = None,
        key_rate_maturity: float | range | list | None = None,
        frequency: int = 1,
        yield_change: float = 0.0001,
        show_input_info: bool = True,
    ):
        """
        Calculates the key rate duration of a bond for one or more individual maturity
        points ("key rates") on the yield curve. Whereas `get_duration` with
        `duration_type='effective'` assumes the entire curve shifts in parallel, key
        rate duration measures the bond's price sensitivity to a shock at a single
        tenor of the curve while every other point is held fixed. Because cash flows
        are discounted using linear interpolation between the curve's tenors, a shock
        at one tenor tapers off towards its neighboring tenors and has no effect beyond
        them.

        Summing the key rate durations across every tenor of the curve approximately
        reproduces the bond's effective (parallel-shift) duration, but key rate
        duration additionally reveals which segment of the curve the bond's price is
        most exposed to — information that is essential for constructing curve-neutral
        hedges or identifying "twist" risk.

        Also known as: partial duration, rate-specific duration.

        Args:
            par_value (float, optional): The par value (face value) of the bond. Defaults to 100.
            coupon_rate (float, optional): The coupon rate of the bond. Defaults to 0.05.
            years_to_maturity (float | list, optional): The years to maturity of the
                bond (or bonds). If not provided, a range of years to maturity will be used.
            spot_rates (pd.Series | dict, optional): The zero-coupon (spot) yield curve
                used to discount the bond's cash flows, indexed by maturity in years (in
                decimal). Defaults to a sample curve.
            key_rate_maturity (float | list, optional): The maturity (or maturities), in
                years, of the curve point(s) to shock. Must be present in the index of
                `spot_rates`. Defaults to every maturity in `spot_rates`.
            frequency (int, optional): The number of coupon payments per year. Defaults to 1.
            yield_change (float, optional): The size of the shock applied to each key
                rate, up and down (in decimal). Defaults to 0.0001 (1 basis point).
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the key rate duration for different
            bond maturities and key rate maturities.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_key_rate_duration(
            coupon_rate=0.05,
            years_to_maturity=[5, 10],
            key_rate_maturity=[2, 5, 10],
            show_input_info=False,
        )
        ```

        Which returns:

        |   Years to Maturity |      2 |      5 |      10 |
        |--------------------:|-------:|-------:|--------:|
        |                   5 | 0.0862 | 4.0561 | -0      |
        |                  10 | 0.0862 | 0.377  |  6.4666 |
        """
        spot_rates_series = (
            pd.Series(DEFAULT_SPOT_CURVE)
            if spot_rates is None
            else pd.Series(spot_rates)
        ).sort_index()

        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )
        key_rate_maturity = (
            list(spot_rates_series.index)
            if key_rate_maturity is None
            else key_rate_maturity
        )

        if isinstance(years_to_maturity, int | float):
            years_to_maturity = [years_to_maturity]
        if isinstance(key_rate_maturity, int | float):
            key_rate_maturity = [key_rate_maturity]

        key_rate_durations: dict[float, dict[float, float]] = {}

        for maturity in years_to_maturity:
            key_rate_durations[maturity] = {}
            for key_rate in key_rate_maturity:
                key_rate_durations[maturity][key_rate] = (
                    bond_model.get_key_rate_duration(
                        par_value=par_value,
                        coupon_rate=coupon_rate,
                        years_to_maturity=maturity,
                        spot_rates=spot_rates_series,
                        key_rate_maturity=key_rate,
                        frequency=frequency,
                        yield_change=yield_change,
                    )
                )

        key_rate_durations_df = pd.DataFrame.from_dict(
            key_rate_durations, orient="index"
        )
        key_rate_durations_df.index.name = "Years to Maturity"
        key_rate_durations_df.columns.name = "Key Rate Maturity"

        if show_input_info:
            logger.info(
                "Par Value: %s, Coupon Rate: %s%%, Frequency: %s, Spot Curve: %s",
                f"{par_value:,}",
                f"{coupon_rate * 100}",
                frequency,
                {k: round(v, 4) for k, v in spot_rates_series.items()},
            )

        return apply_rounding(key_rate_durations_df, self._rounding)

    def get_taylor_price_change(
        self,
        par_value: float = 100,
        coupon_rate: float | np.ndarray | list | None = None,
        years_to_maturity: float | range | list | None = None,
        yield_to_maturity: float = 0.08,
        frequency: int = 1,
        yield_change: float = 0.01,
        show_input_info: bool = True,
    ):
        """
        Estimates the percentage change in a bond's price for a given change in yield,
        using a second-order Taylor series expansion that combines modified duration
        and convexity.

        Modified duration alone only captures the first-order (linear) relationship
        between a bond's price and its yield, which understates the price increase for
        a yield decrease and overstates the price decrease for a yield increase because
        the true price-yield relationship is curved (convex), not linear. Adding a
        convexity term corrects for this and produces a substantially more accurate
        estimate, especially for larger yield changes.

        This method calls `get_modified_duration` and `get_convexity` from
        `bond_model.py` directly rather than recomputing them.

        The Taylor approximation is calculated using the following formula:

        - %ΔPrice ≈ -Modified Duration * Δy + 0.5 * Convexity * Δy^2

        Also known as: duration-convexity approximation, second-order price approximation.

        Args:
            par_value (float, optional): The par value (face value) of the bond. Defaults to 100.
            coupon_rate (float, optional): The coupon rate of the bond. If not provided,
                a range of coupon rates will be used.
            years_to_maturity (float, optional): The years to maturity of the bond in
                years. If not provided, a range of years to maturity will be used.
            yield_to_maturity (float, optional): The current yield to maturity of the
                bond. Defaults to 0.08.
            frequency (int, optional): The number of coupon payments per year. Defaults to 1.
            yield_change (float, optional): The hypothetical change in yield to
                maturity, e.g. 0.01 for a 100 basis point increase. Defaults to 0.01.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the estimated percentage price
            change for different coupon rates and years to maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_taylor_price_change(
            coupon_rate=[0.03, 0.05, 0.07],
            years_to_maturity=[5, 10, 15],
            yield_to_maturity=0.08,
            yield_change=0.01,
            show_input_info=False,
        )
        ```

        Which returns:

        |   Coupon Rate |       5 |      10 |      15 |
        |--------------:|--------:|--------:|--------:|
        |          0.03 | -0.0421 | -0.0744 | -0.097  |
        |          0.05 | -0.0407 | -0.0693 | -0.088  |
        |          0.07 | -0.0394 | -0.0656 | -0.0824 |
        """
        coupon_rate = (
            np.round(
                np.arange(max(0.05 - 0.005 * 20, 0.005), 0.05 + 0.005 * 20, 0.005), 10
            )
            if coupon_rate is None
            else coupon_rate
        )

        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(coupon_rate, int | float):
            coupon_rate = [coupon_rate]
        if isinstance(years_to_maturity, int | float):
            years_to_maturity = [years_to_maturity]

        price_changes: dict[float, dict[float, float]] = {}

        for coupon in coupon_rate:
            price_changes[coupon] = {}
            for maturity in years_to_maturity:
                price_changes[coupon][maturity] = bond_model.get_taylor_price_change(
                    par_value=par_value,
                    coupon_rate=float(coupon),
                    years_to_maturity=maturity,
                    yield_to_maturity=yield_to_maturity,
                    frequency=frequency,
                    yield_change=yield_change,
                )

        price_changes_df = pd.DataFrame.from_dict(price_changes, orient="index")
        price_changes_df.columns = list(years_to_maturity)

        price_changes_df.index.name = "Coupon Rate"

        if show_input_info:
            logger.info(
                "Par Value: %s, Yield to Maturity: %s%%, Frequency: %s, Yield Change: %s%%",
                f"{par_value:,}",
                f"{yield_to_maturity * 100}",
                frequency,
                f"{yield_change * 100}",
            )

        return apply_rounding(price_changes_df, self._rounding)

    def get_derivative_price(
        self,
        model: str = "black",
        forward_rate: float = 0.05,
        strike_rate: float | list | np.ndarray | None = None,
        volatility: float = 0.01,
        years_to_maturity: float | list | range | None = None,
        risk_free_rate: float | None = None,
        notional: float = 10_000_000,
        tenor: float | None = None,
        payment_frequency: int = 2,
        is_receiver: bool = True,
        volatility_type: str | None = None,
        include_payoff: bool = False,
        show_input_info: bool = True,
    ):
        """
        Calculates the derivative price for a fixed income instrument.

        It is possible to use two different models to calculate the derivative price:

        - Black Model: A mathematical model used for pricing financial derivatives, its primary applications are for
            pricing options on future contracts, bond options, interest rate cap and floors, and swaptions.
            For more information, see: https://en.wikipedia.org/wiki/Black_model
        - Bachelier Model: A deviation of the Black Model that is used for pricing future contracts. It is a simple model
            that assumes the price of the underlying asset follows a normal distribution with constant volatility. This
            is in contrast to the Black Model which assumes the price of the underlying asset follows a log-normal distribution.
            For more information, see: https://en.wikipedia.org/wiki/Bachelier_model

        It is possible to alter all parameters within the models, e.g. strike rate, volatility, years to maturity,
        risk-free rate, notional amount, and whether the holder is the receiver or payer of the derivative. Next to that, you can
        provide lists of values for the fixed rate, strike rate, volatility, and years to maturity to calculate the derivative price
        for multiple scenarios outside of the standard sample.

        Exercising a swaption is not a single payment at expiration — it is the right to enter a swap that exchanges
        cash flows at every payment date over the underlying swap's tenor. The price therefore discounts the option
        payoff by the swap's annuity (present value of a basis point) rather than a single discount factor to
        expiration, which is why the tenor and payment frequency of the underlying swap matter.

        Note that a swaption's price scales with the tenor of the underlying swap (a right to enter a
        longer-dated swap is worth more, since it exchanges cash flows over more payment dates) — pass
        `tenor` explicitly to price a swaption whose underlying swap tenor differs from its years to
        maturity, e.g. a 1-year option into a 5-year swap: `tenor=5, years_to_maturity=1`.

        The two models do not quote volatility on the same basis, and this matters a great deal.
        Black's model, being lognormal, reads `volatility` as a fraction of the forward rate, so
        0.20 is a 20% volatility. The Bachelier model, being normal, reads it as an absolute
        movement in rate units, so 0.0065 is 65 basis points. On a 3.25% forward those two quotes
        describe the same market, but swapping one for the other misprices the swaption by a factor
        of roughly thirty. By default `volatility` is therefore interpreted on whichever basis the
        chosen model is defined in; set `volatility_type` explicitly to supply a quote on the other
        basis and have it converted, using the at-the-money approximation
        sigma_normal ≈ sigma_lognormal * forward_rate.

        Black's model is undefined at a zero or negative forward or strike rate because it takes
        the logarithm of their ratio, and raises rather than returning a silent NaN in that case.
        Use the Bachelier model for the negative rates seen in the euro area and Japan.

        Also known as: bond derivative pricing, fixed income derivative, swaption pricing.

        Args:
            model (str, optional): The type of model to use for calculating the derivative price. Defaults to "black".
            forward_rate (float, optional): The forward rate as derived from the swap curve. Defaults to None.
            strike_rate (float | list, optional): The strike rate for the derivative. Defaults to None which means it calculates the
                derivative price a range of strike prices. Can also be a list of strike rates (e.g. [0.01, 0.02, 0.03, 0.04, 0.05]).
            volatility (float, optional): The volatility of the underlying swap rate, quoted on the
                basis given by `volatility_type`. Defaults to 0.01, read as a 1% lognormal volatility
                by the Black model and as 100 basis points of normal volatility by the Bachelier model.
            years_to_maturity (float | list, optional): The years to maturity of the derivative in years. Defaults to None which means it plots
                the derivative price for the next 10 years. Can also be a list of years to maturity (e.g. [1, 2.3, 2.5, 3])
            risk_free_rate (float, optional): The risk-free interest rate. Defaults to None which means it is equal to the fixed rate.
            notional (float, optional): The notional amount of the derivative. Defaults to 10_000_000.
            tenor (float | None, optional): The tenor (length in years) of the underlying swap. Defaults to None,
                which means it is equal to years_to_maturity for each scenario.
            payment_frequency (int, optional): Number of fixed-leg payments per year on the underlying swap
                (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly). Defaults to 2 (semi-annual).
            is_receiver (bool, optional): True if the holder is the receiver of the derivative, False if the holder is the payer. Defaults to True.
            volatility_type (str | None, optional): The convention `volatility` is quoted on, either
                'lognormal' (relative to the forward rate) or 'normal' (absolute, in rate units).
                Defaults to None, which uses the convention the chosen model is natively defined in:
                'lognormal' for the Black model and 'normal' for the Bachelier model.
            include_payoff (bool, optional): True to include the payoff in the output, False otherwise. Defaults to False.
            show_input_info (bool, optional): True to display input information, False otherwise. Defaults to True.

        Returns:
            pandas.DataFrame: The derivative prices rounded to the specified decimal places.
            pandas.DataFrame (optional): The derivative payoffs rounded to the specified decimal places if include_payoff is True.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        fixedincome.get_derivative_price(
            model='black',
            forward_rate=0.0325,
            strike_rate=[0.0275, 0.0325, 0.0375, 0.0425],
            years_to_maturity=[1, 2, 5, 10],
            show_input_info=False,
        )
        ```

        Which returns, with one column per expiry date and one row per strike:

        |   Strike Rate |   2026-12-31 |   2027-12-31 |   2030-12-30 |   2035-12-29 |
        |--------------:|-------------:|-------------:|-------------:|-------------:|
        |        0.0275 |         0    |         0    |          0   |          0   |
        |        0.0325 |      1224.91 |      3300.15 |      11280.4 |      25086.1 |
        |        0.0375 |     47237.2  |     89991.1  |     194547   |     305934   |
        |        0.0425 |     94474.3  |    179982    |     389094   |     611868   |

        The strikes below the 3.25% forward are worthless because a receiver swaption only
        pays when the fixed rate it locks in exceeds the prevailing forward, and at a 1%
        lognormal volatility a 50 basis point gap is far out of reach.
        """
        model_lower = model.lower()

        strike_rate = (
            np.round(
                np.arange(
                    max(
                        (
                            forward_rate - 0.005 * 20
                            if not is_receiver
                            else forward_rate - 0.005 * 5
                        ),
                        0.005,
                    ),
                    (
                        forward_rate + 0.005 * 20
                        if is_receiver
                        else forward_rate + 0.005 * 5
                    ),
                    0.005,
                ),
                10,
            )
            if strike_rate is None
            else strike_rate
        )

        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(years_to_maturity, int | float):
            years_to_maturity = [years_to_maturity]
        if isinstance(strike_rate, int | float):
            strike_rate = [strike_rate]

        years_to_maturity_dates = [
            (
                pd.to_datetime(self._end_date) + pd.Timedelta(days=365 * interval)
            ).strftime("%Y-%m-%d")
            for interval in years_to_maturity
        ]

        derivative_prices: dict[str, dict[float, float]] = {}
        derivative_payoffs: dict[str, dict[float, float]] = {}

        risk_free_rate = risk_free_rate if risk_free_rate is not None else forward_rate

        for strike in strike_rate:
            derivative_prices[strike], derivative_payoffs[strike] = {}, {}
            for maturity in years_to_maturity:
                if model_lower == "black":
                    (
                        derivative_prices[strike][maturity],
                        derivative_payoffs[strike][maturity],
                    ) = derivative_model.get_black_price(
                        forward_rate=forward_rate,
                        strike_rate=float(strike),
                        volatility=volatility,
                        years_to_maturity=maturity,
                        risk_free_rate=risk_free_rate,
                        notional=notional,
                        tenor=tenor,
                        payment_frequency=payment_frequency,
                        is_receiver=is_receiver,
                        volatility_type=(
                            "lognormal" if volatility_type is None else volatility_type
                        ),
                    )
                elif model_lower == "bachelier":
                    (
                        derivative_prices[strike][maturity],
                        derivative_payoffs[strike][maturity],
                    ) = derivative_model.get_bachelier_price(
                        forward_rate=forward_rate,
                        strike_rate=float(strike),
                        volatility=volatility,
                        years_to_maturity=maturity,
                        risk_free_rate=risk_free_rate,
                        notional=notional,
                        tenor=tenor,
                        payment_frequency=payment_frequency,
                        is_receiver=is_receiver,
                        volatility_type=(
                            "normal" if volatility_type is None else volatility_type
                        ),
                    )
                else:
                    raise ValueError(
                        "Please input a valid model type ('black' or 'bachelier')"
                    )

        derivative_prices_df = pd.DataFrame.from_dict(derivative_prices, orient="index")
        derivative_prices_df.columns = years_to_maturity_dates

        derivative_prices_df.index.name = "Strike Rate"

        if show_input_info:
            logger.info(
                "Forward Rate: %s%%, Volatility: %s%%, Risk Free Rate: %s%%, "
                "Holder: %s, Notional: %s, Payment Frequency: %sx/year, Model: %s Model",
                f"{forward_rate * 100}",
                f"{volatility * 100}",
                f"{risk_free_rate * 100}",
                "Receiver" if is_receiver else "Payer",
                f"{notional:,}",
                payment_frequency,
                model_lower.title(),
            )

        if include_payoff:
            derivative_payoffs_df = pd.DataFrame.from_dict(
                derivative_payoffs, orient="index"
            )
            derivative_payoffs_df.columns = years_to_maturity_dates

            derivative_payoffs_df.index.name = "Strike Rate"

            return derivative_prices_df.round(2), apply_rounding(
                derivative_payoffs_df, self._rounding
            )

        return derivative_prices_df.round(2)

    def get_government_bond_yield(
        self,
        short_term: bool = False,
        period: str | None = None,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Get the government bond yield for a variety of countries over time from the OECD. By
        default this is the long-term (10-year) government bond yield; set short_term=True to
        get the short-term (3-month) rate instead. The two maturities are described below.

        Long-term (short_term=False): long-term interest rates refer to government bonds maturing
        in ten years. Rates are mainly determined by the price charged by the lender, the risk
        from the borrower and the fall in the capital value. Long-term interest rates
        are generally averages of daily rates, measured as a percentage. These interest
        rates are implied by the prices at which the government bonds are traded on
        financial markets, not the interest rates at which the loans were issued.

        In all cases, they refer to bonds whose capital repayment is guaranteed by governments.
        Long-term interest rates are one of the determinants of business investment. Low long
        term interest rates encourage investment in new equipment and high interest rates
        discourage it. Investment is, in turn, a major source of economic growth.

        See definition: https://data.oecd.org/interest/long-term-interest-rates.htm

        Short-term (short_term=True): short-term interest rates are the rates at which short-term
        borrowings are effected between financial institutions or the rate at which short-term
        government paper is issued or traded in the market. Short-term interest rates are
        generally averages of daily rates, measured as a percentage. They are based on
        three-month money market rates where available; the OECD source specifically returns
        the 3-month interbank offered rate rather than a government bill yield, so for most
        countries it tracks the central bank's policy rate closely.

        See definition: https://data.oecd.org/interest/short-term-interest-rates.htm

        Also known as: treasury yield, bond yield by maturity.

        Args:
            short_term (bool, optional): Whether to return the short-term interest rate. Defaults to False.
                This means that the long-term interest rate will be returned.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the long-term (10-year) government bond yield, or the
                short-term (3-month) interest rate when short_term=True.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2023-05-01', end_date='2023-12-31')

        long_term_interest_rate = fixedincome.get_government_bond_yield(short_term=False, period='monthly')

        long_term_interest_rate.loc[:, ['Japan', 'United States', 'Brazil']]
        ```

        Which returns:

        |         |   Japan |   United States |   Brazil |
        |:--------|--------:|----------------:|---------:|
        | 2023-05 |  0.0043 |          0.0357 |   0.0728 |
        | 2023-06 |  0.004  |          0.0375 |   0.0728 |
        | 2023-07 |  0.0059 |          0.039  |   0.07   |
        | 2023-08 |  0.0064 |          0.0417 |   0.07   |
        | 2023-09 |  0.0076 |          0.0438 |   0.07   |
        | 2023-10 |  0.0095 |          0.048  |   0.0655 |
        | 2023-11 |  0.0066 |          0.045  |   0.0655 |
        | 2023-12 |  0.0062 |          0.0402 |   0.0655 |
        """
        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )

        if short_term:
            government_bond_yield = oecd_model.get_short_term_interest_rate(
                period=period,
            )
        else:
            government_bond_yield = oecd_model.get_long_term_interest_rate(
                period=period,
            )

        return finalize_dataset(
            dataset=government_bond_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

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

        def united_states() -> pd.DataFrame:
            curve = treasury_model.get_yield_curve("nominal", start_date, end_date)
            return curve.rename(columns=TREASURY_MATURITIES)[
                [
                    label
                    for column, label in TREASURY_MATURITIES.items()
                    if column in curve
                ]
            ]

        def australia() -> pd.DataFrame:
            # The Reserve Bank of Australia publishes one file with the history from 2013.
            curve = rba_model.get_yield_curve()
            return curve.loc[pd.Period(start_date, "D") :] if not curve.empty else curve

        def japan() -> pd.DataFrame:
            # The Ministry of Finance publishes one file with the full history.
            curve = mof_model.get_government_bond_yields()
            return curve.loc[pd.Period(start_date, "D") :] if not curve.empty else curve

        return {
            "United States": united_states,
            "Euro Area": lambda: economics_ecb_model.get_yield_curve(
                start_date, end_date
            ),
            "Germany": lambda: bundesbank_model.get_yield_curve(start_date, end_date),
            "United Kingdom": lambda: boe_model.get_yield_curve(start_date, end_date),
            "Japan": japan,
            "Canada": lambda: boc_model.get_yield_curve(start_date, end_date),
            "Sweden": lambda: riksbank_model.get_yield_curve(start_date, end_date),
            "Norway": lambda: norgesbank_model.get_yield_curve(start_date, end_date),
            "Australia": australia,
        }

    @handle_errors
    def get_government_bond_yield_curve(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the government bond yield curve of a variety of countries, every maturity
        a country's government borrows at, from treasury bills to bonds of 30 years and
        more. The yield curve is the basis for discounting cash flows and pricing bonds,
        and its shape is one of the most watched signals in markets: an inverted curve,
        with short rates above long rates, has preceded most recessions.

        Every curve comes from the official source without an API key:

        - United States: the U.S. Department of the Treasury's par yield curve, 1 month to
          30 years, from 1990.
        - Euro Area: the yield curve the European Central Bank estimates from the bonds of
          all euro area central governments, 3 months to 30 years, from 2004.
        - Germany: the term structure the Deutsche Bundesbank estimates from listed federal
          securities, 1 to 30 years.
        - United Kingdom: the Bank of England's nominal par yields of gilts, 5, 10 and 20
          years.
        - Japan: the Ministry of Finance's Japanese government bond yields, 1 to 40 years,
          from 1974.
        - Canada: the Bank of Canada's treasury bill yields (1 month to 1 year, weekly) and
          benchmark bond yields (2 to 30 years).
        - Sweden: the Riksbank's treasury bill and government bond yields, 1 month to 10
          years.
        - Norway: Norges Bank's generic government bond yields, 3 to 10 years.
        - Australia: the Reserve Bank of Australia's government bond yields interpolated to
          2, 3, 5 and 10 years, from 2013.

        With period="monthly", every other European Union member, such as France, Italy,
        Spain, the Netherlands and Poland, is included with its 10-year yield, the long-term
        interest rate for convergence purposes the ECB publishes monthly. For other
        countries no official source publishes a curve without a key; see
        `get_government_bond_yield` for the monthly 3-month and 10-year rates of around
        forty countries from the OECD. Only the requested countries are retrieved, and
        only the days that are not cached yet.

        The yields are returned as decimal fractions per annum (0.0419 for 4.19%), with one
        column per country and maturity, sorted from the shortest to the longest maturity.
        Weekly and monthly periods take the yields on the last trading day of each period
        (weeks end on Friday).

        Also known as: yield curve, term structure of interest rates, sovereign curve,
        treasury curve, bund curve, gilt curve, JGB curve.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "United States", "Euro Area", "Germany", "United Kingdom", "Japan", "Canada",
                "Sweden", "Norway" and "Australia", and with period="monthly" any European
                Union member.
                Defaults to None, which retrieves every country.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame with the yields as decimals, indexed by date with a
            column per country and maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-09-28', end_date='2026-10-01')

        yield_curve = fixedincome.get_government_bond_yield_curve(
            countries=['United States', 'Germany']
        )

        yield_curve['Germany'][['1Y', '2Y', '5Y', '10Y', '20Y', '30Y']]
        ```

        Which returns:

        |            |     1Y |     2Y |     5Y |    10Y |    20Y |    30Y |
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2026-09-28 | 0.031  | 0.0332 | 0.0344 | 0.0369 | 0.0393 | 0.0401 |
        | 2026-09-29 | 0.0307 | 0.0329 | 0.0341 | 0.0368 | 0.0394 | 0.0403 |
        | 2026-09-30 | 0.0304 | 0.0324 | 0.0336 | 0.0364 | 0.039  | 0.0399 |
        | 2026-10-01 | 0.0302 | 0.0321 | 0.0335 | 0.0366 | 0.0394 | 0.0403 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "government bond yield curve"
        )
        sources = self._get_country_yield_curves(
            buffered_start_date(self._start_date, period)
        )

        if countries is not None and not isinstance(countries, str | list | tuple):
            raise TypeError(
                "The countries must be a country name or a list of country names, such as "
                f"'Germany' or ['Germany', 'Japan'], not a {type(countries).__name__} ({countries!r})."
            )

        requested = (
            list(sources)
            if countries is None
            else [countries] if isinstance(countries, str) else list(countries)
        )

        # The 10-year yield of every other European Union member is published monthly, so a
        # monthly request includes those as a curve of one maturity.
        convergence_yields = (
            economics_ecb_model.get_long_term_convergence_yields(
                buffered_start_date(self._start_date, "monthly"), self._end_date
            )
            if period == "monthly"
            else pd.DataFrame()
        )
        if countries is None and not convergence_yields.empty:
            requested += [
                country
                for country in convergence_yields.columns
                if country not in sources
            ]

        if unavailable := [
            country
            for country in requested
            if country not in sources and country not in convergence_yields.columns
        ]:
            logger.warning(
                "No official daily yield curve is available for %s. The government bond yield "
                "curve covers %s daily, and the 10-year yield of every other European Union "
                "member with period='monthly'; get_government_bond_yield returns the monthly "
                "3-month and 10-year rates of around forty countries from the OECD.",
                ", ".join(unavailable),
                ", ".join(sources),
            )

        curves = {}
        for country in requested:
            if country not in sources:
                if country in convergence_yields.columns:
                    curves[country] = (
                        convergence_yields[[country]]
                        .rename(columns={country: "10Y"})
                        .dropna()
                    )
                continue

            curve = sources[country]()

            if curve.empty:
                continue

            curve = curve[sorted(curve.columns, key=_maturity_in_months)]
            curves[country] = resample_to_period(curve.dropna(how="all"), period)

        if not curves:
            return pd.DataFrame()

        yield_curve = pd.concat(curves, axis=1).sort_index()
        yield_curve.index.name = None
        yield_curve.columns.names = ["Country", "Maturity"]

        return finalize_dataset(
            dataset=yield_curve,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    def _get_keyless_treasury_rates(self) -> pd.DataFrame:
        """
        Retrieves the Treasury par yield curve from the U.S. Department of the Treasury, in
        the maturities and column names FinancialModelingPrep uses.

        Returns:
            pd.DataFrame: The rates as decimals, indexed by date ("Date").
        """
        curve = treasury_model.get_yield_curve(
            "nominal", self._start_date, self._end_date
        )

        if curve.empty:
            return curve

        # The Treasury's names for the maturities FinancialModelingPrep publishes.
        maturities = {
            "1 Mo": "1 Month",
            "2 Mo": "2 Month",
            "3 Mo": "3 Month",
            "6 Mo": "6 Month",
            "1 Yr": "1 Year",
            "2 Yr": "2 Year",
            "3 Yr": "3 Year",
            "5 Yr": "5 Year",
            "7 Yr": "7 Year",
            "10 Yr": "10 Year",
            "20 Yr": "20 Year",
            "30 Yr": "30 Year",
        }
        treasury_rates = curve.rename(columns=maturities)[
            [name for column, name in maturities.items() if column in curve.columns]
        ]
        treasury_rates.index.name = "Date"

        return treasury_rates

    @handle_errors
    def get_treasury_rates(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the daily U.S. Treasury par yield curve rates as officially published by the
        U.S. Department of the Treasury, covering every maturity from 1 Month through 30 Year in
        a single dataset. This is the official, risk-free curve widely used as the discount curve
        for bond valuation and as the benchmark for credit spreads.

        No API key is needed: with a FinancialModelingPrep API key the rates come from
        FinancialModelingPrep, and without one, or when the key's plan does not include them,
        from the U.S. Department of the Treasury directly, which gives the same figures.

        Also known as: the Treasury yield curve, the risk-free curve.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Notes:
            The underlying endpoint caps each request at 90 calendar days of data, so this method
            paginates in 90-day windows to cover the full start_date to end_date range the class
            was initialized with. A long range therefore issues many requests -- be mindful of
            this on a Free plan's daily request limit and consider a narrower start_date where
            possible.

            The U.S. Department of the Treasury publishes these rates in percentage points (a
            ten-year yield of 3.95%), but they are converted to decimals here (0.0395) so that
            they match every other rate method in this module -- `get_euribor_rates`,
            `get_european_central_bank_rates`, `get_federal_reserve_rates`,
            `get_government_bond_yield` and the ICE BofA yield methods -- as well as the risk-free
            rate returned by `Toolkit.get_treasury_data`. They can therefore be passed directly
            into `get_present_value`, `get_z_spread`, `get_par_yield` and `get_key_rate_duration`,
            each of which is documented as taking a rate in decimal form.

            This changed in v2.2.0: prior versions returned percentage points from this one method
            alone, which silently overstated a yield by a factor of 100 whenever the result was fed
            into any of the bond-pricing methods above. Multiply by 100 to recover the published
            Treasury figures.

        Returns:
            pd.DataFrame: A DataFrame containing the Treasury par yield curve rates, as decimals,
            with one column per maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2024-01-01', end_date='2024-01-15')

        fixedincome.get_treasury_rates()
        ```

        Which returns:

        | Date       |   1 Month |   2 Month |   3 Month |   6 Month |   1 Year |   2 Year |   3 Year |   5 Year |   7 Year |   10 Year |   20 Year |   30 Year |
        |:-----------|----------:|----------:|----------:|----------:|---------:|---------:|---------:|---------:|---------:|----------:|----------:|----------:|
        | 2024-01-02 |    0.0555 |    0.0554 |    0.0546 |    0.0524 |   0.048  |   0.0433 |   0.0409 |   0.0393 |   0.0395 |    0.0395 |    0.0425 |    0.0408 |
        | 2024-01-03 |    0.0554 |    0.0554 |    0.0548 |    0.0525 |   0.0481 |   0.0433 |   0.0407 |   0.039  |   0.0392 |    0.0391 |    0.0421 |    0.0405 |
        | 2024-01-04 |    0.0556 |    0.0548 |    0.0548 |    0.0525 |   0.0485 |   0.0438 |   0.0414 |   0.0397 |   0.0399 |    0.0399 |    0.043  |    0.0413 |
        | 2024-01-05 |    0.0554 |    0.0548 |    0.0547 |    0.0524 |   0.0484 |   0.044  |   0.0417 |   0.0402 |   0.0404 |    0.0405 |    0.0437 |    0.0421 |
        | 2024-01-08 |    0.0554 |    0.0548 |    0.0549 |    0.0524 |   0.0482 |   0.0436 |   0.0411 |   0.0397 |   0.0399 |    0.0401 |    0.0433 |    0.0417 |
        | 2024-01-09 |    0.0553 |    0.0546 |    0.0547 |    0.0524 |   0.0482 |   0.0436 |   0.0409 |   0.0397 |   0.04   |    0.0402 |    0.0433 |    0.0418 |
        | 2024-01-10 |    0.0553 |    0.0546 |    0.0546 |    0.0523 |   0.0482 |   0.0437 |   0.041  |   0.0399 |   0.0401 |    0.0404 |    0.0435 |    0.042  |
        | 2024-01-11 |    0.0554 |    0.0547 |    0.0546 |    0.0522 |   0.0475 |   0.0426 |   0.0402 |   0.039  |   0.0395 |    0.0398 |    0.0432 |    0.0418 |
        | 2024-01-12 |    0.0555 |    0.0547 |    0.0545 |    0.0516 |   0.0465 |   0.0414 |   0.0392 |   0.0384 |   0.0391 |    0.0396 |    0.0432 |    0.042  |
        """
        treasury_rates = pd.DataFrame()

        if self._api_key:
            try:
                treasury_rates = fmp_model.get_treasury_rates(
                    api_key=self._api_key,
                    start_date=self._start_date,
                    end_date=self._end_date,
                )
            except (ValueError, requests.exceptions.RequestException) as error:
                logger.warning(
                    "Could not retrieve the Treasury rates from FinancialModelingPrep (%s), "
                    "retrieving them from the U.S. Department of the Treasury instead.",
                    error,
                )
                treasury_rates = pd.DataFrame()

        # The Treasury publishes these in percentage points while every other rate method in this module returns decimals, so convert here rather than in the model -- that keeps the cached payload a faithful mirror of the endpoint. The error path, such as a plan that does not include the endpoint, returns a frame with no numeric columns, hence the guard.  # noqa: E501
        numeric_columns = treasury_rates.select_dtypes(include="number").columns

        if not numeric_columns.empty:
            treasury_rates[numeric_columns] = treasury_rates[numeric_columns] / 100
        else:
            # Without a key, or when FinancialModelingPrep does not serve them, the same
            # rates come from the U.S. Department of the Treasury, already as decimals.
            treasury_rates = self._get_keyless_treasury_rates()

        return finalize_dataset(
            dataset=treasury_rates,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_ice_bofa_option_adjusted_spread(
        self,
        maturity: bool = True,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        The ICE BofA Option-Adjusted Spreads (OASs) are the calculated spreads between a computed OAS index
        of all bonds in a given maturity and rating category and a spot Treasury curve. An OAS index is constructed
        using each constituent bond's OAS, weighted by market capitalization.

        The Option-Adjusted Spread (OAS) is the spread relative to a risk-free interest rate, usually measured in
        basis points (bp), that equates the theoretical present value of a series of uncertain cash flows to the
        market price of a fixed-income investment. The spread is added to the risk-free rate to compensate for the
        uncertainty of the cash flows.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBB
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13Y

        Args:
            maturity (bool, optional): Whether to return the maturity option adjusted spread or the rating option adjusted spread.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.

        Notes:
            ICE restricted the history it licenses to FRED in April 2026: every ICE BofA series now
            carries only the most recent three years of observations. A start_date earlier than that
            silently returns fewer rows rather than an error, and the example below will fall out of
            range in time. Go to the ICE source directly for longer histories.

        Returns:
            pd.DataFrame: A DataFrame containing the Option Adjusted Spread, in basis points. The FRED
            series are published in percent and are multiplied by 100 here, so a 0.77% spread is
            returned as 77.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
        )

        fixedincome.get_ice_bofa_option_adjusted_spread()
        ```

        Which returns:

        | Date       |   1-3 Years |   3-5 Years |   5-7 Years |   7-10 Years |   10-15 Years |   15+ Years |
        |:-----------|------------:|------------:|------------:|-------------:|--------------:|------------:|
        | 2024-01-01 |         nan |         nan |         nan |          nan |           nan |         nan |
        | 2024-01-02 |          78 |          95 |         109 |          128 |           133 |         119 |
        | 2024-01-03 |          80 |          98 |         113 |          133 |           136 |         122 |
        | 2024-01-04 |          80 |          98 |         112 |          133 |           135 |         122 |
        | 2024-01-05 |          80 |          98 |         112 |          132 |           134 |         121 |
        | 2024-01-08 |          79 |          98 |         112 |          132 |           134 |         120 |
        | 2024-01-09 |          78 |          96 |         110 |          130 |           131 |         117 |
        | 2024-01-10 |          77 |          94 |         108 |          128 |           128 |         113 |
        | 2024-01-11 |          75 |          94 |         107 |          128 |           127 |         113 |
        | 2024-01-12 |          74 |          94 |         107 |          128 |           126 |         112 |
        | 2024-01-15 |          74 |          94 |         107 |          128 |           125 |         111 |
        """
        self._require_fred_api_key()

        option_adjusted_spread = (
            fred_model.get_maturity_option_adjusted_spread(
                self._start_date, self._end_date, self._fred_api_key
            )
            if maturity
            else fred_model.get_rating_option_adjusted_spread(
                self._start_date, self._end_date, self._fred_api_key
            )
        )

        return finalize_dataset(
            dataset=option_adjusted_spread,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_ice_bofa_effective_yield(
        self,
        maturity: bool = True,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        This data represents the effective yield of the ICE BofA Indices, When the last calendar day of the month
        takes place on the weekend, weekend observations will occur as a result of month ending accrued interest adjustments.

        The effective yield of an ICE BofA index is the yield of the index as a whole, aggregated from the
        yields of its constituent bonds and weighted by their market capitalisation, on the same
        compounded (effective annual) basis that the accompanying Semi-Annual Yield to Worst series is
        quoted on a semi-annual basis. It is an index-level yield of the corporate bond market segment,
        not a statistic derived from any single bond's coupon — for the single-bond coupon-reinvestment
        calculation, see the "Effective Yield" row of `collect_bond_statistics` instead.

        The FRED series are published in percent and are converted to decimals here, so a 5.40% BBB
        index yield is returned as 0.054.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBBEY
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13YEY

        Also known as: ICE BofA corporate bond yield, credit yield.

        Args:
            maturity (bool, optional): Whether to return the maturity effective yield or the rating effective yield.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.

        Notes:
            ICE restricted the history it licenses to FRED in April 2026: every ICE BofA series now
            carries only the most recent three years of observations. A start_date earlier than that
            silently returns fewer rows rather than an error, and the example below will fall out of
            range in time. Go to the ICE source directly for longer histories.

        Returns:
            pd.DataFrame: A DataFrame containing the ICE BofA Effective Yield

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
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
        self._require_fred_api_key()

        effective_yield = (
            fred_model.get_maturity_effective_yield(
                self._start_date, self._end_date, self._fred_api_key
            )
            if maturity
            else fred_model.get_rating_effective_yield(
                self._start_date, self._end_date, self._fred_api_key
            )
        )

        return finalize_dataset(
            dataset=effective_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_ice_bofa_total_return(
        self,
        maturity: bool = True,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        This data represents the total return of the ICE BofA Indices, When the last calendar day of the month
        takes place on the weekend, weekend observations will occur as a result of month ending accrued interest adjustments.

        The total return is the actual rate of return of an investment or a pool of investments over a given evaluation period.
        Total return includes interest, capital gains, dividends and distributions realized over a given period of time.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLCC0A4BBBTRIV
        - Maturity: https://fred.stlouisfed.org/series/BAMLCC1A013YTRIV

        Args:
            maturity (bool, optional): Whether to return the maturity total return or the rating total return.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.

        Notes:
            ICE restricted the history it licenses to FRED in April 2026: every ICE BofA series now
            carries only the most recent three years of observations. A start_date earlier than that
            silently returns fewer rows rather than an error, and the example below will fall out of
            range in time. Go to the ICE source directly for longer histories.

        Returns:
            pd.DataFrame: A DataFrame containing the ICE BofA Total Return, as an index level rather
            than a rate of return. It is not rescaled, since the FRED series' unit is already an index.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
        )

        fixedincome.get_ice_bofa_total_return(maturity=True)
        ```

        Which returns:

        | Date       |   1-3 Years |   3-5 Years |   5-7 Years |   7-10 Years |   10-15 Years |   15+ Years |
        |:-----------|------------:|------------:|------------:|-------------:|--------------:|------------:|
        | 2024-01-01 |      nan    |      nan    |      nan    |       nan    |        nan    |      nan    |
        | 2024-01-02 |     1912.73 |     2484.25 |      807.62 |       584.32 |       4193.7  |     4343.71 |
        | 2024-01-03 |     1912.18 |     2483.95 |      807.54 |       583.84 |       4194.39 |     4339.07 |
        | 2024-01-04 |     1910.86 |     2477.9  |      804.35 |       580.42 |       4163.24 |     4289.24 |
        | 2024-01-05 |     1910.86 |     2475.75 |      802.82 |       578.73 |       4148.31 |     4262.52 |
        | 2024-01-08 |     1912.48 |     2480.39 |      804.97 |       580.71 |       4167.04 |     4302.16 |
        | 2024-01-09 |     1913.5  |     2482.27 |      805.72 |       581.26 |       4173.04 |     4303.34 |
        | 2024-01-10 |     1914.12 |     2483.6  |      806.21 |       581.29 |       4175.16 |     4304.82 |
        | 2024-01-11 |     1918.28 |     2492.25 |      809.94 |       583.92 |       4200.49 |     4330.72 |
        | 2024-01-12 |     1922.1  |     2498.89 |      812.41 |       585.2  |       4213.47 |     4338.43 |
        | 2024-01-15 |     1922.67 |     2499.76 |      812.67 |       585.41 |       4215.34 |     4340.24 |
        """
        self._require_fred_api_key()

        total_return = (
            fred_model.get_maturity_total_return(
                self._start_date, self._end_date, self._fred_api_key
            )
            if maturity
            else fred_model.get_rating_total_return(
                self._start_date, self._end_date, self._fred_api_key
            )
        )

        return finalize_dataset(
            dataset=total_return,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_ice_bofa_yield_to_worst(
        self,
        maturity: bool = True,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        This data represents the semi-annual yield to worst of the ICE BofA Indices, When the last calendar day of the month
        takes place on the weekend, weekend observations will occur as a result of month ending accrued interest adjustments.

        Yield to worst is the lowest potential yield that a bond can generate without the issuer defaulting. The standard US
        convention for this series is to use semi-annual coupon payments, whereas the standard in the foreign markets is
        to use coupon payments with frequencies of annual, semi-annual, quarterly, and monthly.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBBSYTW
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13YSYTW

        Args:
            maturity (bool, optional): Whether to return the maturity yield to worst or the rating yield to worst.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.

        Notes:
            ICE restricted the history it licenses to FRED in April 2026: every ICE BofA series now
            carries only the most recent three years of observations. A start_date earlier than that
            silently returns fewer rows rather than an error, and the example below will fall out of
            range in time. Go to the ICE source directly for longer histories.

        Returns:
            pd.DataFrame: A DataFrame containing the ICE BofA Yield to Worst. The FRED series are
            published in percent and are converted to decimals here, so 5.42% is returned as 0.0542.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
        )

        fixedincome.get_ice_bofa_yield_to_worst(maturity=False)
        ```

        Which returns:

        | Date       |      AAA |       AA |        A |      BBB |       BB |        B |      CCC |
        |:-----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
        | 2024-01-01 | nan      | nan      | nan      | nan      | nan      | nan      | nan      |
        | 2024-01-02 |   0.046  |   0.0475 |   0.0506 |   0.0546 |   0.0652 |   0.0796 |   0.1329 |
        | 2024-01-03 |   0.0461 |   0.0475 |   0.0507 |   0.0547 |   0.0662 |   0.081  |   0.1353 |
        | 2024-01-04 |   0.0468 |   0.0483 |   0.0515 |   0.0554 |   0.0665 |   0.0814 |   0.136  |
        | 2024-01-05 |   0.0471 |   0.0486 |   0.0518 |   0.0557 |   0.0667 |   0.0816 |   0.1362 |
        | 2024-01-08 |   0.0466 |   0.0482 |   0.0514 |   0.0553 |   0.066  |   0.0806 |   0.1359 |
        | 2024-01-09 |   0.0465 |   0.0481 |   0.0513 |   0.0551 |   0.0656 |   0.0803 |   0.1353 |
        | 2024-01-10 |   0.0465 |   0.0481 |   0.0512 |   0.0551 |   0.065  |   0.0795 |   0.1345 |
        | 2024-01-11 |   0.0458 |   0.0473 |   0.0504 |   0.0543 |   0.0648 |   0.0793 |   0.134  |
        | 2024-01-12 |   0.0453 |   0.0468 |   0.0499 |   0.0537 |   0.0642 |   0.0786 |   0.1335 |
        | 2024-01-15 |   0.0452 |   0.0468 |   0.0498 |   0.0537 |   0.064  |   0.0784 |   0.1325 |
        """
        self._require_fred_api_key()

        yield_to_worst = (
            fred_model.get_maturity_yield_to_worst(
                self._start_date, self._end_date, self._fred_api_key
            )
            if maturity
            else fred_model.get_rating_yield_to_worst(
                self._start_date, self._end_date, self._fred_api_key
            )
        )

        return finalize_dataset(
            dataset=yield_to_worst,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_hqm_corporate_bond_yield_curve(
        self,
        rate: str = "spot",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the High Quality Market (HQM) corporate bond yield curve of the U.S.
        Department of the Treasury: the yields of high quality (AAA, AA and A rated) US
        corporate bonds by maturity, from 6 months to 100 years, monthly from 1984. The
        Treasury builds it for discounting long-dated liabilities, which is why it is the
        prescribed discount curve for US corporate pension obligations, and its long history
        makes it a natural basis for modelling how corporate yields and spreads move.

        Two curves are available. Spot rates (rate="spot") are zero-coupon yields, the rate
        to discount a single payment at that maturity, for 17 maturities. Par yields
        (rate="par") are the coupons a bond priced at par would pay, for 2, 5, 10 and 30
        years, comparable to the Treasury par yield curve (see `get_treasury_rates`). Every
        value is the average over the month.

        The data comes from FRED, which republishes the Treasury's curve, so a free FRED API
        key is required. The rates are returned as decimal fractions (0.0558 for 5.58%).

        See definition: https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curves

        Also known as: HQM curve, corporate bond yield curve, pension discount curve, AA
        corporate curve.

        Args:
            rate (str, optional): "spot" for the spot rates or "par" for the par yields.
                Defaults to "spot".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The monthly average rates, indexed by month with a column per maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-03-01', end_date='2026-08-31')

        fixedincome.get_hqm_corporate_bond_yield_curve()[['1Y', '5Y', '10Y', '30Y', '100Y']]
        ```

        Which returns:

        |         |     1Y |     5Y |    10Y |    30Y |   100Y |
        |:--------|-------:|-------:|-------:|-------:|-------:|
        | 2026-03 | 0.0406 | 0.0447 | 0.0514 | 0.0614 | 0.0653 |
        | 2026-04 | 0.0408 | 0.0449 | 0.0513 | 0.0612 | 0.065  |
        | 2026-05 | 0.0414 | 0.0467 | 0.0528 | 0.0622 | 0.0656 |
        | 2026-06 | 0.0422 | 0.0472 | 0.0527 | 0.0611 | 0.0641 |
        | 2026-07 | 0.0433 | 0.0488 | 0.0545 | 0.0643 | 0.0677 |
        | 2026-08 | 0.0431 | 0.0497 | 0.0558 | 0.0665 | 0.0702 |
        """
        self._require_fred_api_key()
        if rate not in ("spot", "par"):
            raise ValueError(f"The rate must be 'spot' or 'par', not {rate!r}.")

        hqm_curve = fred_model.get_hqm_corporate_bond_yield_curve(
            rate, self._start_date, self._end_date, self._fred_api_key
        )

        return finalize_dataset(
            dataset=hqm_curve,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_hqm_corporate_bond_spread(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Computes the credit spread of high quality US corporate bonds per maturity: the
        Treasury's High Quality Market (HQM) corporate par yield minus the Treasury constant
        maturity yield, at 2, 5, 10 and 30 years, monthly from 1984. Both legs are par
        yields averaged over the month, so they are directly comparable. The spread is what
        investors demand for the default and liquidity risk of investment grade companies,
        and its shape across maturities, usually wider for longer maturities, shows how that
        compensation grows with time.

        Unlike the ICE BofA option-adjusted spreads (see `get_ice_bofa_option_adjusted_spread`),
        of which FRED only carries the last three years, this spread covers four decades, long
        enough to calibrate how credit spreads behave through several business cycles.

        The data comes from FRED, so a free FRED API key is required. The spreads are returned
        as decimal fractions (0.0079 for 0.79 percentage points).

        Also known as: corporate credit spread curve, credit spread term structure,
        investment grade spread.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The monthly spreads, indexed by month with a column per maturity.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-03-01', end_date='2026-08-31')

        fixedincome.get_hqm_corporate_bond_spread()
        ```

        Which returns:

        |         |     2Y |     5Y |    10Y |    30Y |
        |:--------|-------:|-------:|-------:|-------:|
        | 2026-03 | 0.0048 | 0.006  | 0.0078 | 0.0091 |
        | 2026-04 | 0.0042 | 0.0053 | 0.0071 | 0.0084 |
        | 2026-05 | 0.0037 | 0.005  | 0.007  | 0.0082 |
        | 2026-06 | 0.0035 | 0.0049 | 0.0071 | 0.0083 |
        | 2026-07 | 0.0035 | 0.0053 | 0.0075 | 0.0092 |
        | 2026-08 | 0.0036 | 0.0056 | 0.0079 | 0.0095 |
        """
        self._require_fred_api_key()

        hqm_spread = fred_model.get_hqm_corporate_bond_spread(
            self._start_date, self._end_date, self._fred_api_key
        )

        return finalize_dataset(
            dataset=hqm_spread,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_corporate_bond_yields(
        self,
        countries: list[str] | str | None = None,
        spread: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the yields of corporate bonds of a variety of countries, by rating where
        the source splits them, monthly and with long histories: the input for a credit
        spread factor outside the United States, where corporate bond indices by rating
        (iBoxx, ICE BofA) are licensed data.

        - Germany: the average yield on debt securities outstanding of non-MFI corporations
          the Bundesbank publishes monthly from 1957 (monthly averages), all ratings
          together ("All ratings"). No API key is needed.
        - Australia: the yields of non-financial corporate bonds rated A and BBB with a
          target tenor of 3, 5, 7 and 10 years the Reserve Bank of Australia publishes from
          2005 (end of month), e.g. "BBB 10Y". No API key is needed.
        - United States: Moody's seasoned Aaa and Baa corporate bond yields, from 1919. This
          requires a FRED API key (the fred_api_key of the FixedIncome class); see
          get_moodys_corporate_bond_yields for daily data.

        With spread=True each yield is returned as a spread over government bonds: for
        Germany over the yield on public debt securities outstanding (from 1956), for
        Australia over the Australian government bond yield of the same tenor (end of
        month, interpolated between 5 and 10 years for 7 years, from 2013), and for the
        United States over the 10-year Treasury yield.

        The yields are decimal fractions (0.0446 for 4.46%), with one column per country
        and rating.

        See definition: https://www.bundesbank.de/en/statistics/money-and-capital-markets

        Also known as: corporate credit spread, corporate bond yield by rating, euro credit
        spread, BBB spread.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "Germany", "Australia" and "United States". Defaults to None, which
                retrieves every country available, the United States only with a FRED API
                key.
            spread (bool, optional): Whether to return the spread over government bonds
                instead of the yield. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The yields (or spreads) as decimals, indexed by month with a column
            per country and rating.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-03-01', end_date='2026-08-31')

        corporate_bond_yields = fixedincome.get_corporate_bond_yields(
            countries=['Germany', 'Australia'], spread=True
        )

        corporate_bond_yields.loc[
            :, (slice(None), ['All ratings', 'A 5Y', 'BBB 5Y', 'BBB 10Y'])
        ].droplevel(0, axis=1)
        ```

        Which returns:

        |         |   All ratings |   A 5Y |   BBB 5Y |   BBB 10Y |
        |:--------|--------------:|-------:|---------:|----------:|
        | 2026-03 |        0.01   | 0.0097 |   0.0123 |    0.0114 |
        | 2026-04 |        0.0091 | 0.0085 |   0.011  |    0.01   |
        | 2026-05 |        0.0086 | 0.0089 |   0.011  |    0.0097 |
        | 2026-06 |        0.0084 | 0.0086 |   0.0103 |    0.0088 |
        | 2026-07 |        0.0085 | 0.0085 |   0.0102 |    0.0086 |
        | 2026-08 |        0.0085 | 0.0083 |   0.0099 |    0.0084 |
        """
        start_date = buffered_start_date(self._start_date, "monthly")

        def germany() -> pd.DataFrame:
            yields = bundesbank_model.get_corporate_bond_yield(
                start_date, self._end_date
            )

            if yields.empty:
                return yields

            corporate = (
                yields["Corporate"] - yields["Public"]
                if spread
                else yields["Corporate"]
            )

            return corporate.to_frame("All ratings")

        def australia() -> pd.DataFrame:
            yields = rba_model.get_corporate_bond_yields()

            if yields.empty or not spread:
                return yields

            government = rba_model.get_yield_curve()

            if government.empty:
                return pd.DataFrame()

            # The government yields on the last trading day of each month, at the tenors
            # of the corporate bonds.
            government = resample_to_period(government, "monthly")
            government["7Y"] = (government["5Y"] * 3 + government["10Y"] * 2) / 5

            return pd.DataFrame(
                {
                    column: yields[column] - government[column.split(" ")[1]]
                    for column in yields.columns
                }
            ).dropna(how="all")

        def united_states() -> pd.DataFrame:
            moodys = (
                fred_model.get_moodys_corporate_bond_spreads(
                    "monthly", start_date, self._end_date, self._fred_api_key
                )
                if spread
                else fred_model.get_moodys_corporate_bond_yields(
                    "monthly", start_date, self._end_date, self._fred_api_key
                )
            )

            return moodys[[column for column in ("Aaa", "Baa") if column in moodys]]

        sources = {"Germany": germany, "Australia": australia}

        if self._fred_api_key:
            sources["United States"] = united_states

        requested = (
            list(sources)
            if countries is None
            else [countries] if isinstance(countries, str) else list(countries)
        )

        if "United States" in requested and not self._fred_api_key:
            self._require_fred_api_key()

        if unavailable := [country for country in requested if country not in sources]:
            logger.warning(
                "Corporate bond yields are not available for %s. They cover %s.",
                ", ".join(unavailable),
                ", ".join(sources),
            )

        frames = {
            country: values
            for country in requested
            if country in sources and not (values := sources[country]()).empty
        }

        if not frames:
            return pd.DataFrame()

        corporate_bond_yields = pd.concat(frames, axis=1).sort_index()
        corporate_bond_yields.index.name = None
        corporate_bond_yields.columns.names = ["Country", "Rating"]

        return finalize_dataset(
            dataset=corporate_bond_yields,
            indicator_name="Corporate Bond Yields",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_rating_transition_matrix(
        self,
        agency: str = "S&P",
        rating_type: str = "corporate",
        year: int | None = None,
        horizon: int = 1,
        probabilities: bool = True,
        include_withdrawals: bool = False,
        region: str | None = None,
        rounding: int | None = None,
    ):
        """
        Retrieves the rating transition matrix of a credit rating agency: for every rating
        at the start of a period, the share of ratings that ended the period in each rating
        category, in default or withdrawn. It is the input of rating migration and default
        models, such as the Jarrow-Lando-Turnbull model, and of credit portfolio models.

        The data comes from CEREP, the central repository where every credit rating agency
        registered in the European Union reports its ratings to ESMA, published twice a
        year with periods from 1989. It covers the global ratings of agencies such as S&P,
        Moody's, Fitch, DBRS and Scope, for corporates (non-financial, financial and
        insurance), sovereigns, structured finance and covered bonds, in each agency's own
        rating scale. No API key is needed.

        By default the matrix is withdrawal-adjusted, the convention of the agencies' own
        default studies: ratings withdrawn during the period are left out, so each row sums
        to 1. The matrix compares the rating at the start of the period with the rating at
        its end, so a default that was followed by a new rating within the period (as after
        a distressed exchange) shows as that new rating; get_default_rates counts every
        default within the period and is the measure to calibrate default rates on. With include_withdrawals=True they are a separate "Withdrawals" column, and
        with probabilities=False the counts are returned instead.

        See definition: https://registers.esma.europa.eu/cerep-publication/

        Also known as: rating migration matrix, credit migration matrix, transition
        probabilities.

        Args:
            agency (str, optional): The rating agency, e.g. "S&P", "Moody's", "Fitch",
                "DBRS", "Scope" or "Kroll", or the CEREP code of any registered agency.
                Defaults to "S&P".
            rating_type (str, optional): "corporate", "sovereign", "structured_finance" or
                "covered_bonds". Defaults to "corporate".
            year (int | None, optional): The first year of the period. Defaults to None,
                which takes the latest year published.
            horizon (int, optional): The length of the period in years, from 1 January of
                year. Defaults to 1.
            probabilities (bool, optional): Whether to return the share of ratings per row
                instead of the number. Defaults to True.
            include_withdrawals (bool, optional): Whether to keep the ratings withdrawn
                during the period as a "Withdrawals" column. Defaults to False.
            region (str | None, optional): The area of the rated entities: "Africa",
                "America", "Asia", "Europe", "EU members", "International" or "Oceania".
                Defaults to None, which covers all of them.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The transition matrix, with a row per rating at the start of the
            period and a column per rating at its end.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2025-12-31")

        transition_matrix = fixedincome.get_rating_transition_matrix(agency='S&P', year=2025)

        transition_matrix.loc['AAA':'CCC', ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']]
        ```

        Which returns:

        | From Rating   |   AAA |     AA |      A |    BBB |     BB |      B |    CCC |      D |
        |:--------------|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
        | AAA           |     1 | 0      | 0      | 0      | 0      | 0      | 0      | 0      |
        | AA            |     0 | 0.9518 | 0.0482 | 0      | 0      | 0      | 0      | 0      |
        | A             |     0 | 0.0135 | 0.9646 | 0.0219 | 0      | 0      | 0      | 0      |
        | BBB           |     0 | 0.0006 | 0.0355 | 0.9563 | 0.007  | 0.0006 | 0      | 0      |
        | BB            |     0 | 0      | 0      | 0.0353 | 0.9283 | 0.0353 | 0.0011 | 0      |
        | B             |     0 | 0      | 0      | 0      | 0.0387 | 0.9001 | 0.0558 | 0.0023 |
        | CCC           |     0 | 0      | 0      | 0      | 0      | 0.18   | 0.76   | 0.03   |
        """
        agency_code = esma_model.resolve_agency(agency)

        if rating_type not in esma_model.RATING_TYPES:
            raise ValueError(
                f"The rating_type must be one of {', '.join(map(repr, esma_model.RATING_TYPES))}, "
                f"not {rating_type!r}."
            )
        if horizon < 1:
            raise ValueError(f"The horizon must be at least 1 year, not {horizon}.")

        # The statistics of a year are published in the first half of the next one.
        years = [year] if year else [datetime.now().year - 1, datetime.now().year - 2]
        matrix = pd.DataFrame()

        for start_year in years:
            matrix = esma_model.get_transition_matrix(
                agency_code,
                rating_type,
                pd.Timestamp(f"{start_year}-01-01"),
                pd.Timestamp(f"{start_year + horizon - 1}-12-31"),
                region,
            )
            if not matrix.empty:
                break

        if matrix.empty:
            return matrix

        if not include_withdrawals:
            matrix = matrix.drop(columns="Withdrawals", errors="ignore")

        # Ratings no longer outstanding at the start have no row of their own to speak of.
        matrix = matrix.loc[matrix.sum(axis=1) > 0]

        if probabilities:
            matrix = matrix.div(matrix.sum(axis=1), axis=0)

        matrix = finalize_dataset(
            dataset=matrix,
            start_date=None,
            end_date=None,
            default_rounding=self._rounding,
            rounding=rounding,
            apply_slice=False,
        )
        matrix.index.name = "From Rating"

        return matrix

    @handle_errors
    def get_default_rates(
        self,
        agency: str = "S&P",
        rating_type: str = "corporate",
        region: str | None = None,
        rounding: int | None = None,
    ):
        """
        Retrieves the one-year default rates per rating of a credit rating agency, year by
        year from 1989: the share of the ratings outstanding at the start of each year that
        defaulted within it. The history of default rates per rating calibrates the default
        intensity of credit risk models and shows how defaults cluster in recessions.

        The data comes from CEREP, the central repository where every credit rating agency
        registered in the European Union reports its ratings to ESMA, in each agency's own
        rating scale (AAA to C for S&P and Fitch, Aaa to C for Moody's). Only the years
        between the start and end date are retrieved, one request per year. No API key is
        needed. CEREP answers slowly (up to half a minute per year), so a long history takes
        a few minutes the first time; each year is cached afterwards. The years before an
        agency reported to CEREP read as zero defaults and are left out, so the history
        starts at the first year with a default.

        See definition: https://registers.esma.europa.eu/cerep-publication/

        Also known as: default frequency, annual default rate by rating, historical default
        rates.

        Args:
            agency (str, optional): The rating agency, e.g. "S&P", "Moody's", "Fitch",
                "DBRS", "Scope" or "Kroll", or the CEREP code of any registered agency.
                Defaults to "S&P".
            rating_type (str, optional): "corporate", "sovereign", "structured_finance" or
                "covered_bonds". Defaults to "corporate".
            region (str | None, optional): The area of the rated entities: "Africa",
                "America", "Asia", "Europe", "EU members", "International" or "Oceania".
                Defaults to None, which covers all of them.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The default rates as decimals, indexed by year with a column per
            rating.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2018-01-01', end_date='2025-12-31')

        default_rates = fixedincome.get_default_rates(agency='S&P')

        default_rates[['BBB', 'BB', 'B', 'CCC']]
        ```

        Which returns:

        |      |    BBB |     BB |      B |    CCC |
        |:-----|-------:|-------:|-------:|-------:|
        | 2018 | 0      | 0      | 0.0094 | 0.2617 |
        | 2019 | 0.0016 | 0      | 0.0182 | 0.3534 |
        | 2020 | 0      | 0.0097 | 0.038  | 0.4661 |
        | 2021 | 0      | 0      | 0.0039 | 0.0895 |
        | 2022 | 0      | 0.0046 | 0.0115 | 0.1402 |
        | 2023 | 0.0006 | 0.001  | 0.0138 | 0.3116 |
        | 2024 | 0      | 0.0021 | 0.0188 | 0.2852 |
        | 2025 | 0      | 0.0011 | 0.0146 | 0.252  |
        """
        agency_code = esma_model.resolve_agency(agency)

        if rating_type not in esma_model.RATING_TYPES:
            raise ValueError(
                f"The rating_type must be one of {', '.join(map(repr, esma_model.RATING_TYPES))}, "
                f"not {rating_type!r}."
            )

        # CEREP's periods start in 1989; a year is published in the first half of the next.
        first_year = max(pd.Timestamp(self._start_date).year, 1989)
        last_year = min(pd.Timestamp(self._end_date).year, datetime.now().year - 1)
        years = list(range(first_year, last_year + 1))

        if not years:
            logger.warning(
                "CEREP publishes default rates for the years from 1989 to %s.",
                datetime.now().year - 1,
            )
            return pd.DataFrame()

        results = helpers.run_in_parallel(
            lambda year: esma_model.get_default_rates(
                agency_code,
                rating_type,
                pd.Timestamp(f"{year}-01-01"),
                pd.Timestamp(f"{year}-12-31"),
                region,
            ),
            [(year,) for year in years],
            max_workers=6,
        )
        default_rates = {
            year: rates["Default Rate"]
            for year, rates in zip(years, results, strict=True)
            if not rates.empty
        }

        if not default_rates:
            return pd.DataFrame()

        default_rates = pd.DataFrame(default_rates).T
        default_rates.index = pd.PeriodIndex(
            [str(year) for year in default_rates.index], freq="Y"
        )

        # The years before an agency reported to CEREP read as zero for every rating, so
        # the history starts at the first year with a default.
        with_defaults = default_rates.fillna(0).gt(0).any(axis=1)
        default_rates = (
            default_rates.loc[with_defaults.idxmax() :]
            if with_defaults.any()
            else default_rates
        )

        return finalize_dataset(
            dataset=default_rates.dropna(how="all", axis=1),
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_moodys_corporate_bond_yields(
        self,
        period: str = "daily",
        spread: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves Moody's seasoned corporate bond yields for Aaa (the highest rating) and Baa
        (the lowest investment grade rating) US corporate bonds, the longest running corporate
        bond yield benchmark there is: daily from 1986 and monthly from 1919.

        With spread=True the spreads over the 10-year Treasury yield are returned instead,
        together with the Baa minus Aaa spread, the classic measure of default risk in the
        academic literature (e.g. Fama and French, 1989), which widens sharply in recessions.
        Monthly spreads start in 1953, when the monthly 10-year Treasury yield does.

        Monthly values are the averages over the month, as Moody's publishes them; weekly
        values are the yields on the last day of each week (weeks end on Friday).

        The data comes from FRED, so a free FRED API key is required. The yields and spreads
        are returned as decimal fractions (0.0675 for 6.75%).

        Also known as: Moody's Aaa, Moody's Baa, corporate bond yields by rating, default
        spread, credit spread.

        Args:
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            spread (bool, optional): Whether to return the spreads over the 10-year Treasury
                yield instead of the yields. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The "Aaa" and "Baa" yields, or with spread=True the "Aaa", "Baa" and
            "Baa - Aaa" spreads.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-03-01', end_date='2026-08-31')

        fixedincome.get_moodys_corporate_bond_yields(period='monthly', spread=True)
        ```

        Which returns:

        |         |    Aaa |    Baa |   Baa - Aaa |
        |:--------|-------:|-------:|------------:|
        | 2026-03 | 0.0123 | 0.0179 |      0.0056 |
        | 2026-04 | 0.011  | 0.0171 |      0.0061 |
        | 2026-05 | 0.0108 | 0.0162 |      0.0054 |
        | 2026-06 | 0.0105 | 0.0153 |      0.0048 |
        | 2026-07 | 0.0116 | 0.0159 |      0.0043 |
        | 2026-08 | 0.012  | 0.0164 |      0.0044 |
        """
        self._require_fred_api_key()
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "Moody's corporate bond yield"
        )
        frequency = "monthly" if period == "monthly" else "daily"

        moodys = (
            fred_model.get_moodys_corporate_bond_spreads(
                frequency, self._start_date, self._end_date, self._fred_api_key
            )
            if spread
            else fred_model.get_moodys_corporate_bond_yields(
                frequency, self._start_date, self._end_date, self._fred_api_key
            )
        )

        if period == "weekly":
            moodys = resample_to_period(moodys, "weekly")

        return finalize_dataset(
            dataset=moodys,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_excess_bond_premium(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the credit spread of Gilchrist and Zakrajšek (2012) and its excess bond
        premium, monthly from 1973, as the Federal Reserve Board updates them. The GZ credit
        spread is the average spread of US corporate bonds over Treasuries with the same cash
        flows, built bond by bond from the secondary market. The excess bond premium is the
        part of that spread that expected defaults do not explain: a measure of investors'
        appetite for credit risk, and one of the best predictors of economic activity in the
        literature. The Recession Probability column is the probability of a recession over
        the next twelve months the premium implies.

        Gilchrist, S., & Zakrajšek, E. (2012). Credit Spreads and Business Cycle
        Fluctuations. American Economic Review, 102(4), 1692-1720.
        https://doi.org/10.1257/aer.102.4.1692

        No API key is needed. The spread and premium are returned as decimal fractions
        (0.0084 for 0.84 percentage points) and the probability as a fraction (0.108 for
        10.8%).

        See definition: https://www.federalreserve.gov/econres/notes/feds-notes/updating-the-recession-risk-and-the-excess-bond-premium-20161006.html

        Also known as: GZ spread, Gilchrist-Zakrajšek spread, EBP, credit market sentiment.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The "GZ Credit Spread", "Excess Bond Premium" and "Recession
            Probability", indexed by month.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-02-01', end_date='2026-07-31')

        fixedincome.get_excess_bond_premium()
        ```

        Which returns:

        | date    |   GZ Credit Spread |   Excess Bond Premium |   Recession Probability |
        |:--------|-------------------:|----------------------:|------------------------:|
        | 2026-02 |             0.0097 |               -0.0026 |                  0.1222 |
        | 2026-03 |             0.0103 |               -0.0026 |                  0.1208 |
        | 2026-04 |             0.0092 |               -0.002  |                  0.1374 |
        | 2026-05 |             0.0083 |               -0.0037 |                  0.0951 |
        | 2026-06 |             0.0086 |               -0.0028 |                  0.1152 |
        | 2026-07 |             0.0084 |               -0.0031 |                  0.1093 |
        """
        excess_bond_premium = frb_model.get_excess_bond_premium()

        return finalize_dataset(
            dataset=excess_bond_premium,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_eiopa_risk_free_rate(
        self,
        countries: list[str] | str | None = None,
        curve: str = "spot_no_va",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the risk-free interest rate term structures the European Insurance and
        Occupational Pensions Authority (EIOPA) publishes monthly: the curves European
        insurers must discount their liabilities with under Solvency II. They cover the euro
        and the currencies of every country of the European Economic Area, plus Switzerland,
        the United Kingdom, Australia, Canada, China, Colombia, Hong Kong, Japan, Taiwan and
        the United States, for maturities of 1 to 150 years. Beyond the last liquid maturity
        each curve converges to the ultimate forward rate (UFR), as Solvency II prescribes.

        Four curves are published: the basic spot curve (curve="spot_no_va"), the spot curve
        with the volatility adjustment (curve="spot_with_va"), and the basic curve after the
        interest rate shocks of the Solvency II standard formula (curve="shock_up" and
        curve="shock_down"). EIOPA ships the shocked worksheets as formulas without computed
        values, so they are computed here with those formulas: upwards by the relative shock
        per maturity and at least one percentage point, downwards by the relative shock with
        negative rates left unchanged. Every value is a decimal fraction (0.0358 for 3.58%),
        at the end of the month the release is for.

        No API key is needed. EIOPA's page links the releases from January 2023; each is a
        separate file, so only the months between the start and end date are downloaded, and
        a release is cached for a year since it does not change.

        See definition: https://www.eiopa.europa.eu/tools-and-data/risk-free-interest-rate-term-structures_en

        Also known as: Solvency II discount curve, EIOPA RFR, risk-free rate term structure,
        UFR curve.

        Args:
            countries (list[str] | str | None, optional): The currencies or countries to include,
                as EIOPA names them, e.g. 'Euro Area', 'United Kingdom' or 'United States'.
                Defaults to None, which returns all of them.
            curve (str, optional): "spot_no_va", "spot_with_va", "shock_up" or "shock_down".
                Defaults to "spot_no_va".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The rates, indexed by month with a column per currency or country and
            maturity ("1Y" to "150Y").

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-07-01', end_date='2026-09-30')

        curves = fixedincome.get_eiopa_risk_free_rate(countries=['Euro Area', 'United Kingdom'])

        curves.loc[:, (slice(None), ['1Y', '10Y', '30Y', '60Y', '150Y'])]
        ```

        Which returns:

        |         |   ('Euro Area', '1Y') |   ('United Kingdom', '1Y') |   ('Euro Area', '10Y') |   ('United Kingdom', '10Y') |   ('Euro Area', '30Y') |
        |:--------|----------------------:|---------------------------:|-----------------------:|----------------------------:|-----------------------:|
        | 2026-07 |                0.0283 |                     0.0415 |                 0.0316 |                      0.0468 |                 0.0337 |
        | 2026-08 |                0.0292 |                     0.0418 |                 0.0327 |                      0.0473 |                 0.0345 |
        | 2026-09 |                0.0327 |                     0.0445 |                 0.0358 |                      0.0501 |                 0.0351 |
        """
        if curve not in {**eiopa_model.CURVES, **eiopa_model.SHOCKED_CURVES}:
            raise ValueError(
                f"The curve must be one of {', '.join({**eiopa_model.CURVES, **eiopa_model.SHOCKED_CURVES})}, not {curve!r}."
            )
        if countries is not None and not isinstance(countries, str | list | tuple):
            raise TypeError(
                "The countries must be a name or a list of names, such as 'Euro Area' or "
                f"['Euro Area', 'Japan'], not a {type(countries).__name__} ({countries!r})."
            )

        term_structures = eiopa_model.get_risk_free_rate_term_structures(
            curve, self._start_date, self._end_date
        )

        if countries is not None and not term_structures.empty:
            requested = [countries] if isinstance(countries, str) else list(countries)
            available = list(dict.fromkeys(term_structures.columns.get_level_values(0)))

            if unknown := [name for name in requested if name not in available]:
                logger.warning(
                    "EIOPA publishes no risk-free rates for %s. It covers %s.",
                    ", ".join(unknown),
                    ", ".join(available),
                )

            term_structures = term_structures.loc[
                :, term_structures.columns.get_level_values(0).isin(requested)
            ]

        return finalize_dataset(
            dataset=term_structures,
            indicator_name="EIOPA Risk-Free Rate",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_eiopa_symmetric_adjustment(
        self,
        period: str = "daily",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the symmetric adjustment of the equity capital charge of Solvency II, the
        "equity dampener" EIOPA publishes, daily from 1991, and the equity charges of the
        standard formula it results in. The adjustment raises the charge when equity prices
        are above their three-year average and lowers it when they are below, between -10%
        and +10%, so insurers are not forced to sell equities into a falling market. With
        get_eiopa_risk_free_rate it completes the market inputs of the standard formula's
        market risk module.

        The "Type 1 Equity Charge" (equities listed in the EEA or OECD) is 39% plus the
        adjustment and the "Type 2 Equity Charge" (other equities) 49% plus the adjustment.
        Every value is a decimal fraction (0.0771 for 7.71%). No API key is needed. Weekly
        and monthly periods take the value on the last day of each period, which is the
        value insurers apply at that reporting date.

        See definition: https://www.eiopa.europa.eu/tools-and-data/symmetric-adjustment-equity-capital-charge_en

        Also known as: equity dampener, symmetric adjustment, Solvency II equity charge.

        Args:
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The symmetric adjustment and the type 1 and type 2 equity charges,
            indexed by date.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-04-01', end_date='2026-09-30')

        fixedincome.get_eiopa_symmetric_adjustment(period='monthly')
        ```

        Which returns:

        |         |   Symmetric Adjustment |   Type 1 Equity Charge |   Type 2 Equity Charge |
        |:--------|-----------------------:|-----------------------:|-----------------------:|
        | 2026-04 |                 0.0767 |                 0.4667 |                 0.5667 |
        | 2026-05 |                 0.0868 |                 0.4768 |                 0.5768 |
        | 2026-06 |                 0.0894 |                 0.4794 |                 0.5794 |
        | 2026-07 |                 0.0951 |                 0.4851 |                 0.5851 |
        | 2026-08 |                 0.0938 |                 0.4838 |                 0.5838 |
        | 2026-09 |                 0.0771 |                 0.4671 |                 0.5671 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "EIOPA symmetric adjustment"
        )
        symmetric_adjustment = resample_to_period(
            eiopa_model.get_symmetric_adjustment(), period
        )
        if not symmetric_adjustment.empty:
            symmetric_adjustment.index.name = None

        return finalize_dataset(
            dataset=symmetric_adjustment,
            indicator_name="EIOPA Symmetric Adjustment",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_bank_of_england_yield_curve(
        self,
        curve: str = "nominal",
        period: str = "daily",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the UK yield curves the Bank of England estimates daily from 1985, for
        maturities from 0.5 years to 40 years (25 years for OIS): the nominal spot curve of
        UK government bonds (gilts) and the spot curve of overnight index swaps on SONIA (OIS,
        from 2009). Unlike the par yields of 5, 10 and 20 years in
        `get_government_bond_yield_curve`, these are zero-coupon spot rates over the full
        range of maturities with history back to 1985.

        The Bank of England's real and implied inflation curves are part of
        `economics.get_real_yield_curve` and `economics.get_breakeven_inflation_expectations`
        (countries='United Kingdom').

        No API key is needed. The archive is a set of large files, of which only those
        covering the requested years are read, and they are cached since past years do not
        change. Weekly and monthly periods take the curve on the last day of each period.

        See definition: https://www.bankofengland.co.uk/statistics/yield-curves

        Also known as: gilt curve, UK spot curve, SONIA OIS curve.

        Args:
            curve (str, optional): "nominal" or "ois". Defaults to "nominal".
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The spot rates as decimals, indexed by date with a column per
            maturity in years, in steps of half a year.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-04-01', end_date='2026-09-30')

        fixedincome.get_bank_of_england_yield_curve(period='monthly')[
            ['1Y', '3Y', '5Y', '10Y', '20Y', '30Y']
        ]
        ```

        Which returns:

        |         |     1Y |     3Y |     5Y |    10Y |    20Y |    30Y |
        |:--------|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2026-04 | 0.0426 | 0.0433 | 0.0447 | 0.0506 | 0.0573 | 0.0581 |
        | 2026-05 | 0.0403 | 0.0414 | 0.043  | 0.0486 | 0.0555 | 0.0563 |
        | 2026-06 | 0.0402 | 0.041  | 0.0427 | 0.0483 | 0.0552 | 0.0561 |
        | 2026-07 | 0.0417 | 0.0438 | 0.0456 | 0.051  | 0.058  | 0.0589 |
        | 2026-08 | 0.0418 | 0.0441 | 0.046  | 0.0514 | 0.0582 | 0.0591 |
        | 2026-09 | 0.0442 | 0.0478 | 0.0494 | 0.054  | 0.0598 | 0.06   |
        """
        if curve in ("real", "inflation"):
            raise ValueError(
                f"The {curve} curve of the Bank of England is part of economics."
                + (
                    "get_real_yield_curve"
                    if curve == "real"
                    else "get_breakeven_inflation_expectations"
                )
                + "(countries='United Kingdom')."
            )
        if curve not in ("nominal", "ois"):
            raise ValueError(f"The curve must be 'nominal' or 'ois', not {curve!r}.")
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "Bank of England yield curve"
        )

        spot_curve = resample_to_period(
            boe_model.get_spot_curve(
                curve, buffered_start_date(self._start_date, period), self._end_date
            ),
            period,
        )

        return finalize_dataset(
            dataset=spot_curve,
            indicator_name="Bank of England Yield Curve",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_corporate_borrowing_cost(
        self,
        countries: list[str] | str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the composite cost of borrowing for non-financial corporations the European
        Central Bank publishes monthly from 2003, for the euro area and every member: the
        average interest rate on new bank loans to companies across maturities and loan sizes.
        Euro area companies borrow mostly from banks rather than in the bond market, which
        makes this the broadest measure of their cost of credit, and the closest freely
        available proxy for euro area corporate credit conditions, since euro area corporate
        bond indices are licensed data.

        No API key is needed. The rate is a decimal fraction (0.0377 for 3.77%).

        See definition: https://data.ecb.europa.eu/data/datasets/MIR

        Also known as: cost of borrowing for corporations, euro area lending rate, corporate
        credit cost.

        Args:
            countries (list[str] | str | None, optional): The countries to include, e.g.
                'Euro Area' or ['Germany', 'Italy']. Defaults to None, which returns all of them.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The rate, indexed by month with a column per country.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-03-01', end_date='2026-08-31')

        fixedincome.get_corporate_borrowing_cost(
            countries=['Euro Area', 'Germany', 'France', 'Italy', 'Spain']
        )
        ```

        Which returns:

        |         |   Euro Area |   Germany |   France |   Italy |   Spain |
        |:--------|------------:|----------:|---------:|--------:|--------:|
        | 2026-03 |      0.0358 |    0.0381 |   0.035  |  0.0349 |  0.0328 |
        | 2026-04 |      0.0362 |    0.0378 |   0.0353 |  0.0365 |  0.0345 |
        | 2026-05 |      0.0363 |    0.0372 |   0.0352 |  0.0377 |  0.0351 |
        | 2026-06 |      0.0379 |    0.04   |   0.0367 |  0.0377 |  0.0355 |
        | 2026-07 |      0.038  |    0.0398 |   0.0363 |  0.0386 |  0.037  |
        | 2026-08 |      0.0377 |    0.0388 |   0.037  |  0.0385 |  0.0363 |
        """
        borrowing_cost = economics_ecb_model.get_corporate_borrowing_cost(
            buffered_start_date(self._start_date, "monthly"), self._end_date
        )

        return finalize_dataset(
            dataset=borrowing_cost,
            indicator_name="Corporate Borrowing Cost",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            countries=countries,
            dropna=True,
        )

    @handle_errors
    def get_financial_stress_index(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the Composite Indicator of Systemic Stress (CISS) the European Central Bank
        publishes daily for the euro area, its largest members, the United Kingdom, the United
        States and China, back to 1980 for some. The index combines stress in money, bond,
        equity and foreign exchange markets and among financial intermediaries into a number
        between 0 (calm) and 1 (crisis), weighting the segments more heavily when they are
        stressed at the same time; it peaked in 2008 and in the euro area debt crisis.

        Hollo, D., Kremer, M., & Lo Duca, M. (2012). CISS - A Composite Indicator of
        Systemic Stress in the Financial System. ECB Working Paper No. 1426.

        No API key is needed. Weekly and monthly periods take the index on the last day of
        each period.

        See definition: https://data.ecb.europa.eu/data/datasets/CISS

        Also known as: CISS, systemic stress, financial stress, financial conditions.

        Args:
            countries (list[str] | str | None, optional): The areas to include, e.g. 'Euro Area'
                or ['United States', 'United Kingdom']. Defaults to None, which returns all of them.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The index, indexed by date with a column per area.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-04-01', end_date='2026-09-30')

        fixedincome.get_financial_stress_index(
            countries=['Euro Area', 'United States', 'United Kingdom', 'China'], period='monthly'
        )
        ```

        Which returns:

        |         |   Euro Area |   United States |   United Kingdom |   China |
        |:--------|------------:|----------------:|-----------------:|--------:|
        | 2026-04 |      0.004  |          0.0134 |           0.0222 |  0.0199 |
        | 2026-05 |      0.0058 |          0.0069 |           0.0309 |  0.0059 |
        | 2026-06 |      0.0098 |          0.0095 |           0.0072 |  0.0308 |
        | 2026-07 |      0.0129 |          0.0493 |           0.0075 |  0.0209 |
        | 2026-08 |      0.0205 |          0.0194 |           0.0011 |  0.0072 |
        | 2026-09 |      0.0132 |          0.0092 |           0.0122 |  0.0015 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "financial stress index"
        )

        stress_index = resample_to_period(
            economics_ecb_model.get_financial_stress_index(
                buffered_start_date(self._start_date, period), self._end_date
            ),
            period,
        )

        return finalize_dataset(
            dataset=stress_index,
            indicator_name="Financial Stress Index",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            countries=countries,
            dropna=True,
        )

    @handle_errors
    def get_euribor_rates(
        self,
        maturities: str | list | None = None,
        nominal: bool = True,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Euribor rates, short for Euro Interbank Offered Rate, are the interest rates at which a panel
        of European banks lend funds to one another in the interbank market. These rates are published
        daily by the European Money Markets Institute (EMMI) and serve as a benchmark for various
        financial products and contracts, including mortgages, loans, and derivatives, across the Eurozone.

        The Euribor rates are determined for different maturities, typically ranging from overnight to 12 months.
        The most common maturities are 1 month, 3 months, 6 months, and 12 months. Each maturity represents
        the time period for which the funds are borrowed, with longer maturities generally implying higher
        interest rates due to increased uncertainty and risk over longer time horizons.

        For more information, see for example: https://data.ecb.europa.eu/data/datasets/FM/FM.M.U2.EUR.RT.MM.EURIBOR6MD_.HSTA

        Also known as: euro interbank offered rate, eurozone money market.

        Args:
            maturities (str | list | None, optional): Maturities for which to retrieve rates. Defaults to None.
                When set to None, it will retrieve rates for 1 month, 3 months, 6 months, and 12 months.
            nominal (bool, optional): Whether to retrieve the nominal Euribor fixings or their real
                (inflation-adjusted) counterpart. The ECB only publishes a real Euribor for the 3-month
                maturity, so nominal=False returns that maturity alone and warns about any others that
                were requested rather than silently answering them with a nominal rate. Defaults to True.
            rounding (int | None, optional): Rounding precision for the rates. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.

        Returns:
            pandas.DataFrame: DataFrame containing the Euribor rates for the specified maturities.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2023-12-01', end_date="2025-12-31")

        euribor_rates = fixedincome.get_euribor_rates()
        ```

        Which returns:

        |         |   1-Month |   3-Month |   6-Month |   12-Month |
        |:--------|----------:|----------:|----------:|-----------:|
        | 2023-12 |    0.0386 |    0.0393 |    0.0393 |     0.0368 |
        | 2024-01 |    0.0387 |    0.0393 |    0.0389 |     0.0361 |
        | 2024-02 |    0.0387 |    0.0392 |    0.039  |     0.0367 |
        | 2024-03 |    0.0385 |    0.0392 |    0.0389 |     0.0372 |
        | 2024-04 |    0.0385 |    0.0389 |    0.0384 |     0.037  |
        | 2024-05 |    0.0382 |    0.0381 |    0.0379 |     0.0368 |
        | 2024-06 |    0.0363 |    0.0372 |    0.0371 |     0.0365 |
        | 2024-07 |    0.0362 |    0.0368 |    0.0364 |     0.0353 |
        | 2024-08 |    0.036  |    0.0355 |    0.0342 |     0.0317 |
        | 2024-09 |    0.0344 |    0.0343 |    0.0326 |     0.0294 |
        | 2024-10 |    0.0321 |    0.0317 |    0.03   |     0.0269 |
        | 2024-11 |    0.0307 |    0.0301 |    0.0279 |     0.0251 |
        | 2024-12 |    0.0289 |    0.0282 |    0.0263 |     0.0244 |
        | 2025-01 |    0.0279 |    0.027  |    0.0261 |     0.0253 |
        | 2025-02 |    0.0261 |    0.0252 |    0.0246 |     0.0241 |
        | 2025-03 |    0.024  |    0.0244 |    0.0239 |     0.024  |
        | 2025-04 |    0.0224 |    0.0225 |    0.022  |     0.0214 |
        | 2025-05 |    0.0209 |    0.0209 |    0.0212 |     0.0208 |
        | 2025-06 |    0.0193 |    0.0198 |    0.0205 |     0.0208 |
        | 2025-07 |    0.0189 |    0.0199 |    0.0206 |     0.0208 |
        | 2025-08 |    0.0189 |    0.0202 |    0.0208 |     0.0211 |
        | 2025-09 |    0.019  |    0.0203 |    0.021  |     0.0217 |
        | 2025-10 |    0.0191 |    0.0203 |    0.0211 |     0.0219 |
        | 2025-11 |    0.0191 |    0.0204 |    0.0213 |     0.0222 |
        | 2025-12 |    0.0192 |    0.0205 |    0.0214 |     0.0227 |
        """
        if isinstance(maturities, str):
            maturities = [maturities]

        maturity_names = {
            "1M": "1-Month",
            "3M": "3-Month",
            "6M": "6-Month",
            "1Y": "12-Month",
        }

        maturities = ["1M", "3M", "6M", "1Y"] if maturities is None else maturities

        collected_rates = {}
        unavailable_real_maturities = []

        for maturity in maturities:
            if maturity not in ["1M", "3M", "6M", "1Y"]:
                logger.error(
                    "Invalid maturity: %s, please choose from 1M, 3M, 6M, 1Y.", maturity
                )
                continue

            # Only the 3-Month Euribor is published as a real rate; the others are left out rather than silently answered with their nominal rate.
            if not nominal and maturity != "3M":
                unavailable_real_maturities.append(maturity)
                continue

            collected_rates[maturity_names[maturity]] = euribor_model.get_euribor_rate(
                maturity=maturity, nominal=nominal
            )

        if unavailable_real_maturities:
            logger.warning(
                "Only the 3-Month Euribor rate is available as a real rate, so no data is "
                "returned for the following maturities: %s.",
                ", ".join(unavailable_real_maturities),
            )

        # Concatenated in one go so that a maturity with a longer history is not truncated to the index of whichever maturity happened to be assigned first.
        euribor_rates = (
            pd.concat(collected_rates, axis=1) if collected_rates else pd.DataFrame()
        )

        return finalize_dataset(
            dataset=euribor_rates,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    def get_european_central_bank_rates(
        self,
        rate: str | None = None,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        The Governing Council of the ECB sets the key interest rates for the
        euro area. The available rates are:

        - Main refinancing operations (refinancing)
        - Marginal lending facility (lending)
        - Deposit facility (deposit)

        The main refinancing operations (MRO) rate is the interest rate banks
        pay when they borrow money from the ECB for one week. When they do this,
        they have to provide collateral to guarantee that the money will be paid back.

        The marginal lending facility rate is the interest rate banks pay when they
        borrow from the ECB overnight. When they do this, they have to provide collateral,
        for example securities, to guarantee that the money will be paid back.

        The deposit facility rate is one of the three interest rates the ECB sets every
        six weeks as part of its monetary policy. The rate defines the interest banks
        receive for depositing money with the central bank overnight.

        See source: https://data.ecb.europa.eu/main-figures/

        Also known as: ECB rates, deposit facility rate.

        Args:
            rate (str, optional): The rate to return. Defaults to None, which returns all rates.
                Choose between 'refinancing', 'lending' or 'deposit'.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the ECB rates.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2023-12-01', end_date="2025-12-31")

        fixedincome.get_european_central_bank_rates()
        ```

        Which returns:

        |            |   Refinancing |   Lending |   Deposit |
        |:-----------|--------------:|----------:|----------:|
        | 2025-12-22 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-23 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-24 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-25 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-26 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-27 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-28 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-29 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-30 |        0.0215 |     0.024 |      0.02 |
        | 2025-12-31 |        0.0215 |     0.024 |      0.02 |
        """
        if rate and rate not in ["refinancing", "lending", "deposit"]:
            raise ValueError(
                "Rate must be one of 'refinancing', 'lending' or 'deposit' or left empty for all."
            )

        collected_rates = {}

        if not rate or rate == "refinancing":
            collected_rates["Refinancing"] = ecb_model.get_main_refinancing_operations()
        if not rate or rate == "lending":
            collected_rates["Lending"] = ecb_model.get_marginal_lending_facility()
        if not rate or rate == "deposit":
            collected_rates["Deposit"] = ecb_model.get_deposit_facility()

        # Concatenated in one go so that a rate with a longer history is not truncated to the index of whichever rate happened to be assigned to the frame first.
        ecb_rates = (
            pd.concat(collected_rates, axis=1) if collected_rates else pd.DataFrame()
        )

        return finalize_dataset(
            dataset=ecb_rates,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    def get_federal_reserve_rates(
        self,
        rate: str = "EFFR",
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Get the Federal Reserve rates as published by the Federal Reserve Bank of New York.
        The federal funds market consists of domestic unsecured borrowings in U.S. dollars
        by depository institutions from other depository institutions and certain other
        entities, primarily government-sponsored enterprises.

        The following rates are available:

        - Effective Federal Funds Rate (EFFR)
        - Overnight Bank Funding Rate (OBFR)
        - Tri-Party General Collateral Rate (TGCR)
        - Broad General Collateral Rate (BGCR)
        - Secured Overnight Financing Rate (SOFR)

        The effective federal funds rate (EFFR) is calculated as a volume-weighted median
        of overnight federal funds transactions reported in the FR 2420 Report of Selected
        Money Market Rates.

        The overnight bank funding rate (OBFR) is calculated as a volume-weighted median
        of overnight federal funds transactions, Eurodollar transactions, and the
        domestic deposits reported as “Selected Deposits” in the FR 2420 Report.

        The TGCR is calculated as a volume-weighted median of transaction-level
        tri-party repo data collected from the Bank of New York Mellon.

        The BGCR is calculated as a volume-weighted median of transaction-level
        tri-party repo data collected from the Bank of New York Mellon as well
        as GCF Repo transaction data obtained from the U.S. Department of the
        Treasury's Office of Financial Research (OFR).

        The SOFR is calculated as a volume-weighted median of transaction-level
        tri-party repo data collected from the Bank of New York Mellon as well as
        GCF Repo transaction data and data on bilateral Treasury repo transactions
        cleared through FICC's DVP service, which are obtained from the U.S.
        Department of the Treasury's Office of Financial Research (OFR).

        The New York Fed publishes the rates for the prior business day on the New
        York Fed's website between 8:00 and 9:00 a.m.

        See source: https://www.newyorkfed.org/markets/reference-rates/

        Also known as: Fed rates, federal funds rate, FOMC rate.

        Args:
            rate (str): The rate to return. Defaults to 'EFFR' (Effective Federal Funds Rate).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the Federal Reserve rates including the rate,
                percentiles, volume and upper and lower bounds.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2023-12-01', end_date="2025-12-31")

        effr = fixedincome.get_federal_reserve_rates()

        effr.loc[:, ['Rate', '1st Percentile', '25th Percentile', '75th Percentile', '99th Percentile']]
        ```

        Which returns:

        | Effective Date   |   Rate |   1st Percentile |   25th Percentile |   75th Percentile |   99th Percentile |
        |:-----------------|-------:|-----------------:|------------------:|------------------:|------------------:|
        | 2025-12-17       | 0.0364 |            0.036 |            0.0363 |            0.0365 |            0.0366 |
        | 2025-12-18       | 0.0364 |            0.036 |            0.0364 |            0.0365 |            0.0366 |
        | 2025-12-19       | 0.0364 |            0.036 |            0.0364 |            0.0365 |            0.0365 |
        | 2025-12-22       | 0.0364 |            0.036 |            0.0364 |            0.0365 |            0.0365 |
        | 2025-12-23       | 0.0364 |            0.036 |            0.0363 |            0.0365 |            0.0365 |
        | 2025-12-24       | 0.0364 |            0.036 |            0.0364 |            0.0365 |            0.0365 |
        | 2025-12-26       | 0.0364 |            0.036 |            0.0363 |            0.0365 |            0.0365 |
        | 2025-12-29       | 0.0364 |            0.036 |            0.0363 |            0.0365 |            0.0368 |
        | 2025-12-30       | 0.0364 |            0.036 |            0.0364 |            0.0365 |            0.0368 |
        | 2025-12-31       | 0.0364 |            0.036 |            0.0364 |            0.0365 |            0.0369 |
        """
        rate = rate.upper()

        if rate == "EFFR":
            fed_data = fed_model.get_effective_federal_funds_rate()
        elif rate == "OBFR":
            fed_data = fed_model.get_overnight_banking_funding_rate()
        elif rate == "TGCR":
            fed_data = fed_model.get_tri_party_general_collateral_rate()
        elif rate == "BGCR":
            fed_data = fed_model.get_broad_general_collateral_rate()
        elif rate == "SOFR":
            fed_data = fed_model.get_secured_overnight_financing_rate()
        else:
            raise ValueError(
                "Rate must be one of 'EFFR', 'OBFR', 'TGCR', 'BGCR' or 'SOFR'."
            )

        return finalize_dataset(
            dataset=fed_data,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )
