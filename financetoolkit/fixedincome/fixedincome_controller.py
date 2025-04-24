"""Fixed Income Module"""

__docformat__ = "google"


import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from financetoolkit.economics import oecd_model
from financetoolkit.fixedincome import (
    bond_model,
    derivative_model,
    ecb_model,
    euribor_model,
    fed_model,
    fred_model,
)
from financetoolkit.helpers import calculate_growth
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.error_model import handle_errors

logger = logger_model.get_logger()

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,
# pylint: disable=too-many-locals,line-too-long,too-many-public-methods
# ruff: noqa: E501


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
    ):
        """
        Initializes the Fixed Income Controller Class.

        Args:
            start_date (str | None, optional): The start date to retrieve data from. Defaults to None.
            end_date (str | None, optional): The end date to retrieve data from. Defaults to None.
            quarterly (bool, optional): Whether to return the data quarterly. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

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

        | Date       |    AAA |     AA |      A |    BBB |     BB |      B |    CCC |
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2024-01-01 | 0.0456 | 0.047  | 0.0505 | 0.054  | 0.0613 | 0.0752 | 0.1319 |
        | 2024-01-02 | 0.0459 | 0.0473 | 0.0509 | 0.0543 | 0.0622 | 0.0763 | 0.1333 |
        | 2024-01-03 | 0.0459 | 0.0474 | 0.051  | 0.0544 | 0.0634 | 0.0779 | 0.1358 |
        | 2024-01-04 | 0.0466 | 0.0481 | 0.0518 | 0.0551 | 0.0639 | 0.0784 | 0.1367 |
        | 2024-01-05 | 0.047  | 0.0485 | 0.0521 | 0.0554 | 0.0641 | 0.0787 | 0.137  |
        | 2024-01-08 | 0.0465 | 0.0481 | 0.0517 | 0.055  | 0.0633 | 0.0776 | 0.1365 |
        | 2024-01-09 | 0.0464 | 0.048  | 0.0516 | 0.0548 | 0.0629 | 0.0771 | 0.1359 |
        | 2024-01-10 | 0.0464 | 0.048  | 0.0515 | 0.0547 | 0.0622 | 0.0762 | 0.1351 |
        | 2024-01-11 | 0.0456 | 0.0472 | 0.0507 | 0.054  | 0.0619 | 0.076  | 0.1344 |
        | 2024-01-12 | 0.0451 | 0.0467 | 0.0502 | 0.0534 | 0.0613 | 0.0753 | 0.1338 |
        | 2024-01-15 | 0.0451 | 0.0467 | 0.0501 | 0.0533 | 0.0611 | 0.0751 | 0.1328 |
        """
        if start_date and re.match(r"^\d{4}-\d{2}-\d{2}$", start_date) is None:
            raise ValueError(
                "Please input a valid start date (%Y-%m-%d) like '2010-01-01'"
            )
        if end_date and re.match(r"^\d{4}-\d{2}-\d{2}$", end_date) is None:
            raise ValueError(
                "Please input a valid end date (%Y-%m-%d) like '2020-01-01'"
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
            - Macaulay's Duration: The weighted average time to receive the bond's cash flows.
            - Modified Duration: The Macaulay's duration divided by 1 plus the yield to maturity.
            - Effective Duration: The percentage change in the bond price for a 1% change in the yield to maturity.
            - Dollar Duration: The modified duration multiplied by the bond price.
            - DV01: The dollar value of a 0.01% change in yield to maturity.
            - Convexity: The second derivative of the bond price with respect to the yield to maturity.

        These statistics can be used to evaluate the bond's performance as opposed to other bonds or to estimate the bond's
        sensitivity to changes in interest rates to be able to apply a hedging strategy.

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

        fixedincome = FixedIncome()

        # This is one example and below a collection of different bonds
        # is shown with different characteristics
        fixedincome.collect_bond_statistics(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            yield_to_maturity=0.08,
            frequency=1,
        )
        ```

        Which returns:

        |                     |   Bond 1 |   Bond 2 |   Bond 3 |    Bond 4 |   Bond 5 |   Bond 6 |
        |:--------------------|---------:|---------:|---------:|----------:|---------:|---------:|
        | Par Value           | 100      | 250      |  50      | 1000      |  85      | 320      |
        | Coupon Rate         |   0.05   |   0.02   |   0.075  |    0      |   0.15   |   0.015  |
        | Years to Maturity   |   5      |  10      |   2      |   10      |   3      |   1      |
        | Yield to Maturity   |   0.08   |   0.021  |   0.03   |    0      |   0.16   |   0.04   |
        | Frequency           |   1      |   1      |   4      |    1      |   2      |  12      |
        | Present Value       |  88.0219 | 247.766  |  54.3518 | 1000      |  83.0353 | 312.171  |
        | Current Yield       |   0.0568 |   0.0202 |   0.069  |    0      |   0.1535 |   0.0154 |
        | Effective Yield     |   0.05   |   0.02   |   0.0771 |    0      |   0.1556 |   0.0151 |
        | Macaulay's Duration |   4.5116 |   9.1576 |   1.8849 |   10      |   2.5667 |   0.9932 |
        | Modified Duration   |   4.1774 |   8.9693 |   1.8709 |   10      |   2.3766 |   0.9899 |
        | Effective Duration  |   4.0677 |   8.5181 |   1.8477 |    9.4713 |   2.2952 |   0.9844 |
        | Dollar Duration     |   3.677  |  22.2228 |   1.0168 |  100      |   1.9734 |   3.0902 |
        | DV01                |   0.0004 |   0.0022 |   0      |    0.01   |   0.0001 |   0      |
        | Convexity           |  22.4017 |  93.7509 |   4.0849 |  110      |   7.0923 |   1.0662 |
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

        return pd.Series(bond_statistics).round(self._rounding)

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

        Args:
            par_value (float): The par value (face value) of the bond.
            coupon_rate (float, optional): The coupon rate of the bond. If not provided, a range of coupon rates will be used.
            years_to_maturity (float, optional): The years to maturity of the bond in years. If not provided, a range of years to maturity will be used.
            yield_to_maturity (float, optional): The yield to maturity of the bond. If not provided, a default value of 0.05 will be used.
            frequency (int, optional): The frequency of coupon payments per year. Defaults to 1.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the bond prices for different coupon rates and years to maturity.
        """
        coupon_rate = (
            np.arange(max(0.05 - 0.005 * 20, 0.005), 0.05 + 0.005 * 20, 0.005)
            if coupon_rate is None
            else coupon_rate
        )

        years_to_maturity_dates = (
            [
                pd.to_datetime(self._end_date) + pd.Timedelta(days=365 * interval)
                for interval in range(1, 11)
            ]
            if years_to_maturity is None
            else [years_to_maturity]
        )
        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(coupon_rate, (int, float)):
            coupon_rate = [coupon_rate]
        if isinstance(years_to_maturity, (int, float)):
            years_to_maturity = [years_to_maturity]

        bond_prices: dict[int, dict[float, dict[float, float]]] = {}

        for coupon in coupon_rate:
            bond_prices[coupon] = {}
            for maturity in years_to_maturity:
                (bond_prices[coupon][maturity]) = bond_model.get_bond_price(
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
        - Modified Duration: The Macaulay's duration divided by 1 plus the yield to maturity.
        - Effective Duration: The percentage change in the bond price for a 1% change in the yield to maturity.
        - Dollar Duration: The modified duration multiplied by the bond price.

        These duration measures can be used to estimate the sensitivity of a bond's price to changes in interest rates as well as
        to compare the risk of different bonds. The modified duration is particularly useful for estimating the percentage change
        in the bond price for a 1% change in the yield to maturity. This is also known as the bond's price value of a basis point (PVBP),
        or the bond's dollar duration (DD) or dollar value of a .01% change (DV01).

        Args:
            duration_type (str, optional): The type of duration to calculate. Defaults to 'modified' but can also
                be 'macaulay', 'effective' or 'dollar'.
            par_value (float, optional): The par value (face value) of the bond. Defaults to None.
            coupon_rate (float, optional): The coupon rate of the bond. If not provided, a range of coupon
                rates will be used. Defaults to None.
            years_to_maturity (float, optional): The years to maturity of the bond in years. If not provided, a range of years
                to maturity will be used. Defaults to None.
            yield_to_maturity (float, optional): The yield to maturity of the bond. If not provided, a default
                value of 0.05 will be used. Defaults to None.
            frequency (int, optional): The frequency of coupon payments per year. Defaults to 1.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the bond duration for different coupon rates and years to maturity.
        """
        duration_type_lower = duration_type.lower()

        coupon_rate = (
            np.arange(max(0.05 - 0.005 * 20, 0.005), 0.05 + 0.005 * 20, 0.005)
            if coupon_rate is None
            else coupon_rate
        )

        years_to_maturity_dates = (
            [
                pd.to_datetime(self._end_date) + pd.Timedelta(days=365 * interval)
                for interval in range(1, 11)
            ]
            if years_to_maturity is None
            else [years_to_maturity]
        )
        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(coupon_rate, (int, float)):
            coupon_rate = [coupon_rate]
        if isinstance(years_to_maturity, (int, float)):
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

        The goal is to find the yield to maturity that satisfies the equation above. This is done using the Newton-Raphson method
        which is an iterative method that converges to the root of a function.

        Args:
            par_value (float): The par value (face value) of the bond. This is the original price when it was issued by the issuer.
            coupon_rate (float, optional): The coupon rate of the bond. Defaults to None.
            years_to_maturity (float, optional): The years to maturity of the bond in years. Defaults to None.
            bond_price (float, optional): The price of the bond. Defaults to None.
            frequency (int, optional): The number of coupon payments per year. Defaults to 1.
            guess (float, optional): The initial guess for the yield to maturity. Defaults to 0.05.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations for convergence. Defaults to 100.
            show_input_info (bool, optional): Whether to display input information. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the yield to maturity for different bond prices and years to maturity.
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

        years_to_maturity_dates = (
            [
                pd.to_datetime(self._end_date) + pd.Timedelta(days=365 * interval)
                for interval in range(1, 11)
            ]
            if years_to_maturity is None
            else [years_to_maturity]
        )
        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(bond_price, (int, float)):
            bond_price = [bond_price]
        if isinstance(years_to_maturity, (int, float)):
            years_to_maturity = [years_to_maturity]

        yield_to_maturities: dict[int, dict[float, dict[float, float]]] = {}

        for price in bond_price:
            yield_to_maturities[price] = {}
            for maturity in years_to_maturity:
                (yield_to_maturities[price][maturity]) = (
                    bond_model.get_yield_to_maturity(
                        par_value=par_value,
                        coupon_rate=coupon_rate,
                        years_to_maturity=maturity,
                        bond_price=price,
                        frequency=frequency,
                        guess=guess,
                        tolerance=tolerance,
                        max_iterations=max_iterations,
                    )
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

        return yield_to_maturities_df.round(self._rounding)

    def get_derivative_price(
        self,
        model: str = "black",
        forward_rate: float = 0.05,
        strike_rate: float | list | np.ndarray | None = None,
        volatility: float = 0.01,
        years_to_maturity: float | list | range | None = None,
        risk_free_rate: float | None = None,
        notional: float = 10_000_000,
        is_receiver: bool = True,
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

        Args:
            model (str, optional): The type of model to use for calculating the derivative price. Defaults to "black".
            forward_rate (float, optional): The forward rate as derived from the swap curve. Defaults to None.
            strike_rate (float | list, optional): The strike rate for the derivative. Defaults to None which means it calculates the
                derivative price a range of strike prices. Can also be a list of strike rates (e.g. [0.01, 0.02, 0.03, 0.04, 0.05]).
            volatility (float, optional): The volatility of the underlying asset. Defaults to None.
            years_to_maturity (float | list, optional): The years to maturity of the derivative in years. Defaults to None which means it plots
                the derivative price for the next 10 years. Can also be a list of years to maturity (e.g. [1, 2.3, 2.5, 3])
            risk_free_rate (float, optional): The risk-free interest rate. Defaults to None which means it is equal to the fixed rate.
            notional (float, optional): The notional amount of the derivative. Defaults to 10_000_000.
            is_receiver (bool, optional): True if the holder is the receiver of the derivative, False if the holder is the payer. Defaults to True.
            include_payoff (bool, optional): True to include the payoff in the output, False otherwise. Defaults to False.
            show_input_info (bool, optional): True to display input information, False otherwise. Defaults to True.

        Returns:
            pandas.DataFrame: The Black derivative prices rounded to the specified decimal places.
            pandas.DataFrame (optional): The Black derivative payoffs rounded to the specified decimal places if include_payoff is True.

        For example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome()

        # You can also provide lists of values for the strike rate and years to maturity
        # to define your own strike rates and years to maturity to display in the DataFrame
        fixedincome.get_derivative_price(model_type='black', forward_rate=0.0325)
        ```

        Which returns:

        |   Strike Rate |   2025-04-21 |   2026-04-21 |   2027-04-21 |   2028-04-20 |   2029-04-20 |   2030-04-20 |   2031-04-20 |   2032-04-19 |   2033-04-19 |   2034-04-19 |
        |--------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |         0.005 |          0   |          0   |          0   |         0    |         0    |          0   |          0   |         0    |         0    |         0    |
        |         0.01  |          0   |          0   |          0   |         0    |         0    |          0   |          0   |         0    |         0    |         0    |
        |         0.015 |          0   |          0   |          0   |         0    |         0    |          0   |          0   |         0    |         0    |         0    |
        |         0.02  |          0   |          0   |          0   |         0    |         0    |          0   |          0   |         0    |         0    |         0    |
        |         0.025 |          0   |          0   |          0   |         0    |         0    |          0   |          0   |         0    |         0    |         0    |
        |         0.03  |          0   |          0   |          0   |         0.04 |         0.25 |          0.9 |          2.3 |         4.68 |         8.22 |        12.98 |
        |         0.035 |      24200.6 |      23426.7 |      22677.6 |     21952.5  |     21251.2  |      20573.2 |      19918.5 |     19286.4  |     18676.5  |     18088    |
        |         0.04  |      72601.7 |      70280.1 |      68032.7 |     65857.2  |     63751.2  |      61712.6 |      59739.2 |     57828.9  |     55979.6  |     54189.6  |
        |         0.045 |     121003   |     117133   |     113388   |    109762    |    106252    |     102854   |      99565.3 |     96381.4  |     93299.4  |     90315.9  |
        |         0.05  |     169404   |     163987   |     158743   |    153667    |    148753    |     143996   |     139391   |    134934    |    130619    |    126442    |
        |         0.055 |     217805   |     210840   |     204098   |    197571    |    191254    |     185138   |     179218   |    173487    |    167939    |    162569    |
        |         0.06  |     266206   |     257694   |     249453   |    241476    |    233754    |     226280   |     219044   |    212039    |    205259    |    198695    |
        |         0.065 |     314607   |     304547   |     294808   |    285381    |    276255    |     267421   |     258870   |    250592    |    242578    |    234821    |
        |         0.07  |     363008   |     351400   |     340163   |    329286    |    318756    |     308563   |     298696   |    289144    |    279898    |    270948    |
        |         0.075 |     411410   |     398254   |     385518   |    373191    |    361257    |     349705   |     338522   |    327697    |    317218    |    307074    |
        |         0.08  |     459811   |     445107   |     430874   |    417095    |    403758    |     390846   |     378348   |    366250    |    354538    |    343200    |
        |         0.085 |     508212   |     491960   |     476229   |    461000    |    446258    |     431988   |     418174   |    404802    |    391858    |    379327    |
        |         0.09  |     556613   |     538814   |     521584   |    504905    |    488759    |     473130   |     458000   |    443355    |    429177    |    415453    |
        |         0.095 |     605014   |     585667   |     566939   |    548810    |    531260    |     514272   |     497827   |    481907    |    466497    |    451580    |
        |         0.1   |     653415   |     632521   |     612294   |    592714    |    573761    |     555413   |     537653   |    520460    |    503817    |    487706    |
        |         0.105 |     701816   |     679374   |     657649   |    636619    |    616262    |     596555   |     577479   |    559012    |    541137    |    523832    |
        |         0.11  |     750217   |     726227   |     703004   |    680524    |    658762    |     637697   |     617305   |    597565    |    578456    |    559959    |
        |         0.115 |     798619   |     773081   |     748359   |    724429    |    701263    |     678839   |     657131   |    636118    |    615776    |    596085    |
        |         0.12  |     847020   |     819934   |     793715   |    768334    |    743764    |     719980   |     696957   |    674670    |    653096    |    632211    |
        |         0.125 |     895421   |     866787   |     839070   |    812238    |    786265    |     761122   |     736783   |    713223    |    690416    |    668338    |
        |         0.13  |     943822   |     913641   |     884425   |    856143    |    828766    |     802264   |     776609   |    751775    |    727735    |    704464    |

        """
        model_lower = model.lower()

        strike_rate = (
            np.arange(
                max(
                    (
                        forward_rate - 0.005 * 20
                        if not is_receiver
                        else forward_rate - 0.005 * 5
                    ),
                    0.005,
                ),
                forward_rate + 0.005 * 20 if is_receiver else forward_rate + 0.005 * 5,
                0.005,
            )
            if strike_rate is None
            else strike_rate
        )

        years_to_maturity = (
            range(1, 11) if years_to_maturity is None else years_to_maturity
        )

        if isinstance(years_to_maturity, (int, float)):
            years_to_maturity = [years_to_maturity]
        if isinstance(strike_rate, (int, float)):
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
                        is_receiver=is_receiver,
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
                        is_receiver=is_receiver,
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
                "Holder: %s, Notional: %s, Model: %s Model",
                f"{forward_rate * 100}",
                f"{volatility * 100}",
                f"{risk_free_rate * 100}",
                "Receiver" if is_receiver else "Payer",
                f"{notional:,}",
                model_lower.title(),
            )

        if include_payoff:
            derivative_payoffs_df = pd.DataFrame.from_dict(
                derivative_payoffs, orient="index"
            )
            derivative_payoffs_df.columns = years_to_maturity_dates

            derivative_payoffs_df.index.name = "Strike Rate"

            return derivative_prices_df.round(2), derivative_payoffs_df.round(
                self._rounding
            )

        return derivative_prices_df.round(2)

    def get_government_bond_yield(
        self,
        short_term: bool = False,
        period: str | None = None,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Long-term interest rates refer to government bonds maturing in ten years.
        Rates are mainly determined by the price charged by the lender, the risk
        from the borrower and the fall in the capital value. Long-term interest rates
        are generally averages of daily rates, measured as a percentage. These interest
        rates are implied by the prices at which the government bonds are traded on
        financial markets, not the interest rates at which the loans were issued.

        In all cases, they refer to bonds whose capital repayment is guaranteed by governments.
        Long-term interest rates are one of the determinants of business investment. Low long
        term interest rates encourage investment in new equipment and high interest rates
        discourage it. Investment is, in turn, a major source of economic growth

        See definition: https://data.oecd.org/interest/long-term-interest-rates.htm

        Short-term interest rates are the rates at which short-term borrowings are
        effected between financial institutions or the rate at which short-term government
        paper is issued or traded in the market. Short-term interest rates are generally
        averages of daily rates, measured as a percentage.

        Short-term interest rates are based on three-month money market rates where available.
        Typical standardised names are "money market rate" and "treasury bill rate".

        See definition: https://data.oecd.org/interest/short-term-interest-rates.htm

        Args:
            short_term (bool, optional): Whether to return the short-term interest rate. Defaults to False.
                This means that the long-term interest rate will be returned.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Long Term Interest Rate.

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

        if government_bond_yield.empty:
            raise ValueError("No data available for the selected period ")

        if growth:
            government_bond_yield = calculate_growth(
                government_bond_yield,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_bond_yield = government_bond_yield.loc[
            self._start_date : self._end_date
        ]

        return government_bond_yield.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_ice_bofa_option_adjusted_spread(
        self,
        maturity: bool = True,
        rounding: int | None = None,
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

        Returns:
            pd.DataFrame: A DataFrame containing the Option Adjusted Spread

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
        )

        fixedincome.get_option_adjusted_spread()
        ```

        Which returns:

        | Date       |   1-3 Years |   3-5 Years |   5-7 Years |   7-10 Years |   10-15 Years |   15+ Years |
        |:-----------|------------:|------------:|------------:|-------------:|--------------:|------------:|
        | 2024-01-01 |          77 |          94 |       108.5 |          127 |         131.5 |         118 |
        | 2024-01-02 |          78 |          95 |       109   |          128 |         133   |         119 |
        | 2024-01-03 |          80 |          98 |       113   |          133 |         136   |         122 |
        | 2024-01-04 |          80 |          98 |       112   |          133 |         135   |         122 |
        | 2024-01-05 |          80 |          98 |       112   |          132 |         134   |         121 |
        | 2024-01-08 |          79 |          98 |       112   |          132 |         134   |         120 |
        | 2024-01-09 |          78 |          96 |       110   |          130 |         131   |         117 |
        | 2024-01-10 |          77 |          94 |       108   |          128 |         128   |         113 |
        | 2024-01-11 |          75 |          94 |       107   |          128 |         127   |         113 |
        | 2024-01-12 |          74 |          94 |       107   |          128 |         126   |         112 |
        | 2024-01-15 |          74 |          94 |       107   |          128 |         125   |         111 |
        """
        option_adjusted_spread = (
            fred_model.get_maturity_option_adjusted_spread()
            if maturity
            else fred_model.get_rating_option_adjusted_spread()
        )

        option_adjusted_spread = option_adjusted_spread.loc[
            self._start_date : self._end_date
        ]

        option_adjusted_spread = option_adjusted_spread.round(
            rounding if rounding else self._rounding
        )

        return option_adjusted_spread

    @handle_errors
    def get_ice_bofa_effective_yield(
        self,
        maturity: bool = True,
        rounding: int | None = None,
    ):
        """
        This data represents the effective yield of the ICE BofA Indices, When the last calendar day of the month
        takes place on the weekend, weekend observations will occur as a result of month ending accrued interest adjustments.

        The Effective Yield is the yield of a bond, calculated by dividing the bond's coupon payments by its market price.
        The effective yield is not the same as the stated yield, which is the yield on the bond's coupon payments divided
        by the bond's principal value. The effective yield is a more accurate measure of a bond's return, as it takes into
        account the fact that the investor will not hold the bond to maturity and will likely sell it before it matures.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBBEY
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13YEY

        Args:
            maturity (bool, optional): Whether to return the maturity effective yield or the rating effective yield.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
        )

        fixedincome.get_effective_yield(maturity=False)
        ```

        Which returns:

        | Date       |    AAA |     AA |      A |    BBB |     BB |      B |    CCC |
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2024-01-01 | 0.0456 | 0.047  | 0.0505 | 0.054  | 0.0613 | 0.0752 | 0.1319 |
        | 2024-01-02 | 0.0459 | 0.0473 | 0.0509 | 0.0543 | 0.0622 | 0.0763 | 0.1333 |
        | 2024-01-03 | 0.0459 | 0.0474 | 0.051  | 0.0544 | 0.0634 | 0.0779 | 0.1358 |
        | 2024-01-04 | 0.0466 | 0.0481 | 0.0518 | 0.0551 | 0.0639 | 0.0784 | 0.1367 |
        | 2024-01-05 | 0.047  | 0.0485 | 0.0521 | 0.0554 | 0.0641 | 0.0787 | 0.137  |
        | 2024-01-08 | 0.0465 | 0.0481 | 0.0517 | 0.055  | 0.0633 | 0.0776 | 0.1365 |
        | 2024-01-09 | 0.0464 | 0.048  | 0.0516 | 0.0548 | 0.0629 | 0.0771 | 0.1359 |
        | 2024-01-10 | 0.0464 | 0.048  | 0.0515 | 0.0547 | 0.0622 | 0.0762 | 0.1351 |
        | 2024-01-11 | 0.0456 | 0.0472 | 0.0507 | 0.054  | 0.0619 | 0.076  | 0.1344 |
        | 2024-01-12 | 0.0451 | 0.0467 | 0.0502 | 0.0534 | 0.0613 | 0.0753 | 0.1338 |
        | 2024-01-15 | 0.0451 | 0.0467 | 0.0501 | 0.0533 | 0.0611 | 0.0751 | 0.1328 |
        """
        effective_yield = (
            fred_model.get_maturity_effective_yield()
            if maturity
            else fred_model.get_rating_effective_yield()
        )

        effective_yield = effective_yield.loc[self._start_date : self._end_date]

        effective_yield = effective_yield.round(
            rounding if rounding else self._rounding
        )

        return effective_yield

    @handle_errors
    def get_ice_bofa_total_return(
        self,
        maturity: bool = True,
        rounding: int | None = None,
    ):
        """
        This data represents the total return of the ICE BofA Indices, When the last calendar day of the month
        takes place on the weekend, weekend observations will occur as a result of month ending accrued interest adjustments.

        The total return is the actual rate of return of an investment or a pool of investments over a given evaluation period.
        Total return includes interest, capital gains, dividends and distributions realized over a given period of time.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBBEY
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13YEY

        Args:
            maturity (bool, optional): Whether to return the maturity total return or the rating total return.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
        )

        fixedincome.get_total_return(maturity=True)
        ```

        Which returns:

        | Date       |   1-3 Years |   3-5 Years |   5-7 Years |   7-10 Years |   10-15 Years |   15+ Years |
        |:-----------|------------:|------------:|------------:|-------------:|--------------:|------------:|
        | 2024-01-01 |     1913.78 |     2487.68 |      809.13 |      585.705 |       4206.25 |     4358.69 |
        | 2024-01-02 |     1912.73 |     2484.25 |      807.62 |      584.32  |       4193.7  |     4343.71 |
        | 2024-01-03 |     1912.18 |     2483.95 |      807.54 |      583.84  |       4194.39 |     4339.07 |
        | 2024-01-04 |     1910.86 |     2477.9  |      804.35 |      580.42  |       4163.24 |     4289.24 |
        | 2024-01-05 |     1910.86 |     2475.75 |      802.82 |      578.73  |       4148.31 |     4262.52 |
        | 2024-01-08 |     1912.48 |     2480.39 |      804.97 |      580.71  |       4167.04 |     4302.16 |
        | 2024-01-09 |     1913.5  |     2482.27 |      805.72 |      581.26  |       4173.04 |     4303.34 |
        | 2024-01-10 |     1914.12 |     2483.6  |      806.21 |      581.29  |       4175.16 |     4304.82 |
        | 2024-01-11 |     1918.28 |     2492.25 |      809.94 |      583.92  |       4200.49 |     4330.72 |
        | 2024-01-12 |     1922.1  |     2498.89 |      812.41 |      585.2   |       4213.47 |     4338.43 |
        | 2024-01-15 |     1922.67 |     2499.76 |      812.67 |      585.41  |       4215.34 |     4340.24 |
        """
        total_return = (
            fred_model.get_maturity_total_return()
            if maturity
            else fred_model.get_rating_total_return()
        )

        total_return = total_return.loc[self._start_date : self._end_date]

        total_return = total_return.round(rounding if rounding else self._rounding)

        return total_return

    @handle_errors
    def get_ice_bofa_yield_to_worst(
        self,
        maturity: bool = True,
        rounding: int | None = None,
    ):
        """
        This data represents the semi-annual yield to worst of the ICE BofA Indices, When the last calendar day of the month
        takes place on the weekend, weekend observations will occur as a result of month ending accrued interest adjustments.

        Yield to worst is the lowest potential yield that a bond can generate without the issuer defaulting. The standard US
        convention for this series is to use semi-annual coupon payments, whereas the standard in the foreign markets is
        to use coupon payments with frequencies of annual, semi-annual, quarterly, and monthly.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBBEY
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13YEY

        Args:
            maturity (bool, optional): Whether to return the maturity yield to worst or the rating yield to worst.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
        )

        fixedincome.get_yield_to_worst(maturity=False)
        ```

        Which returns:

        | Date       |    AAA |     AA |      A |    BBB |     BB |      B |    CCC |
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2024-01-01 | 0.0456 | 0.0472 | 0.0503 | 0.0542 | 0.0645 | 0.0786 | 0.1316 |
        | 2024-01-02 | 0.046  | 0.0475 | 0.0506 | 0.0546 | 0.0652 | 0.0796 | 0.1329 |
        | 2024-01-03 | 0.0461 | 0.0475 | 0.0507 | 0.0547 | 0.0662 | 0.081  | 0.1353 |
        | 2024-01-04 | 0.0468 | 0.0483 | 0.0515 | 0.0554 | 0.0665 | 0.0814 | 0.136  |
        | 2024-01-05 | 0.0471 | 0.0486 | 0.0518 | 0.0557 | 0.0667 | 0.0816 | 0.1362 |
        | 2024-01-08 | 0.0466 | 0.0482 | 0.0514 | 0.0553 | 0.066  | 0.0806 | 0.1359 |
        | 2024-01-09 | 0.0465 | 0.0481 | 0.0513 | 0.0551 | 0.0656 | 0.0803 | 0.1353 |
        | 2024-01-10 | 0.0465 | 0.0481 | 0.0512 | 0.0551 | 0.065  | 0.0795 | 0.1345 |
        | 2024-01-11 | 0.0458 | 0.0473 | 0.0504 | 0.0543 | 0.0648 | 0.0793 | 0.134  |
        | 2024-01-12 | 0.0453 | 0.0468 | 0.0499 | 0.0537 | 0.0642 | 0.0786 | 0.1335 |
        | 2024-01-15 | 0.0452 | 0.0468 | 0.0498 | 0.0537 | 0.064  | 0.0784 | 0.1325 |
        """
        yield_to_worst = (
            fred_model.get_maturity_yield_to_worst()
            if maturity
            else fred_model.get_rating_yield_to_worst()
        )

        yield_to_worst = yield_to_worst.loc[self._start_date : self._end_date]

        yield_to_worst = yield_to_worst.round(rounding if rounding else self._rounding)

        return yield_to_worst

    @handle_errors
    def get_euribor_rates(
        self,
        maturities: str | list | None = None,
        nominal: bool = True,
        rounding: int | None = None,
    ):
        """
        Euribor rates, short for Euro Interbank Offered Rate, are the interest rates at which a panel
        of European banks lend funds to one another in the interbank market. These rates are published
        daily by the European Money Markets Institute (EMMI) and serve as a benchmark for various
        financial products and contracts, including mortgages, loans, and derivatives, across the Eurozone.

        The Euribor rates are determined for different maturities, typically ranging from overnight to 12 months
        The most common maturities are 1 month, 3 months, 6 months, and 12 months. Each maturity represents
        the time period for which the funds are borrowed, with longer maturities generally implying higher
        interest rates due to increased uncertainty and risk over longer time horizons.

        For more information, see for example: https://data.ecb.europa.eu/data/datasets/FM/FM.M.U2.EUR.RT.MM.EURIBOR6MD_.HSTA

        Args:
            maturities (str | list | None, optional): Maturities for which to retrieve rates. Defaults to None.
                When set to None, it will retrieve rates for 1 month, 3 months, 6 months, and 12 months.
            nominal (bool, optional): Flag indicating whether to retrieve nominal rates. Defaults to True.
            rounding (int | None, optional): Rounding precision for the rates. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the Euribor rates for the specified maturities.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2023-12-01')

        euribor_rates = fixedincome.get_euribor_rates()
        ```

        Which returns:

        |         |   1-Month |   3-Month |   6-Month |   12-Month |
        |:--------|----------:|----------:|----------:|-----------:|
        | 2023-12 |    0.0386 |    0.0393 |    0.0392 |     0.0367 |
        | 2024-01 |    0.0387 |    0.0393 |    0.0389 |     0.0361 |
        | 2024-02 |    0.0387 |    0.0392 |    0.039  |     0.0367 |
        | 2024-03 |    0.0385 |    0.0392 |    0.0389 |     0.0372 |
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
        euribor_rates = pd.DataFrame(
            columns=[maturity_names[maturity] for maturity in maturities]
        )

        for maturity in maturities:
            if maturity not in ["1M", "3M", "6M", "1Y"]:
                logger.error(
                    "Invalid maturity: %s, please choose from 1M, 3M, 6M, 1Y.", maturity
                )

            maturity_name = maturity_names[maturity]

            if not nominal and maturity == "3M" and len(maturities) > 1:
                logger.warning(
                    "Please note that only the 3-Month Euribor rate has a real rate."
                )

            euribor_rates[maturity_name] = euribor_model.get_euribor_rate(
                maturity=maturity,
                nominal=nominal if not nominal and maturity == "3M" else True,
            )

        euribor_rates = euribor_rates.loc[self._start_date : self._end_date]

        euribor_rates = euribor_rates.round(rounding if rounding else self._rounding)

        return euribor_rates

    def get_european_central_bank_rates(
        self, rate: str | None = None, rounding: int | None = None
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

        Args:
            rate (str, optional): The rate to return. Defaults to None, which returns all rates.
                Choose between 'refinancing', 'lending' or 'deposit'.

        Returns:
            pd.DataFrame: A DataFrame containing the ECB rates.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2023-12-01')

        fixedincome.get_european_central_bank_rates()
        ```

        Which returns:

        |            |   Refinancing |   Lending |   Deposit |
        |:-----------|--------------:|----------:|----------:|
        | 2023-12-01 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-02 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-03 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-04 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-05 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-06 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-07 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-08 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-09 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-10 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-11 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-12 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-13 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-14 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-15 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-16 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-17 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-18 |         0.045 |    0.0475 |      0.04 |
        """
        ecb_rates = pd.DataFrame()

        if rate and rate not in ["refinancing", "lending", "deposit"]:
            raise ValueError(
                "Rate must be one of 'refinancing', 'lending' or 'deposit' or left empty for all."
            )

        if not rate or rate == "refinancing":
            ecb_rates["Refinancing"] = ecb_model.get_main_refinancing_operations()
        if not rate or rate == "lending":
            ecb_rates["Lending"] = ecb_model.get_marginal_lending_facility()
        if not rate or rate == "deposit":
            ecb_rates["Deposit"] = ecb_model.get_deposit_facility()

        ecb_rates = ecb_rates.loc[self._start_date : self._end_date]

        ecb_rates = ecb_rates.round(rounding if rounding else self._rounding)

        return ecb_rates

    def get_federal_reserve_rates(
        self, rate: str = "EFFR", rounding: int | None = None
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
        Treasury’s Office of Financial Research (OFR).

        The SOFR is calculated as a volume-weighted median of transaction-level
        tri-party repo data collected from the Bank of New York Mellon as well as
        GCF Repo transaction data and data on bilateral Treasury repo transactions
        cleared through FICC's DVP service, which are obtained from the U.S.
        Department of the Treasury’s Office of Financial Research (OFR).

        The New York Fed publishes the rates for the prior business day on the New
        York Fed’s website between 8:00 and 9:00 a.m.

        See source: https://www.newyorkfed.org/markets/reference-rates/

        Args:
            rate (str): The rate to return. Defaults to 'EFFR' (Effective Federal Funds Rate).

        Returns:
            pd.DataFrame: A DataFrame containing the Federal Reserve rates including the rate,
                percentiles, volume and upper and lower bounds.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2023-12-01')

        effr = fixedincome.get_federal_reserve_rates()

        effr.loc[:, ['Rate', '1st Percentile', '25th Percentile', '75th Percentile', '99th Percentile']]
        ```

        Which returns:

        | Effective Date   |   Rate |   1st Percentile |   25th Percentile |   75th Percentile |   99th Percentile |
        |:-----------------|-------:|-----------------:|------------------:|------------------:|------------------:|
        | 2023-12-01       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0544 |
        | 2023-12-04       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-05       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-06       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-07       | 0.0533 |            0.053 |            0.0531 |            0.0534 |            0.0545 |
        | 2023-12-08       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-11       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-12       | 0.0533 |            0.053 |            0.0531 |            0.0533 |            0.0544 |
        | 2023-12-13       | 0.0533 |            0.053 |            0.0531 |            0.0533 |            0.0545 |
        | 2023-12-14       | 0.0533 |            0.053 |            0.0531 |            0.0533 |            0.0535 |
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

        fed_data = fed_data.loc[self._start_date : self._end_date]

        fed_data = fed_data.round(rounding if rounding else self._rounding)

        return fed_data
