"""Credit Module"""

__docformat__ = "google"

import functools
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit import helpers
from financetoolkit.economics.helpers import (
    average_monthly_index,
    buffered_start_date,
    resample_to_period,
    validate_period,
)
from financetoolkit.economics.sources import (
    ecb_model as economics_ecb_model,
    frb_model,
)
from financetoolkit.fixedincome.credit import cds_model, credit_curve_model
from financetoolkit.fixedincome.helpers import maturity_label, parse_maturities
from financetoolkit.fixedincome.short_rates import short_rate_model
from financetoolkit.fixedincome.sources import (
    bundesbank_model,
    esma_model,
    fred_model,
    hqm_model,
    rba_model,
)
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import check_option, handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding, finalize_dataset

# ruff: noqa: E501

logger = logger_model.get_logger()

# The maturity in years each ICE BofA US Corporate maturity bucket is taken at: its middle,
# and 20 years for the open-ended 15+ years bucket.
ICE_BOFA_BUCKET_MATURITIES = {
    "1-3 Years": 2,
    "3-5 Years": 4,
    "5-7 Years": 6,
    "7-10 Years": 8.5,
    "10-15 Years": 12.5,
    "15+ Years": 20,
}

# The maturity of the par bond a rating index's single spread is read as; its flat hazard
# rate moves by less than 0.2% between 5 and 10 years.
ICE_BOFA_RATING_MATURITY = 7

DEFAULT_SURVIVAL_HORIZONS = (1, 2, 3, 5, 7, 10, 15, 20, 30)
DEFAULT_CDS_TENORS = (1, 3, 5, 7, 10)


@filter_shown_columns
@track_sources
class CreditFixedIncome:
    """
    Credit: ICE BofA corporate bond indices, HQM and Moody's corporate yields, corporate bond yields by country, rating transitions, default rates, implied default probabilities, survival curves, CDS-equivalent spreads, the excess bond premium, corporate borrowing costs and the financial stress index.
    """

    @handle_errors
    def get_ice_bofa_option_adjusted_spread(
        self,
        maturity: bool = True,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the ICE BofA Option-Adjusted Spreads (OASs): the extra yield that US corporate bonds of a given
        maturity or rating category pay over a spot Treasury curve once their embedded options are taken out, the
        compensation investors demand for credit risk.

        The spreads are calculated between a computed OAS index of all bonds in a given maturity and rating category
        and a spot Treasury curve. An OAS index is constructed using each constituent bond's OAS, weighted by market
        capitalization.

        The Option-Adjusted Spread (OAS) is the spread relative to a risk-free interest rate, usually quoted in
        basis points, that equates the theoretical present value of a series of uncertain cash flows to the
        market price of a fixed-income investment. The spread is added to the risk-free rate to compensate for the
        uncertainty of the cash flows. ICE adds it to the semi-annually compounded forward rates of the Treasury
        curve (ICE Bond Index Methodologies), so it is a semi-annually compounded spread.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBB
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13Y

        Args:
            maturity (bool, optional): Whether to return the maturity option adjusted spread or the rating option adjusted spread.
                Defaults to True (by maturity).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Notes:
            ICE licenses only the most recent three years of every ICE BofA series to FRED. A
            start_date earlier than that returns fewer rows rather than an error, and the example
            below falls out of that window over time. Go to the ICE source directly for longer
            histories.

        Returns:
            pd.DataFrame: A DataFrame containing the Option Adjusted Spread as decimal fractions, like
            every other rate. The FRED series are published in percent and are divided by 100 here, so
            a 0.77% (77 basis point) spread is returned as 0.0077.

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
        | 2024-01-01 |    nan      |    nan      |    nan      |     nan      |      nan      |    nan      |
        | 2024-01-02 |      0.0078 |      0.0095 |      0.0109 |       0.0128 |        0.0133 |      0.0119 |
        | 2024-01-03 |      0.008  |      0.0098 |      0.0113 |       0.0133 |        0.0136 |      0.0122 |
        | 2024-01-04 |      0.008  |      0.0098 |      0.0112 |       0.0133 |        0.0135 |      0.0122 |
        | 2024-01-05 |      0.008  |      0.0098 |      0.0112 |       0.0132 |        0.0134 |      0.0121 |
        | 2024-01-08 |      0.0079 |      0.0098 |      0.0112 |       0.0132 |        0.0134 |      0.012  |
        | 2024-01-09 |      0.0078 |      0.0096 |      0.011  |       0.013  |        0.0131 |      0.0117 |
        | 2024-01-10 |      0.0077 |      0.0094 |      0.0108 |       0.0128 |        0.0128 |      0.0113 |
        | 2024-01-11 |      0.0075 |      0.0094 |      0.0107 |       0.0128 |        0.0127 |      0.0113 |
        | 2024-01-12 |      0.0074 |      0.0094 |      0.0107 |       0.0128 |        0.0126 |      0.0112 |
        | 2024-01-15 |      0.0074 |      0.0094 |      0.0107 |       0.0128 |        0.0125 |      0.0111 |
        """
        helpers.require_fred_api_key(self._fred_api_key, "ICE BofA data")

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
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the effective yield of the ICE BofA Indices: the yield that US corporate bonds of a given maturity
        or rating category pay, an index of the cost of corporate borrowing.

        When the last calendar day of the month takes place on the weekend, weekend observations will occur as a
        result of month ending accrued interest adjustments.

        The effective yield of an ICE BofA index is the yield of the index as a whole, aggregated from the
        yields of its constituent bonds and weighted by their market capitalisation. It is an index-level yield of the corporate bond market segment,
        not a statistic derived from any single bond's coupon; for the yield of a single bond, see the
        "Yield to Maturity" of `collect_bond_statistics` instead. ICE states each bond's yield in conventional
        terms, compounded at the bond's coupon frequency (ICE Bond Index Methodologies): twice a year for the US
        indices, so their effective yields compound semi-annually like Treasury yields.

        The FRED series are published in percent and are converted to decimals here, so a 5.40% BBB
        index yield is returned as 0.054.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBBEY
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13YEY

        Also known as: ICE BofA corporate bond yield, credit yield.

        Args:
            maturity (bool, optional): Whether to return the maturity effective yield or the rating effective yield.
                Defaults to True (by maturity).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Notes:
            ICE licenses only the most recent three years of every ICE BofA series to FRED. A
            start_date earlier than that returns fewer rows rather than an error, and the example
            below falls out of that window over time. Go to the ICE source directly for longer
            histories.

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
        helpers.require_fred_api_key(self._fred_api_key, "ICE BofA data")

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
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the total return of the ICE BofA Indices: the gain of holding US corporate bonds of a given
        maturity or rating category, coupons and price changes included.

        When the last calendar day of the month takes place on the weekend, weekend observations will occur as a
        result of month ending accrued interest adjustments.

        The total return is the actual rate of return of an investment or a pool of investments over a given evaluation period.
        Total return includes interest, capital gains, dividends and distributions realized over a given period of time.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLCC0A4BBBTRIV
        - Maturity: https://fred.stlouisfed.org/series/BAMLCC1A013YTRIV

        Args:
            maturity (bool, optional): Whether to return the maturity total return or the rating total return.
                Defaults to True (by maturity).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Notes:
            ICE licenses only the most recent three years of every ICE BofA series to FRED. A
            start_date earlier than that returns fewer rows rather than an error, and the example
            below falls out of that window over time. Go to the ICE source directly for longer
            histories.

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
        helpers.require_fred_api_key(self._fred_api_key, "ICE BofA data")

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
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the semi-annual yield to worst of the ICE BofA Indices: the lowest yield US corporate bonds of a
        given maturity or rating category pay if their issuers redeem them at the least favourable moment for
        investors.

        When the last calendar day of the month takes place on the weekend, weekend observations will occur as a
        result of month ending accrued interest adjustments.

        Yield to worst is the lowest potential yield that a bond can generate without the issuer defaulting. The standard US
        convention for this series is to use semi-annual coupon payments, whereas the standard in the foreign markets is
        to use coupon payments with frequencies of annual, semi-annual, quarterly, and monthly. These series state
        every index in semi-annual terms, so the yields are compounded twice a year whatever the coupon frequency.

        See definitions:

        - Ratings: https://fred.stlouisfed.org/series/BAMLC0A4CBBBSYTW
        - Maturity: https://fred.stlouisfed.org/series/BAMLC1A0C13YSYTW

        Args:
            maturity (bool, optional): Whether to return the maturity yield to worst or the rating yield to worst.
                Defaults to True (by maturity).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Notes:
            ICE licenses only the most recent three years of every ICE BofA series to FRED. A
            start_date earlier than that returns fewer rows rather than an error, and the example
            below falls out of that window over time. Go to the ICE source directly for longer
            histories.

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
        helpers.require_fred_api_key(self._fred_api_key, "ICE BofA data")

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
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the High Quality Market (HQM) corporate bond yield curve of the U.S.
        Department of the Treasury: the yields of high quality (AAA, AA and A rated) US
        corporate bonds by maturity, from 6 months to 100 years, monthly from 1984. The
        Treasury builds it for discounting long-dated liabilities, which is why it is the
        prescribed discount curve for US corporate pension obligations, and its long history
        makes it a natural basis for modelling how corporate yields and spreads move.

        Two curves are available. Spot rates are zero-coupon yields, the rate to discount a
        single payment at that maturity, for 17 maturities. Par yields are the coupons a bond
        priced at par would pay, for 2, 5, 10 and 30 years, comparable to the Treasury par yield
        curve (see `get_treasury_rates`). Every value is the average over the month.

        Without a FRED API key the curve comes from the Treasury's own
        HQM files, which gives the same figures. When a FRED API key is set (see the
        `fred_api_key` parameter of the `FixedIncome` class), it comes from FRED, which
        republishes the Treasury's curve. The rates are returned as decimal fractions (0.0558
        for 5.58%), semi-annual bond-equivalent as the Treasury publishes them.

        See definition: https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curves

        Also known as: HQM curve, corporate bond yield curve, pension discount curve, AA
        corporate curve.

        Args:
            rate (str, optional): The curve. Options are: "spot" (the spot rates), "par" (the par
                yields). Defaults to "spot".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        check_option("rate", rate, ("spot", "par"))

        # FRED republishes the Treasury's curve, so it is used when a key is set and the
        # Treasury's files otherwise.
        if self._fred_api_key:
            hqm_curve = fred_model.get_hqm_corporate_bond_yield_curve(
                rate, self._start_date, self._end_date, self._fred_api_key
            )
        elif rate == "spot":
            hqm_curve = hqm_model.get_hqm_spot_rates(self._start_date, self._end_date)
        else:
            hqm_curve = hqm_model.get_hqm_par_yields()

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
        show_columns: str | list[str] | None = None,
    ):
        """
        Computes the credit spread of high quality US corporate bonds per maturity: the
        Treasury's High Quality Market (HQM) corporate par yield minus the Treasury constant
        maturity yield, at 2, 5, 10 and 30 years, monthly. Both legs are par
        yields averaged over the month, so they are directly comparable. The spread is what
        investors demand for the default and liquidity risk of investment grade companies,
        and its shape across maturities, usually wider for longer maturities, shows how that
        compensation grows with time.

        Unlike the ICE BofA option-adjusted spreads (see `get_ice_bofa_option_adjusted_spread`),
        of which FRED only carries the last three years, this spread covers decades, long
        enough to calibrate how credit spreads behave through several business cycles.

        The formula is as follows:

        - HQM Spread (maturity) = HQM Corporate Par Yield (maturity) - Treasury Constant Maturity Yield (maturity),
          both monthly averages, for 2, 5, 10 and 30 years

        Without a FRED API key both legs come from the Treasury, the HQM
        par yields from its HQM files and the Treasury yields as the monthly averages of its
        daily par yield curve, rounded to two decimals in percent as the Federal Reserve's H.15
        release publishes them. That curve starts in 1990 and has no 30-year yield from
        February 2002 to February 2006, so the spread without a FRED API key starts in 1990
        and leaves those 30-year months empty. When a FRED API key is set (see the `fred_api_key` parameter of
        the `FixedIncome` class), the Treasury yields are FRED's constant maturity series, from
        1984. The spreads are returned as decimal fractions (0.0079 for 0.79 percentage points).

        Also known as: corporate credit spread curve, credit spread term structure,
        investment grade spread.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        # FRED carries both legs, so it is used when a key is set and the Treasury otherwise.
        hqm_spread = (
            fred_model.get_hqm_corporate_bond_spread(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else hqm_model.get_hqm_corporate_bond_spread(
                self._start_date, self._end_date
            )
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the yields of corporate bonds of a variety of countries, by rating where
        the source splits them, monthly and with long histories: the input for a credit
        spread factor outside the United States, where corporate bond indices by rating
        (iBoxx, ICE BofA) are licensed data.

        - Germany: the average yield on debt securities outstanding of non-MFI corporations
          the Bundesbank publishes monthly from 1957 (monthly averages), all ratings
          together ("All ratings").
        - Australia: the yields of non-financial corporate bonds rated A and BBB with a
          target tenor of 3, 5, 7 and 10 years the Reserve Bank of Australia publishes from
          2005 (end of month), e.g. "BBB 10Y".
        - United States: Moody's seasoned Aaa and Baa corporate bond yields, from 1919. This
          requires a FRED API key (the fred_api_key of the FixedIncome class); see
          get_moodys_corporate_bond_yields for daily data.

        As spreads, each yield is returned over government bonds: for
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        sources = {
            "Germany": functools.partial(
                bundesbank_model.get_all_ratings_corporate_bond_yield,
                start_date,
                self._end_date,
                spread,
            ),
            "Australia": (
                rba_model.get_corporate_bond_spreads
                if spread
                else rba_model.get_corporate_bond_yields
            ),
        }

        if self._fred_api_key:
            sources["United States"] = functools.partial(
                fred_model.get_moodys_aaa_baa_corporate_bonds,
                "monthly",
                start_date,
                self._end_date,
                self._fred_api_key,
                spread,
            )

        requested = (
            list(sources)
            if countries is None
            else [countries] if isinstance(countries, str) else list(countries)
        )

        if "United States" in requested and not self._fred_api_key:
            helpers.require_fred_api_key(self._fred_api_key, "ICE BofA data")

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
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the rating transition matrix of a credit rating agency: for every rating
        at the start of a period, the share of ratings that ended the period in each rating
        category, in default or withdrawn, which shows how likely a rating is to be
        upgraded, downgraded or to default.

        It is the input of rating migration and default models, such as the
        Jarrow-Lando-Turnbull model, and of credit portfolio models.

        The data comes from CEREP, the central repository where every credit rating agency
        registered in the European Union reports its ratings to ESMA, published twice a
        year with periods from 1989. It covers the global ratings of agencies such as S&P,
        Moody's, Fitch, DBRS and Scope, for corporates (non-financial, financial and
        insurance), sovereigns, structured finance and covered bonds, in each agency's own
        rating scale.

        By default the matrix is withdrawal-adjusted, the convention of the agencies' own
        default studies: ratings withdrawn during the period are left out, so each row sums
        to 1. The matrix compares the rating at the start of the period with the rating at its
        end, so a default that was followed by a new rating within the period (as after a
        distressed exchange) shows as that new rating; get_default_rates counts every default
        within the period and is the measure to calibrate default rates on. The withdrawals can
        be kept as a separate "Withdrawals" column, and the counts can be returned instead of
        the probabilities.

        The formula of the probabilities is as follows:

        - Transition Probability (from, to) = Count (from, to) / SUM(Count (from, every end category)), with
          the end categories the ratings and default, plus "Withdrawals" only when they are kept

        A single year's matrix rests on few observations in the top and bottom rows: an AAA
        row with few issuers is often 100% stable one year and shows a downgrade the next.
        Credit models therefore use the average over many years, or a longer horizon, and
        check how far the latest year deviates from it, which also shows where in the credit
        cycle the year stands.

        See definition: https://registers.esma.europa.eu/cerep-publication/

        Also known as: rating migration matrix, credit migration matrix, transition
        probabilities.

        Args:
            agency (str, optional): The rating agency, e.g. "S&P", "Moody's", "Fitch",
                "DBRS", "Scope" or "Kroll", or the CEREP code of any registered agency.
                Defaults to "S&P".
            rating_type (str, optional): The rated instruments. Options are: "corporate" (companies),
                "sovereign" (governments), "structured_finance" (structured finance products),
                "covered_bonds" (covered bonds). Defaults to "corporate".
            year (int | None, optional): The first year of the period. Defaults to None,
                which takes the latest year published.
            horizon (int, optional): The length of the period in years, from 1 January of
                year. Defaults to 1.
            probabilities (bool, optional): Whether to return the share of ratings per row
                instead of the number. Defaults to True.
            include_withdrawals (bool, optional): Whether to keep the ratings withdrawn
                during the period as a "Withdrawals" column. Defaults to False.
            region (str | None, optional): The area of the rated entities. Options are: "Africa",
                "America", "Asia", "Europe", "EU members", "International", "Oceania", None (all of
                them). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        check_option("rating_type", rating_type, esma_model.RATING_TYPES)
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the one-year default rates per rating of a credit rating agency, year by
        year from 1989: the share of the ratings outstanding at the start of each year that
        defaulted within it.

        The history of default rates per rating calibrates the default intensity of credit
        risk models and shows how defaults cluster in recessions.

        The data comes from CEREP, the central repository where every credit rating agency
        registered in the European Union reports its ratings to ESMA, in each agency's own
        rating scale (AAA to C for S&P and Fitch, Aaa to C for Moody's). Only the years
        between the start and end date are retrieved, one request per year. CEREP answers
        slowly (up to half a minute per year), so a long history takes a few minutes the
        first time; each year is cached afterwards. The years before an
        agency reported to CEREP read as zero defaults and are left out, so the history
        starts at the first year with a default.

        Default rates rise steeply down the rating scale, from close to zero for investment
        grade to tens of percent a year for CCC, and because they cluster in recessions a
        probability of default calibrated on calm years understates the risk. Practitioners
        average over a full cycle, often decades, for through-the-cycle estimates and use the
        worst years for stress scenarios. Rates for the top ratings rest on very few defaults.

        See definition: https://registers.esma.europa.eu/cerep-publication/

        Also known as: default frequency, annual default rate by rating, historical default
        rates.

        Args:
            agency (str, optional): The rating agency, e.g. "S&P", "Moody's", "Fitch",
                "DBRS", "Scope" or "Kroll", or the CEREP code of any registered agency.
                Defaults to "S&P".
            rating_type (str, optional): The rated instruments. Options are: "corporate" (companies),
                "sovereign" (governments), "structured_finance" (structured finance products),
                "covered_bonds" (covered bonds). Defaults to "corporate".
            region (str | None, optional): The area of the rated entities. Options are: "Africa",
                "America", "Asia", "Europe", "EU members", "International", "Oceania", None (all of
                them). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        check_option("rating_type", rating_type, esma_model.RATING_TYPES)

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
            lambda year: (
                year,
                esma_model.get_default_rates(
                    agency_code,
                    rating_type,
                    pd.Timestamp(f"{year}-01-01"),
                    pd.Timestamp(f"{year}-12-31"),
                    region,
                ),
            ),
            [(year,) for year in years],
            max_workers=6,
        )
        default_rates = {
            year: rates["Default Rate"] for year, rates in results if not rates.empty
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
    def get_implied_default_probability(
        self,
        ratings: list[str] | str | None = None,
        recovery_rate: float = 0.4,
        horizons: list[float] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the default probabilities that credit spreads imply: how likely the
        market prices a default of an issuer of each rating within each horizon.

        Investors in a corporate bond are compensated for the loss they expect from a
        default, so with a constant default intensity (hazard rate) λ and a share R
        recovered on default the spread s is about λ (1 - R), the "credit triangle":

            λ = s / (1 - R),  P(default within t years) = 1 - e^(-λt)

        The spreads are the latest ICE BofA option-adjusted spreads of the US corporate bond
        indices per rating (AAA to CCC) up to the end date, which need a FRED API key. For
        a spread of your own, use `short_rate_model.get_implied_default_probability`. Spreads
        also pay for liquidity and for the uncertainty of the losses, so these risk-neutral
        probabilities are higher than the default rates that are observed historically (see
        `get_default_rates`); the gap is the credit risk premium.

        For more information, see the following book:

        - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th
          edition, chapter 24.

        Also known as: implied default probability, hazard rate, credit triangle,
        risk-neutral default probability.

        Args:
            ratings (list[str] | str | None, optional): The ratings: "AAA", "AA", "A", "BBB",
                "BB", "B" or "CCC", as a list or a comma-separated string. Defaults to None,
                every rating.
            recovery_rate (float, optional): The share of the face value recovered on
                default, the scenario the probabilities assume. Defaults to 0.4, about the
                average recovery of senior unsecured bonds.
            horizons (list[float] | str | None, optional): The horizons in years, as a list or
                a comma-separated string. Defaults to None, which uses 1, 3, 5 and 10 years.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Spread, the Hazard Rate and the default probability per horizon
            as decimals, with a row per rating.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2026-11-05", fred_api_key="your_fred_api_key")

        fixedincome.get_implied_default_probability()
        ```

        Which returns:

        | Rating   |   Spread |   Hazard Rate |   1Y Default Probability |   3Y Default Probability |   5Y Default Probability |   10Y Default Probability |
        |:---------|---------:|--------------:|-------------------------:|-------------------------:|-------------------------:|--------------------------:|
        | AAA      |   0.0048 |        0.008  |                   0.008  |                   0.0237 |                   0.0392 |                    0.0769 |
        | AA       |   0.0069 |        0.0115 |                   0.0114 |                   0.0339 |                   0.0559 |                    0.1086 |
        | A        |   0.0077 |        0.0128 |                   0.0128 |                   0.0378 |                   0.0622 |                    0.1204 |
        | BBB      |   0.0116 |        0.0193 |                   0.0191 |                   0.0564 |                   0.0921 |                    0.1758 |
        | BB       |   0.0213 |        0.0355 |                   0.0349 |                   0.101  |                   0.1626 |                    0.2988 |
        | B        |   0.0367 |        0.0612 |                   0.0593 |                   0.1676 |                   0.2635 |                    0.4576 |
        | CCC      |   0.132  |        0.22   |                   0.1975 |                   0.4831 |                   0.6671 |                    0.8892 |
        """
        horizons = [
            float(horizon)
            for horizon in helpers.split_values(horizons) or [1, 3, 5, 10]
        ]

        helpers.require_fred_api_key(self._fred_api_key, "ICE BofA data")
        start_date = (
            datetime.strptime(self._end_date, "%Y-%m-%d") - timedelta(days=31)
        ).strftime("%Y-%m-%d")
        option_adjusted_spread = fred_model.get_rating_option_adjusted_spread(
            start_date, self._end_date, self._fred_api_key
        ).dropna(how="all")
        if option_adjusted_spread.empty:
            raise ValueError(
                f"FRED has no ICE BofA option-adjusted spreads in the month before {self._end_date}."
            )
        # The FRED spreads arrive as decimals (0.0077 for 77 basis points), the unit the model takes.
        spread_series = option_adjusted_spread.iloc[-1].dropna()
        if ratings is not None:
            requested = [rating.upper() for rating in helpers.split_values(ratings)]
            if unknown := [
                rating for rating in requested if rating not in spread_series.index
            ]:
                raise ValueError(
                    f"No spread is available for the rating {', '.join(unknown)}. Choose from "
                    f"{', '.join(spread_series.index)}."
                )
            spread_series = spread_series[requested]
        spread_series.index.name = "Rating"

        result = pd.DataFrame(
            {
                name: {
                    "Spread": spread,
                    **short_rate_model.get_implied_default_probability(
                        spread, recovery_rate, tuple(horizons)
                    ),
                }
                for name, spread in spread_series.items()
            }
        ).T
        result.index.name = "Rating"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def _credit_curves(
        self,
        maturity: bool,
        ratings: list[str] | str | None,
        recovery_rate: float,
    ) -> tuple[pd.Series, dict[str, pd.Series]]:
        """
        The Treasury zero-coupon curve and the hazard rate curves the latest ICE BofA
        option-adjusted spreads up to the end date imply, see get_survival_curve.

        Args:
            maturity (bool): Whether to bootstrap one curve from the maturity buckets of the
                US Corporate index, or a flat curve per rating.
            ratings (list[str] | str | None): The ratings, when maturity is False.
            recovery_rate (float): The share of the face value recovered on default.

        Returns:
            tuple[pd.Series, dict[str, pd.Series]]: The annually compounded zero-coupon
            curve and the piecewise flat hazard rates per curve, indexed by maturity.
        """
        helpers.require_fred_api_key(self._fred_api_key, "ICE BofA data")
        start_date = (
            datetime.strptime(self._end_date, "%Y-%m-%d") - timedelta(days=31)
        ).strftime("%Y-%m-%d")
        option_adjusted_spread = (
            fred_model.get_maturity_option_adjusted_spread
            if maturity and ratings is None
            else fred_model.get_rating_option_adjusted_spread
        )(start_date, self._end_date, self._fred_api_key).dropna(how="all")
        if option_adjusted_spread.empty:
            raise ValueError(
                f"FRED has no ICE BofA option-adjusted spreads in the month before {self._end_date}."
            )
        spreads = option_adjusted_spread.iloc[-1].dropna()
        # The Treasury curve of the day the spreads are of (or the last day before it), so
        # neither looks ahead of the other.
        spread_date = pd.Timestamp(str(option_adjusted_spread.index[-1])).strftime(
            "%Y-%m-%d"
        )
        zero_curve = self._latest_zero_curve("United States", spread_date)

        if maturity and ratings is None:
            maturities = [
                ICE_BOFA_BUCKET_MATURITIES[bucket] for bucket in spreads.index
            ]
            return zero_curve, {
                "Corporate": credit_curve_model.get_bond_implied_hazard_rates(
                    maturities, spreads.to_numpy(), zero_curve, recovery_rate
                )
            }

        if ratings is not None:
            requested = [rating.upper() for rating in helpers.split_values(ratings)]
            if unknown := [
                rating for rating in requested if rating not in spreads.index
            ]:
                raise ValueError(
                    f"No spread is available for the rating {', '.join(unknown)}. Choose from "
                    f"{', '.join(spreads.index)}."
                )
            spreads = spreads[requested]

        return zero_curve, {
            rating: credit_curve_model.get_bond_implied_hazard_rates(
                [ICE_BOFA_RATING_MATURITY], [spread], zero_curve, recovery_rate
            )
            for rating, spread in spreads.items()
        }

    @handle_errors
    def get_survival_curve(
        self,
        maturity: bool = True,
        ratings: list[str] | str | None = None,
        recovery_rate: float = 0.4,
        horizons: list[float] | str | None = None,
        hazard_rates: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Bootstraps survival curves, the probability that a US corporate issuer has not
        defaulted after each horizon, from the ICE BofA option-adjusted spreads. The default
        time follows a hazard rate that is piecewise flat: constant between the maturities of
        the spreads it is bootstrapped from and held at its last value beyond them (Jarrow and
        Turnbull, 1995; O'Kane, 2008, ch. 7).

        Each spread is read as a bond that trades at par: its coupon is the par yield of the
        Treasury zero-coupon curve plus the spread. The hazard rate up to its maturity is
        solved, in maturity order with the earlier ones fixed, so that the bond priced on the
        Treasury curve with that default risk is worth par again, every coupon and the
        principal paid only on survival and the recovery rate times the face value paid at
        default:

        The formula is as follows:

        - D_s(t) = P(t) * (1 + s_i / 2)^(-2t), c_i = 2 * (1 - D_s(T_i)) / SUM(D_s(t_k))
        - Solve λ_i: c_i / 2 * SUM(P(t_k) Q(t_k)) + P(T_i) Q(T_i) + R ∫_0^T_i P(t) (-dQ(t)) = 1
        - Q(t) = exp(-SUM(λ_j * (min(t, T_j) - T_(j-1))^+))

        where s_i is the option-adjusted spread of maturity T_i, compounded twice a year as ICE
        adds it to the semi-annual Treasury rates (an option-adjusted spread is a Z-spread
        once the call options are taken out), P(t) = (1 + r(t))^(-t) the discount factor of
        the annually compounded US Treasury zero-coupon curve (see `get_zero_coupon_curve`),
        t_k the semi-annual coupon dates and R the recovery rate. The default integral is
        exact on the grid of the curve, hazard rate and coupon dates (ISDA CDS Standard Model).
        The credit triangle λ ≈ s / (1 - R) of `get_implied_default_probability` is its
        first-order version.

        By maturity, the curve is bootstrapped from the six maturity buckets of the ICE BofA
        US Corporate Index (investment grade), each taken at its middle: 2, 4, 6, 8.5 and 12.5
        years, and 20 years for 15+ years. By rating, each rating index (AAA to CCC and lower)
        has a single spread and so a flat hazard rate, bootstrapped from a par bond of 7 years;
        the hazard rate moves by less than 0.2% between 5 and 10 years.

        The spreads are the latest up to the end date, and the Treasury curve that of the same
        day (or the last day before it). Spreads also pay for liquidity and the uncertainty of
        losses, so these risk-neutral survival probabilities are below those of the default
        rates observed (see `get_default_rates`).

        For more information, see the following sources:

        - Jarrow, R.A. and Turnbull, S.M. (1995). "Pricing Derivatives on Financial Securities
          Subject to Credit Risk." The Journal of Finance, 50(1), 53-85.
        - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
          ch. 3 and 7.
        - Markit (2009). "ISDA CDS Standard Model."

        Referenced against:
        - QuantLib, HazardRateCurve: https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_hazard_rate_curve.html
        - QuantLib, RiskyBondEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_risky_bond_engine.html

        Shares similar logic:
        - QuantLib, PiecewiseFlatHazardRate: https://www.quantlib.org/reference/class_quant_lib_1_1_piecewise_default_curve.html

        Also known as: credit curve, hazard rate curve, default probability term structure,
        bond-implied default probabilities.

        Args:
            maturity (bool, optional): Whether to bootstrap one curve from the maturity buckets
                of the US Corporate Index, or a flat curve per rating. Defaults to True (by
                maturity).
            ratings (list[str] | str | None, optional): The ratings: "AAA", "AA", "A", "BBB",
                "BB", "B" or "CCC", as a list or a comma-separated string; giving them returns
                the curves by rating. Defaults to None, every rating when maturity is False.
            recovery_rate (float, optional): The share of the face value recovered on
                default. Defaults to 0.4, about the average recovery of senior unsecured bonds
                and the ISDA standard.
            horizons (list[float] | str | None, optional): The horizons in years or as labels
                such as "6M", as a list or a comma-separated string. Defaults to None, which
                uses 1, 2, 3, 5, 7, 10, 15, 20 and 30 years.
            hazard_rates (bool, optional): Whether to return the hazard rate in force at each
                horizon (continuous, per year) instead of the survival probability. Defaults
                to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the FixedIncome rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The survival probabilities (or hazard rates) as decimals, with a row
            per horizon and a column per curve: Corporate by maturity, or the ratings.

        Notes:
            ICE licenses only the most recent three years of every ICE BofA series to FRED,
            so the end date must lie within them. The hazard rate between two maturities is a
            forward rate of default: where the spreads flatten between two buckets, it falls
            below the hazard rate of the bucket before.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2026-11-05", fred_api_key="your_fred_api_key")

        fixedincome.get_survival_curve(maturity=False)
        ```

        Which returns:

        | Maturity   |    AAA |     AA |      A |    BBB |     BB |      B |    CCC |
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
        | 1Y         | 0.9921 | 0.9887 | 0.9873 | 0.981  | 0.9656 | 0.9418 | 0.8124 |
        | 2Y         | 0.9842 | 0.9774 | 0.9749 | 0.9624 | 0.9323 | 0.887  | 0.66   |
        | 3Y         | 0.9764 | 0.9663 | 0.9625 | 0.9442 | 0.9003 | 0.8353 | 0.5361 |
        | 5Y         | 0.961  | 0.9445 | 0.9383 | 0.9087 | 0.8394 | 0.7409 | 0.3538 |
        | 7Y         | 0.9459 | 0.9232 | 0.9147 | 0.8745 | 0.7826 | 0.6572 | 0.2335 |
        | 10Y        | 0.9236 | 0.8921 | 0.8805 | 0.8257 | 0.7045 | 0.549  | 0.1252 |
        | 15Y        | 0.8876 | 0.8427 | 0.8261 | 0.7503 | 0.5913 | 0.4067 | 0.0443 |
        | 20Y        | 0.8531 | 0.7959 | 0.7752 | 0.6818 | 0.4963 | 0.3014 | 0.0157 |
        | 30Y        | 0.7879 | 0.7101 | 0.6825 | 0.563  | 0.3497 | 0.1654 | 0.002  |
        """
        survival_horizons = parse_maturities(horizons, DEFAULT_SURVIVAL_HORIZONS)
        _, curves = self._credit_curves(maturity, ratings, recovery_rate)
        calculate = (
            credit_curve_model.get_hazard_rate
            if hazard_rates
            else credit_curve_model.get_survival_probability
        )

        result = pd.DataFrame(
            {
                name: calculate(hazard, hazard.index.to_numpy(), survival_horizons)
                for name, hazard in curves.items()
            },
            index=[maturity_label(horizon) for horizon in survival_horizons],
        )
        result.index.name = "Maturity"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_credit_default_swap_spread(
        self,
        maturity: bool = True,
        ratings: list[str] | str | None = None,
        recovery_rate: float = 0.4,
        tenors: list[float] | str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculates the CDS-equivalent spreads of US corporate credit: the spread a
        protection buyer would pay each year to insure against a default of an issuer with
        that credit, by maturity or by rating.

        These are the par spreads of credit default swaps on the survival curves the ICE
        BofA option-adjusted spreads imply (see `get_survival_curve`), which turns the
        spreads of corporate bonds into CDS spreads.

        The swaps follow the ISDA CDS Standard Model: the premium is paid quarterly on an
        ACT/360 basis, the premium accrued since the last payment is paid on default, and the
        protection pays (1 - R) of the notional at the time of default. The par spread is the
        premium at which both legs are worth the same:

        The formula is as follows:

        - RPV01 = SUM(τ_k P(t_k) Q(t_k)) + 365 / 360 * SUM(∫_t_(k-1)^t_k (t - t_(k-1)) P(t) (-dQ(t)))
        - Protection Leg = (1 - R) * ∫_0^T P(t) (-dQ(t))
        - CDS Spread = Protection Leg / RPV01

        where τ_k is the ACT/360 accrual of the k-th quarter, P(t) = (1 + r(t))^(-t) the
        discount factor of the annually compounded US Treasury zero-coupon curve, Q(t) the
        survival probability of the piecewise flat hazard rate curve and R the recovery rate.
        The integrals are exact on the grid of the curve, hazard rate and premium dates, as in
        the standard model. The spreads are discounted on the Treasury curve, the curve the
        option-adjusted spreads are measured over, rather than the swap curve dealers use, and
        the protection starts today without the standard model's step-in day.

        The gap between a CDS-equivalent spread and the bond spread at the same maturity is a
        model basis: a bond loses its price less the recovery at default while a CDS pays the
        face value less the recovery, and the CDS premium stops at default. Traded CDS spreads
        also differ by the liquidity of each market, the cheapest-to-deliver option and the
        funding of the bonds.

        For more information, see the following sources:

        - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
          ch. 5 to 7.
        - Markit (2009). "ISDA CDS Standard Model."
        - Hull, J.C. and White, A. (2000). "Valuing Credit Default Swaps I: No Counterparty
          Default Risk." The Journal of Derivatives, 8(1), 29-40.

        For a single name with spreads or hazard rates of your own, use
        `cds_model.get_credit_default_swap` and `cds_model.get_cds_implied_hazard_rates`.

        Also known as: CDS spread, CDS-equivalent spread, credit default swap premium,
        bond-implied CDS spread.

        Referenced against:
        - QuantLib, CreditDefaultSwap: https://www.quantlib.org/reference/class_quant_lib_1_1_credit_default_swap.html
        - QuantLib, IsdaCdsEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_isda_cds_engine.html

        Shares similar logic:
        - FinancePy, CDS: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/credit/cds.py#L262

        Args:
            maturity (bool, optional): Whether to use the survival curve of the maturity
                buckets of the US Corporate Index, or one per rating. Defaults to True (by
                maturity).
            ratings (list[str] | str | None, optional): The ratings: "AAA", "AA", "A", "BBB",
                "BB", "B" or "CCC", as a list or a comma-separated string; giving them returns
                the spreads by rating. Defaults to None, every rating when maturity is False.
            recovery_rate (float, optional): The share of the face value recovered on
                default, for the bonds and the swaps alike. Defaults to 0.4, the ISDA standard
                for senior unsecured debt.
            tenors (list[float] | str | None, optional): The tenors of the swaps in years or as
                labels such as "6M", as a list or a comma-separated string. Defaults to None,
                which uses 1, 3, 5, 7 and 10 years.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the FixedIncome rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The par CDS spreads as decimals (0.01 for 100 basis points), with a
            row per tenor and a column per curve: Corporate by maturity, or the ratings.

        Notes:
            ICE licenses only the most recent three years of every ICE BofA series to FRED,
            so the end date must lie within them.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(end_date="2026-11-05", fred_api_key="your_fred_api_key")

        fixedincome.get_credit_default_swap_spread()
        ```

        Which returns:

        | Maturity   |   Corporate |
        |:-----------|------------:|
        | 1Y         |      0.0061 |
        | 3Y         |      0.0075 |
        | 5Y         |      0.0091 |
        | 7Y         |      0.0104 |
        | 10Y        |      0.011  |
        """
        cds_tenors = parse_maturities(tenors, DEFAULT_CDS_TENORS)
        zero_curve, curves = self._credit_curves(maturity, ratings, recovery_rate)

        result = pd.DataFrame(
            {
                name: [
                    cds_model.get_credit_default_swap(
                        tenor,
                        zero_curve,
                        hazard,
                        hazard.index.to_numpy(),
                        recovery_rate=recovery_rate,
                    )["Par Spread"]
                    for tenor in cds_tenors
                ]
                for name, hazard in curves.items()
            },
            index=[maturity_label(tenor) for tenor in cds_tenors],
        )
        result.index.name = "Maturity"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves Moody's seasoned corporate bond yields for Aaa (the highest rating) and
        Baa (the lowest investment grade rating) US corporate bonds: what the safest and the
        riskiest investment grade companies pay to borrow, the longest running corporate
        bond yield benchmark there is, daily from 1986 and monthly from 1919.

        The spreads over the 10-year Treasury yield can be returned instead,
        together with the Baa minus Aaa spread, the classic measure of default risk in the
        academic literature (e.g. Fama and French, 1989), which widens sharply in recessions.
        Monthly spreads start in 1953, when the monthly 10-year Treasury yield does.

        Monthly values are the averages over the month, as Moody's publishes them; weekly
        values are the yields on the last day of each week (weeks end on Friday).

        The formula of the spreads is as follows:

        - Aaa = Aaa Yield - 10-Year Treasury Yield, and Baa likewise
        - Baa - Aaa = Baa Yield - Aaa Yield

        Monthly spreads subtract the monthly average 10-year Treasury constant maturity yield; daily
        spreads are the ones FRED publishes against the daily 10-year yield.

        The data comes from FRED, so a free FRED API key is required. The yields and spreads
        are returned as decimal fractions (0.0675 for 6.75%).

        Also known as: Moody's Aaa, Moody's Baa, corporate bond yields by rating, default
        spread, credit spread.

        Args:
            period (str, optional): The frequency of the data. Options are: "daily" (daily),
                "weekly" (the last day of every week), "monthly" (monthly averages), "quarterly"
                and "yearly" (the averages of the monthly averages of the complete quarters and
                years). Defaults to "daily".
            spread (bool, optional): Whether to return the spreads over the 10-year Treasury
                yield instead of the yields. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The "Aaa" and "Baa" yields, or as spreads the "Aaa", "Baa" and
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
        helpers.require_fred_api_key(self._fred_api_key, "ICE BofA data")
        period = validate_period(
            period,
            ["daily", "weekly", "monthly", "quarterly", "yearly"],
            "Moody's corporate bond yield",
        )
        # Quarters and years average the monthly averages, which reach back to 1919.
        frequency = "daily" if period in ("daily", "weekly") else "monthly"

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
        elif period in ("quarterly", "yearly"):
            moodys = average_monthly_index(moodys, period)

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
        show_columns: str | list[str] | None = None,
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

        The spread and premium are returned as decimal fractions
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
    def get_corporate_borrowing_cost(
        self,
        countries: list[str] | str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the composite cost of borrowing for non-financial corporations the European
        Central Bank publishes monthly from 2003, for the euro area and every member: the
        average interest rate on new bank loans to companies across maturities and loan sizes.
        Euro area companies borrow mostly from banks rather than in the bond market, which
        makes this the broadest measure of their cost of credit, and the closest freely
        available proxy for euro area corporate credit conditions, since euro area corporate
        bond indices are licensed data.

        The rate is a decimal fraction (0.0377 for 3.77%).

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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
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

        Weekly and monthly periods take the index on the last day of
        each period.

        See definition: https://data.ecb.europa.eu/data/datasets/CISS

        Also known as: CISS, systemic stress, financial stress, financial conditions.

        Args:
            countries (list[str] | str | None, optional): The areas to include, e.g. 'Euro Area'
                or ['United States', 'United Kingdom']. Defaults to None, which returns all of them.
            period (str, optional): The frequency of the data. Options are: "daily", "weekly",
                "monthly". Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        | 2026-09 |      0.0133 |          0.0091 |           0.0122 |  0.0015 |
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
