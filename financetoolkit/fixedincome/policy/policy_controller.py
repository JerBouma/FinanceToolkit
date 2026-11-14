"""Policy Rates Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.economics.helpers import (
    resample_to_period,
    validate_period,
)
from financetoolkit.fixedincome.base.base_controller import FixedIncomeBase
from financetoolkit.fixedincome.sources import ecb_model, euribor_model, fed_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import finalize_dataset

# ruff: noqa: E501

logger = logger_model.get_logger()


@filter_shown_columns
@track_sources
class PolicyFixedIncome(FixedIncomeBase):
    """
    Money market and policy rates: EURIBOR, the European Central Bank rates and the Federal Reserve rates.
    """

    @handle_errors
    def get_euribor_rates(
        self,
        maturities: str | list | None = None,
        nominal: bool = True,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Euribor rates, short for Euro Interbank Offered Rate, are the interest rates at which a panel
        of European banks lend funds to one another in the interbank market. These rates are fixed
        daily by the European Money Markets Institute (EMMI) and serve as a benchmark for various
        financial products and contracts, including mortgages, loans, and derivatives, across the Eurozone.

        Euribor is fixed for maturities of one week to 12 months; this method returns the 1, 3, 6 and 12-month
        rates as monthly averages of the daily fixings, from the European Central Bank, as decimals. Longer
        maturities carry a term premium but also reflect where markets expect short rates to go, so the curve
        inverts when cuts are expected.

        For more information, see for example: https://data.ecb.europa.eu/data/datasets/FM/FM.M.U2.EUR.RT.MM.EURIBOR6MD_.HSTA

        Also known as: euro interbank offered rate, eurozone money market.

        Args:
            maturities (str | list | None, optional): The maturities to retrieve, from "1M", "3M", "6M" and "1Y",
                as a list or a comma-separated string ("1M,3M").
                Defaults to None, which retrieves all four.
            nominal (bool, optional): Whether to retrieve the nominal Euribor fixings or their real
                (inflation-adjusted) counterpart. The ECB only publishes a real Euribor for the 3-month
                maturity, so nominal=False returns that maturity alone and warns about any others that
                were requested rather than silently answering them with a nominal rate. Defaults to True.
            rounding (int | None, optional): Rounding precision for the rates. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            maturities = [
                maturity.strip()
                for maturity in maturities.split(",")
                if maturity.strip()
            ]

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

    @handle_errors
    def get_european_central_bank_rates(
        self,
        rate: str | None = None,
        period: str = "daily",
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
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

        With ample excess liquidity in the banking system the deposit facility rate is the
        rate that steers money markets: the euro short-term rate (€STR) trades just below it.
        The marginal lending rate forms the top of the corridor and the deposit facility rate
        the bottom, with the main refinancing rate in between; the main refinancing rate
        matters more when banks rely on the ECB's weekly operations for funding.

        The rates change on the dates the Governing Council's decisions take effect. Weekly,
        monthly, quarterly and yearly data hold the rate in force on the last day of each
        period, so the current period shows the rate so far.

        See source: https://data.ecb.europa.eu/main-figures/

        Also known as: ECB rates, deposit facility rate.

        Args:
            rate (str, optional): The rate to return. Defaults to None, which returns all rates.
                Choose between 'refinancing', 'lending' or 'deposit'.
            period (str, optional): Whether to return the daily, weekly, monthly, quarterly or
                yearly data. Defaults to 'daily'.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        period = validate_period(
            period,
            ["daily", "weekly", "monthly", "quarterly", "yearly"],
            "ECB rate",
        )

        collected_rates = {}

        if not rate or rate == "refinancing":
            collected_rates["Refinancing"] = ecb_model.get_main_refinancing_operations()
        if not rate or rate == "lending":
            collected_rates["Lending"] = ecb_model.get_marginal_lending_facility()
        if not rate or rate == "deposit":
            collected_rates["Deposit"] = ecb_model.get_deposit_facility()

        # Concatenated in one go so that a rate with a longer history is not truncated to the index of whichever rate happened to be assigned to the frame first.
        ecb_rates = resample_to_period(
            pd.concat(collected_rates, axis=1) if collected_rates else pd.DataFrame(),
            period,
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

    @handle_errors
    def get_federal_reserve_rates(
        self,
        rate: str = "EFFR",
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
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

        SOFR is the US dollar reference rate that replaced dollar LIBOR in loans and derivatives.
        Being secured by Treasuries, it usually trades a few basis points from the effective federal
        funds rate but jumps at month and quarter ends and when reserves get scarce, so the spread
        between its 1st and 99th percentiles, or between SOFR and the EFFR, is watched as a sign of
        strain in the repo market.

        See source: https://www.newyorkfed.org/markets/reference-rates/

        Also known as: Fed rates, federal funds rate, FOMC rate.

        Args:
            rate (str): The rate to return. Defaults to 'EFFR' (Effective Federal Funds Rate).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
