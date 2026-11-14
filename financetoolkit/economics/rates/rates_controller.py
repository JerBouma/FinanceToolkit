"""Interest Rates and Money Module"""

__docformat__ = "google"

import itertools

import pandas as pd

from financetoolkit import helpers
from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    OVERNIGHT_RATE_DAY_COUNTS,
    buffered_start_date,
    check_period_type,
    compound_overnight_rate,
    extend_with_recent,
    resample_to_period,
    to_gmdb_country_names,
    validate_period,
)
from financetoolkit.economics.sources import (
    bis_model,
    boe_model,
    ecb_model,
    fred_model,
    freddie_mac_model,
    gmdb_model,
    mof_model,
    oecd_model,
    treasury_model,
)
from financetoolkit.fixedincome.sources import (
    bundesbank_model,
    ecb_model as ecb_rates_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import (
    collect_sources,
    track_sources,
)
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    euro_area_hint,
    finalize_dataset,
)

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class RatesEconomics(EconomicsBase):
    """
    Interest rates and money: policy, overnight, short and long-term and real interest rates, the yield curve slope, the 30-year mortgage rate, real yields, breakeven and expected inflation, the yield decomposition, the money supply and exchange rates.
    """

    def _get_daily_central_bank_policy_rate(
        self, start_date: str, countries: list[str] | str | None = None
    ) -> pd.DataFrame:
        """
        Retrieves the daily policy rates of the Bank for International Settlements (BIS),
        extended with the days the Bank of England and the European Central Bank have
        published since. The BIS updates its series about once a week, so a decision of
        the last few days would otherwise be missing.

        Args:
            start_date (str): The start date (YYYY-MM-DD).
            countries (list[str] | str | None, optional): The requested countries, so the
                central banks of other countries are not retrieved. Defaults to None.

        Returns:
            pd.DataFrame: One column per country, indexed by day.
        """
        policy_rate = bis_model.get_central_bank_policy_rate(start_date, self._end_date)

        requested = (
            {country.strip() for country in countries.split(",")}
            if isinstance(countries, str)
            else set(countries or policy_rate.columns)
        )
        # Since 18 September 2024 the BIS reports the ECB's deposit facility rate for the
        # euro area (the main refinancing rate before), the rate the ECB steers with.
        central_banks = {
            "United Kingdom": lambda start: boe_model.get_bank_rate(
                start, self._end_date
            ),
            "Euro Area": lambda _: ecb_rates_model.get_deposit_facility()
            .rename("Euro Area")
            .to_frame(),
        }

        for country, fetch in central_banks.items():
            last_day = (
                policy_rate[country].last_valid_index()
                if country in policy_rate and country in requested
                else None
            )
            if last_day is None:
                continue

            # A month of overlap, on which both publishers have to agree.
            with collect_sources() as recent_sources:
                recent = fetch((last_day - 31).strftime("%Y-%m-%d"))
            recent = recent[
                (recent.index <= pd.Period(self._end_date, freq="D"))
                # Business days, like the BIS series (the ECB lists every calendar day).
                & ~recent.index.dayofweek.isin([5, 6])
            ]
            policy_rate = extend_with_recent(
                policy_rate, recent, country, recent_sources
            )

        return policy_rate

    def _get_daily_long_term_interest_rate(
        self, countries: list[str] | str | None = None
    ) -> pd.DataFrame:
        """
        Combines the daily 10-year government bond yields published by the U.S.
        Department of the Treasury, the Bank of England, the Japanese Ministry of
        Finance and the European Central Bank (the euro area yield curve).

        Returns:
            pd.DataFrame: One column per country, indexed by day.
        """
        start_date = buffered_start_date(self._start_date, "daily")
        end_date = self._end_date

        return self._combine_sources_for(
            [
                (
                    lambda: treasury_model.get_long_term_interest_rate(
                        start_date, end_date
                    ),
                    "United States",
                ),
                (
                    lambda: boe_model.get_long_term_interest_rate(start_date, end_date),
                    "United Kingdom",
                ),
                # The Ministry of Finance publishes one file with the full history.
                (mof_model.get_long_term_interest_rate, "Japan"),
                (
                    lambda: ecb_model.get_long_term_interest_rate(start_date, end_date),
                    "Euro Area",
                ),
            ],
            countries,
        )

    @handle_errors
    def get_exchange_rates(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        end_of_period: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Exchange rates are defined as the price of one country's currency in relation
        to another country's currency. This indicator is measured in terms of
        national currency per US dollar.

        See definition: https://data.oecd.org/conversion/exchange-rates.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Both sources are quoted the same way (national currency per US dollar), but the OECD
        source omits the United States itself, since the rate is trivially 1, whereas the GMDB
        source carries it as 1.0. Only the OECD source supports monthly and quarterly
        frequency; the GMDB is annual only, so the period argument has no effect when
        gmdb_source is True.

        Both are averages over each period. For daily and weekly rates (period="daily" or
        "weekly"), or the rate at the end of each month, quarter or year (end_of_period=True),
        the rates come from the Bank for International Settlements instead, without an API
        key: daily for around 80 currencies with history back to 1945 for some, and at the
        end of each month for around 130, back to the 1950s for most. Weekly and quarterly or
        yearly end-of-period rates take the last observation of each period.

        Also known as: currency exchange, FX rates, foreign exchange rates.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the daily, weekly, monthly, quarterly
                or the annual data. Defaults to None, which is quarterly when the Economics class
                was initialized with quarterly=True and otherwise the highest frequency available: daily.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            end_of_period (bool, optional): Whether to return the rate at the end of each month,
                quarter or year instead of its average. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Exchange Rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2000-01-01', end_date='2010-12-31')

        economics.get_exchange_rates(countries=['Japan', 'Indonesia', "China"], period='yearly')
        ```

        Which returns:

        |      |    Japan |   Indonesia |   China |
        |:-----|---------:|------------:|--------:|
        | 2000 | 107.746  |     8401.53 |  8.2771 |
        | 2001 | 121.409  |    10252.9  |  8.277  |
        | 2002 | 125.138  |     9315.13 |  8.277  |
        | 2003 | 115.929  |     8570.07 |  8.277  |
        | 2004 | 108.158  |     8945.34 |  8.2777 |
        | 2005 | 110.181  |     9714.85 |  8.1924 |
        | 2006 | 116.307  |     9170.49 |  7.9737 |
        | 2007 | 117.774  |     9138.88 |  7.6063 |
        | 2008 | 103.444  |     9682.61 |  6.9485 |
        | 2009 |  93.5426 |    10397.6  |  6.8311 |
        | 2010 |  87.7075 |     9082.76 |  6.7678 |
        """
        check_period_type(period)

        period = self._default_period(period, "daily")
        # The GMDB is annual, so a monthly or quarterly request goes to the sources that
        # have that frequency unless the GMDB is asked for explicitly.
        gmdb_source = (
            gmdb_source
            if gmdb_source is not None
            else self._gmdb_source and period.lower() not in ("monthly", "quarterly")
        )
        period = "yearly" if period.lower() == "annual" else period.lower()

        if period in ("daily", "weekly"):
            exchange_rates = resample_to_period(
                bis_model.get_exchange_rates(
                    "daily",
                    buffered_start_date(self._start_date, period),
                    self._end_date,
                ),
                period,
            )
        elif end_of_period:
            monthly_rates = bis_model.get_exchange_rates(
                "monthly",
                buffered_start_date(self._start_date, "monthly"),
                self._end_date,
            )
            exchange_rates = (
                monthly_rates
                if period == "monthly"
                else monthly_rates.groupby(
                    monthly_rates.index.asfreq("Q" if period == "quarterly" else "Y")
                ).last()
            )
        elif gmdb_source:

            exchange_rates = gmdb_model.get_usd_exchange_rate(
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            )
        else:
            exchange_rates = oecd_model.get_exchange_rates(
                period=period, start_date=self._start_date, end_date=self._end_date
            )

        return finalize_dataset(
            dataset=exchange_rates,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Exchange Rates",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # The BIS lists weekends and holidays without a rate.
            dropna=period in ("daily", "weekly") or end_of_period,
        )

    @handle_errors
    def get_real_effective_exchange_rate(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Real Effective Exchange Rate (REER) for a variety of countries over time from the
        Global Macro Database (GMDB). The REER is a trade-weighted average of a country's currency
        relative to a basket of other major currencies, adjusted for relative price levels
        (inflation) between the country and its trading partners.

        Unlike a simple bilateral exchange rate, the REER captures a currency's overall
        competitiveness: a rising REER indicates that a country's exports are becoming more
        expensive (and imports cheaper) relative to its trading partners after accounting for
        inflation differentials, while a falling REER indicates the opposite. The index is
        annual with 2015 = 100 for every country.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: REER, trade-weighted exchange rate, currency competitiveness index.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Real Effective Exchange Rate

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date="2025-12-31")

        economics.get_real_effective_exchange_rate(countries=['United States', 'Japan', 'Netherlands'])
        ```

        Which returns:

        |      |   United States |    Japan |   Netherlands |
        |:-----|----------------:|---------:|--------------:|
        | 2018 |         103.02  | 107.295  |       104.015 |
        | 2019 |         105.92  | 110.309  |       103.969 |
        | 2020 |         107.337 | 111.285  |       106.116 |
        | 2021 |         105.222 | 101.827  |       106.389 |
        | 2022 |         115.234 |  87.8727 |       106.548 |
        | 2023 |         116.036 |  83.6944 |       107.665 |
        | 2024 |         118.848 |  79.2776 |       109.002 |
        | 2025 |         119.249 |  80.8126 |       111.404 |
        """

        real_effective_exchange_rate = gmdb_model.get_real_effective_exchange_rate(
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
        )

        return finalize_dataset(
            dataset=real_effective_exchange_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Real Effective Exchange Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_money_supply(
        self,
        countries: list[str] | str | None = None,
        measure: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Money Supply is the total amount of money that is in circulation in a country.
        It includes currency, demand deposits, and other liquid assets that can be easily
        converted into cash. Money supply is an important economic indicator that the
        Federal Reserve uses to implement its monetary policy.

        Money supply can be divided into five categories: M0, M1, M2, M3 and M4.
            - M0: The total of all physical currency, plus accounts at the central bank that can be exchanged for physical currency.
            - M1: The total of all physical currency part of bank reserves + the amount in demand accounts ("checking" or "current" accounts).
            - M2: M1 + most savings accounts, money market accounts, retail money market mutual funds, and small denomination time deposits.
            - M3: M2 + large time deposits, institutional money market funds, short-term repurchase agreements, and other larger liquid assets.
            - M4: M3 + all other financial assets.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The aggregates are annual and expressed in millions of national currency, so levels
        are not comparable across countries with different currencies, but growth rates are.
        Not every country publishes every aggregate; the ones it does not are NaN.

        Also known as: M1, M2, M3, monetary aggregate.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            measure (str | None, optional): Which single aggregate to return, one of 'M0', 'M1',
                'M2', 'M3' or 'M4'. Defaults to None, which returns all five with the aggregate
                as the first level of the column index.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Money Supply

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date='2020-12-31')

        money_supply = economics.get_money_supply(
            countries=['Netherlands', 'Germany', 'United States'],
            measure='M2'
        )
        ```

        Which returns:

        |      |   Netherlands |    Germany |   United States |
        |:-----|--------------:|-----------:|----------------:|
        | 2010 |        701718 | 1.9878e+06 |     8.4731e+06  |
        | 2011 |        727265 | 2.1053e+06 |     8.8445e+06  |
        | 2012 |        746482 | 2.2556e+06 |     9.7533e+06  |
        | 2013 |        741372 | 2.3144e+06 |     1.05023e+07 |
        | 2014 |        743043 | 2.4272e+06 |     1.11275e+07 |
        | 2015 |        822382 | 2.6518e+06 |     1.18062e+07 |
        | 2016 |        841302 | 2.8022e+06 |     1.25222e+07 |
        | 2017 |        851237 | 2.9236e+06 |     1.33189e+07 |
        | 2018 |        846513 | 3.0562e+06 |     1.39024e+07 |
        | 2019 |        889033 | 3.1968e+06 |     1.44602e+07 |
        | 2020 |        974276 | 3.4582e+06 |     1.5425e+07  |
        """

        money_supply = gmdb_model.get_money_supply(
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
        )

        money_supply = finalize_dataset(
            dataset=money_supply,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            rolling=rolling,
            trailing=trailing,
        )

        if measure:
            if measure not in money_supply.columns.get_level_values(0):
                logger.warning(
                    f"The following measure is not available for Money Supply: {measure}"
                )
            else:
                money_supply = money_supply[measure]

        if countries:
            if isinstance(countries, str):
                countries = [countries]

            full_money_supply = money_supply

            if isinstance(money_supply.columns, pd.MultiIndex):
                available_countries = money_supply.columns.get_level_values(1).unique()
                missing_countries = [
                    country
                    for country in countries
                    if country not in available_countries
                ]
                money_supply = money_supply.loc[
                    :, money_supply.columns.get_level_values(1).isin(countries)
                ]
            else:
                missing_countries = [
                    country
                    for country in countries
                    if country not in money_supply.columns
                ]
                # A set would make the column order depend on hash randomisation.
                money_supply = money_supply[
                    [
                        country
                        for country in countries
                        if country not in missing_countries
                    ]
                ]

            if missing_countries:
                logger.warning(
                    f"The following countries are not available for Money Supply: {missing_countries}"
                )

            # Euro members have no money supply of their own since 1999, only the area does.
            if not money_supply.empty and bool(money_supply.isna().all().all()):
                found = [
                    country for country in countries if country not in missing_countries
                ]
                logger.warning(
                    "There are no Money Supply observations for %s between %s and %s.%s",
                    ", ".join(found),
                    self._start_date,
                    self._end_date,
                    euro_area_hint(found, full_money_supply),
                )

        return money_supply

    @handle_errors
    def get_central_bank_policy_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        The Central Bank Policy Rate is the interest rate that a central bank sets on its
        loans and advances to a commercial bank. This interest rate is used by the monetary
        authorities to control inflation and stabilize the country's currency.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The rate is annual and expressed as a decimal fraction per annum (0.0538 for 5.375%),
        taken at the end of the year rather than averaged over it.

        With period="daily", "weekly" or "monthly" the rates come from the Bank for
        International Settlements (BIS) instead. It publishes the
        policy rates of around fifty central banks daily, including the Federal Reserve,
        the European Central Bank (as "Euro Area"), the Bank of England and the Bank of
        Japan, some back to the 1940s. Weekly and monthly take the rate on the last day of
        each period, so the current week or month shows the rate so far.

        The BIS updates its series about once a week. The days since are added from the
        Bank of England (Bank Rate) and the European Central Bank, which publish their
        rates the same day; other countries end at the latest BIS update. For the euro
        area the BIS reports the main refinancing rate up to 17 September 2024 and the
        deposit facility rate from 18 September 2024, when the ECB began steering money
        markets with it; get_european_central_bank_rates has either rate throughout.

        The result is a decimal fraction (0.05375 for 5.375%), like every other rate in the
        Finance Toolkit.

        Also known as: policy rate, benchmark rate, base rate.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the daily, weekly, monthly or the annual
                data. Defaults to None, which is the daily data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Central Bank Policy Rate

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-01-01', end_date='2025-12-31')

        economics.get_central_bank_policy_rate(countries=['Netherlands', 'Germany', 'United States'], period='yearly')
        ```

        Which returns:

        |      |   Netherlands |   Germany |   United States |
        |:-----|--------------:|----------:|----------------:|
        | 2021 |       -0.005  |   -0.005  |          0.0012 |
        | 2022 |        0.0044 |    0.0044 |          0.0438 |
        | 2023 |        0.0362 |    0.0362 |          0.0538 |
        | 2024 |        0.0356 |    0.0356 |          0.0438 |
        | 2025 |        0.0213 |    0.0213 |          0.0362 |
        """
        check_period_type(period)

        period = validate_period(
            self._default_period(period, "daily", quarterly_available=False),
            ["daily", "weekly", "monthly", "yearly"],
            "central bank policy rate",
        )

        if period == "yearly":

            central_bank_policy_rate = gmdb_model.get_central_bank_policy_rate(
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            )
        else:
            # Weeks and months are taken from the daily rates, which the BIS updates
            # sooner than its own end-of-month series.
            central_bank_policy_rate = resample_to_period(
                self._get_daily_central_bank_policy_rate(
                    buffered_start_date(self._start_date, period), countries
                ),
                period,
            )

        return finalize_dataset(
            dataset=central_bank_policy_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Central Bank Policy Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # Rows empty for every selected country, e.g. a month only another source has.
            dropna=period != "yearly",
        )

    @handle_errors
    def get_short_term_interest_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Short-term interest rates are the rates at which short-term borrowings are
        effected between financial institutions or the rate at which short-term government
        paper is issued or traded in the market. Short-term interest rates are generally
        averages of daily rates, measured as a percentage.

        Short-term interest rates are based on three-month money market rates where available.
        Typical standardised names are "money market rate" and "treasury bill rate". The OECD
        source specifically returns the 3-month interbank offered rate.

        See definition: https://data.oecd.org/interest/short-term-interest-rates.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Both sources return the rate as a decimal fraction per annum (0.0513 for 5.13%), so
        the two are directly interchangeable. Only the OECD source supports monthly and
        quarterly frequency; the GMDB is annual only, so the period argument has no effect
        when gmdb_source is True.

        Also known as: 3-month rate, money market rate, short-term yield.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Defaults to None, which is quarterly when the Economics class was initialized with
                quarterly=True and monthly otherwise.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Short Term Interest Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-05-01', end_date="2025-12-31")

        economics.get_short_term_interest_rate(
            countries=['Japan', 'United States', 'China'],
            gmdb_source=False,
            period='quarterly'
        )
        ```

        Which returns:

        |        |   Japan |   United States |   China |
        |:-------|--------:|----------------:|--------:|
        | 2023Q2 | -0.0001 |          0.0513 |  0.0289 |
        | 2023Q3 |  0.0001 |          0.0543 |  0.0261 |
        | 2023Q4 |  0.0002 |          0.054  |  0.0288 |
        | 2024Q1 |  0.0005 |          0.0526 |  0.0267 |
        | 2024Q2 |  0.0013 |          0.0531 |  0.0235 |
        | 2024Q3 |  0.0023 |          0.051  |  0.021  |
        | 2024Q4 |  0.0033 |          0.0454 |  0.0205 |
        | 2025Q1 |  0.0079 |          0.0432 |  0.0202 |
        | 2025Q2 |  0.0078 |          0.0431 |  0.0188 |
        | 2025Q3 |  0.0079 |          0.042  |  0.0171 |
        | 2025Q4 |  0.009  |          0.0386 |  0.0168 |
        """
        check_period_type(period)

        period = self._default_period(period, "monthly")

        # The GMDB is annual, so a monthly or quarterly request goes to the sources that
        # have that frequency unless the GMDB is asked for explicitly.
        gmdb_source = (
            gmdb_source
            if gmdb_source is not None
            else self._gmdb_source and period.lower() not in ("monthly", "quarterly")
        )

        if gmdb_source:

            short_term_interest_rate = gmdb_model.get_short_term_interest_rate(
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            )
        else:
            short_term_interest_rate = oecd_model.get_short_term_interest_rate(
                period=period,
                start_date=self._start_date,
                end_date=self._end_date,
            )

        return finalize_dataset(
            dataset=short_term_interest_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Short Term Interest Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_long_term_interest_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
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

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Both sources return the rate as a decimal fraction per annum (0.0357 for 3.57%), so
        the two are directly interchangeable. Only the OECD source supports monthly and
        quarterly frequency; the GMDB is annual only, so the period argument has no effect
        when gmdb_source is True.

        With period="daily" or "weekly" the 10-year yields come from the issuers
        themselves: the U.S. Department of the Treasury (par yield,
        from 1990), the Bank of England (nominal par yield of gilts, from 1993), the
        Japanese Ministry of Finance (JGBs, from 1974) and the European Central Bank (the
        euro area yield curve of all central government bonds, from 2004). The euro area
        rate is the 10-year zero-coupon rate of the ECB's Svensson curve, converted from
        continuous to annual compounding, while the other three are par yields. Weekly takes
        the yield on the last trading day of each week. Other countries have no daily source
        and are not included.

        A few countries report a national long-term rate to the OECD that is not a 10-year
        government bond yield, such as Brazil, whose series is the TJLP (Taxa de Juros de Longo
        Prazo), the long-term lending rate of the national development bank BNDES.

        Also known as: 10-year yield, government bond rate, long-term yield.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the daily, weekly, monthly, quarterly
                or the annual data. Defaults to None, which is quarterly when the Economics class
                was initialized with quarterly=True and otherwise the highest frequency available: daily
                for the United States, the United Kingdom, Japan and the Euro Area, monthly when
                other countries are requested.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
                Not used for daily and weekly data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Long Term Interest Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-05-01', end_date='2023-12-31')

        economics.get_long_term_interest_rate(
            countries=['Japan', 'United States', 'Germany'],
            gmdb_source=False,
            period='monthly'
        )
        ```

        Which returns:

        |         |   Japan |   United States |   Germany |
        |:--------|--------:|----------------:|----------:|
        | 2023-05 |  0.0043 |          0.0357 |    0.0234 |
        | 2023-06 |  0.004  |          0.0375 |    0.0238 |
        | 2023-07 |  0.0059 |          0.039  |    0.0246 |
        | 2023-08 |  0.0064 |          0.0417 |    0.0255 |
        | 2023-09 |  0.0076 |          0.0438 |    0.0266 |
        | 2023-10 |  0.0095 |          0.048  |    0.0282 |
        | 2023-11 |  0.0066 |          0.045  |    0.026  |
        | 2023-12 |  0.0062 |          0.0402 |    0.021  |
        """
        check_period_type(period)

        # Daily 10-year yields exist for the United States, the United Kingdom, Japan and
        # the euro area; any other country is published monthly at most.
        requested = (
            [country.strip() for country in countries.split(",")]
            if isinstance(countries, str)
            else list(countries or [])
        )
        period = self._default_period(
            period,
            (
                "daily"
                if requested
                and set(requested)
                <= {"United States", "United Kingdom", "Japan", "Euro Area"}
                else "monthly"
            ),
        )

        # The GMDB is annual, so a monthly or quarterly request goes to the sources that
        # have that frequency unless the GMDB is asked for explicitly.
        gmdb_source = (
            gmdb_source
            if gmdb_source is not None
            else self._gmdb_source and period.lower() not in ("monthly", "quarterly")
        )

        if period.lower() in ("daily", "weekly"):
            long_term_interest_rate = resample_to_period(
                self._get_daily_long_term_interest_rate(countries), period.lower()
            )
        elif gmdb_source:

            long_term_interest_rate = gmdb_model.get_long_term_interest_rate(
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            )
        else:
            long_term_interest_rate = oecd_model.get_long_term_interest_rate(
                period=period,
                start_date=self._start_date,
                end_date=self._end_date,
            )

        return finalize_dataset(
            dataset=long_term_interest_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Long Term Interest Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # Rows empty for every selected country, e.g. a month only another source has.
            dropna=period.lower() in ("daily", "weekly"),
        )

    @handle_errors
    def get_overnight_rate(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        cumulative: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        The overnight rate is the interest rate at which banks borrow from each other for
        a single day. It is the rate central banks steer with their policy rate, and the
        benchmark that replaced LIBOR for loans, derivatives and floating rate notes in
        each of these currencies, which makes it the most direct daily read of monetary
        conditions.

        Each currency has its own benchmark, published by its central bank:

        - Euro Area: the euro short-term rate (€STR) from the European Central Bank, since
          October 2019.
        - United Kingdom: the Sterling Overnight Index Average (SONIA) from the Bank of
          England, since 1997.
        - Japan: the uncollateralized overnight call rate, the Tokyo Overnight Average
          Rate (TONA), from the Bank of Japan, since 1985.
        - United States: the Secured Overnight Financing Rate (SOFR) from the Federal
          Reserve Bank of New York, since April 2018.

        The rates are returned as decimal fractions per annum (0.0244 for 2.44%). Weekly
        and monthly periods take the rate
        on the last day of each period (weeks end on Friday).

        With cumulative=True, the rates are compounded day by day into the growth of 1
        deposited on the first day of the date range, the method of the compounded €STR
        index of the European Central Bank and the SOFR Index of the New York Fed:

        - Growth (t) = Growth (t-1) × (1 + Overnight Rate (t-1) × Days (t-1) / Day Count)

        Days (t-1) is the number of calendar days the previous rate applies to (three over
        a weekend) and the Day Count is that of each money market: 360 for €STR and SOFR
        (ACT/360) and 365 for SONIA and TONA (ACT/365). The value on a day includes the
        interest up to that day, so the first day is 1, and weekly and monthly periods
        take the value on the last day of each period. It is the cash return of money
        deposited at the overnight rate, before costs and taxes.

        Also known as: €STR, ESTR, SONIA, TONA, SOFR, overnight interbank rate, risk-free
        rate.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str, optional): Whether to return the daily, weekly or monthly data. Defaults to "daily".
            cumulative (bool, optional): Whether to return the growth of 1 deposited at the
                overnight rate on the first day of the date range instead of the rate.
                Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the overnight rate per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        economics.get_overnight_rate(period='monthly')
        ```

        Which returns:

        |         |   Euro Area |   Japan |   United Kingdom |   United States |
        |:--------|------------:|--------:|-----------------:|----------------:|
        | 2026-04 |      0.0193 |  0.0073 |           0.0373 |          0.0366 |
        | 2026-05 |      0.0193 |  0.0073 |           0.0373 |          0.0363 |
        | 2026-06 |      0.0218 |  0.0098 |           0.0373 |          0.0368 |
        | 2026-07 |      0.0218 |  0.0098 |           0.0373 |          0.0366 |
        | 2026-08 |      0.0218 |  0.0098 |           0.0373 |          0.0368 |
        | 2026-09 |      0.0244 |  0.0123 |           0.0373 |          0.039  |
        """
        check_period_type(period)

        period = validate_period(
            period, ["daily", "weekly", "monthly"], "overnight rate"
        )

        overnight_rate = self._get_overnight_rate(countries, period)

        if cumulative:
            # Compounded only after slicing to the date range, so the growth starts at 1
            # on the first day of the range.
            overnight_rate = compound_overnight_rate(
                overnight_rate.loc[self._start_date : self._end_date],
                OVERNIGHT_RATE_DAY_COUNTS,
            )

        return finalize_dataset(
            dataset=resample_to_period(overnight_rate, period),
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Overnight Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_real_interest_rate(
        self,
        countries: list[str] | str | None = None,
        rate_type: str = "long_term",
        period: str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Real Interest Rate for a variety of countries over time. The Real Interest Rate
        is the nominal interest rate adjusted for inflation, and reflects the true cost of
        borrowing (or the true return earned on savings) once the erosion of purchasing power
        by inflation is taken into account.

        The formula is as follows (the Fisher equation, approximation):

        - Real Interest Rate = Nominal Interest Rate - Inflation Rate, with the nominal rate the Long Term Interest
          Rate by default (rate_type="long_term") and the inflation rate the year-over-year change in consumer prices

        The nominal interest rate is the Long Term Interest Rate (the 10-year government bond
        yield), the Short Term Interest Rate (the 3-month money market rate) or the Central
        Bank Policy Rate, selected via the rate_type parameter.

        Monthly (the default) subtracts the monthly year-over-year inflation rate of
        get_inflation_rate (national statistical offices, Eurostat for the euro area and the
        BIS) from the OECD's monthly long or short-term rate, the monthly average of the daily
        rates, or from the policy rate at the end of the month (see
        get_central_bank_policy_rate). The OECD has no euro area rates, so the euro area is
        available with rate_type="policy": the ECB's deposit facility rate minus euro area
        inflation. A month the inflation rate is not published yet is empty.

        Yearly subtracts the annual inflation rate of the Global Macro Database (GMDB) from
        its annual long or short-term rate (or the OECD's with gmdb_source=False) or its
        end-of-year policy rate.

        Both legs are decimal fractions (0.05 for 5%), so the result is itself a decimal
        fraction (-0.0016 for a real rate of -0.16%).

        A negative real interest rate means that, after inflation, savers are effectively losing
        purchasing power and borrowers are being subsidized in real terms; this occurred in many
        countries during the 2021-2022 inflation surge.

        This is the ex post real rate: the nominal rate minus the inflation that was realised
        in the same year, not the inflation investors expected when the rate was set. It shows
        how savers and borrowers actually fared, but for valuation or the stance of monetary
        policy practitioners use the ex ante real rate, such as the yield on inflation-linked
        bonds (get_real_yield_curve) or the nominal rate minus survey or breakeven inflation
        expectations (get_inflation_expectations, get_breakeven_inflation_expectations).

        Also known as: real yield, inflation-adjusted interest rate.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rate_type (str, optional): Which nominal interest rate to use. Can be 'long_term'
                (10-year government bond yield), 'short_term' (3-month money market rate) or
                'policy' (central bank policy rate). Defaults to 'long_term'.
            period (str | None, optional): Whether to return the monthly or the yearly data.
                Defaults to None, which is the monthly data.
            gmdb_source (bool | None, optional): Whether to get the yearly long or short-term
                rate from the Global Macro Database (GMDB) instead of the OECD. Defaults to
                None, which falls back to the gmdb_source set on the Economics class (True by
                default). Either way the yearly countries carry the GMDB names, such as
                "Russian Federation", "Eswatini" and "Congo DR", as the yearly inflation
                rate comes from the GMDB.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Real Interest Rate

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-01-01', end_date='2026-09-30')

        economics.get_real_interest_rate(
            countries=['United States', 'Euro Area', 'Japan'], rate_type='policy'
        )
        ```

        Which returns:

        |         |   United States |   Euro Area |   Japan |
        |:--------|----------------:|------------:|--------:|
        | 2026-01 |          0.0124 |      0.003  | -0.0079 |
        | 2026-02 |          0.0121 |      0.001  | -0.0054 |
        | 2026-03 |          0.0037 |     -0.006  | -0.0073 |
        | 2026-04 |         -0.0019 |     -0.01   | -0.006  |
        | 2026-05 |         -0.0062 |     -0.012  | -0.0074 |
        | 2026-06 |          0.0009 |     -0.0055 | -0.0064 |
        | 2026-07 |          0.0026 |     -0.0065 | -0.0095 |
        | 2026-08 |          0.0023 |     -0.0095 | -0.0089 |
        | 2026-09 |          0.0014 |     -0.013  | -0.009  |
        """
        check_period_type(period)

        rate_type = rate_type.lower()

        if rate_type not in ["long_term", "short_term", "policy"]:
            raise ValueError(
                "Please choose 'long_term', 'short_term' or 'policy' for the rate_type "
                "parameter."
            )

        # The inflation rate is monthly or yearly, so the real rate is as well.
        period = validate_period(
            self._default_period(period, "monthly", quarterly_available=False),
            ["monthly", "yearly"],
            "real interest rate",
        )

        if period == "monthly":
            if rate_type == "policy":
                nominal_interest_rate = resample_to_period(
                    self._get_daily_central_bank_policy_rate(
                        buffered_start_date(self._start_date, "monthly"), countries
                    ),
                    "monthly",
                )
            else:
                nominal_interest_rate = (
                    oecd_model.get_long_term_interest_rate
                    if rate_type == "long_term"
                    else oecd_model.get_short_term_interest_rate
                )(
                    period="monthly",
                    start_date=self._start_date,
                    end_date=self._end_date,
                )

            inflation_rate = self._get_monthly_price_data("inflation_rate", countries)

            # Only the countries both legs have, so a country without one is reported as
            # unavailable rather than shown as an empty column.
            shared_countries = nominal_interest_rate.columns.intersection(
                inflation_rate.columns
            )
            nominal_interest_rate = nominal_interest_rate[shared_countries]
            inflation_rate = inflation_rate[shared_countries]
        else:
            gmdb_dataset = self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

            if rate_type == "policy":
                nominal_interest_rate = gmdb_model.get_central_bank_policy_rate(
                    gmd_dataset=gmdb_dataset
                )
            elif gmdb_source:
                nominal_interest_rate = (
                    gmdb_model.get_long_term_interest_rate(gmd_dataset=gmdb_dataset)
                    if rate_type == "long_term"
                    else gmdb_model.get_short_term_interest_rate(
                        gmd_dataset=gmdb_dataset
                    )
                )
            else:
                nominal_interest_rate = (
                    oecd_model.get_long_term_interest_rate
                    if rate_type == "long_term"
                    else oecd_model.get_short_term_interest_rate
                )(
                    period="yearly",
                    start_date=self._start_date,
                    end_date=self._end_date,
                )

                # The inflation leg comes from the GMDB, which names a few countries
                # differently.
                nominal_interest_rate = to_gmdb_country_names(nominal_interest_rate)

            inflation_rate = gmdb_model.get_inflation_rate(gmd_dataset=gmdb_dataset)

        real_interest_rate = nominal_interest_rate - inflation_rate

        return finalize_dataset(
            dataset=real_interest_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Real Interest Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # Months empty for every selected country, e.g. one without inflation yet.
            dropna=period == "monthly",
        )

    @handle_errors
    def get_yield_curve_slope(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Yield Curve Slope for a variety of countries over time. The Yield Curve Slope is
        the difference between the Long Term Interest Rate (the 10-year government bond yield)
        and the Short Term Interest Rate (the 3-month money market rate), and summarizes the
        overall shape of the yield curve in a single number.

        The formula is as follows:

        - Yield Curve Slope (t) = Long Term Interest Rate (t) - Short Term Interest Rate (t), both from the OECD at
          the requested frequency (monthly by default), or both annual from the Global Macro Database

        A positive (upward-sloping) yield curve is the historical norm and reflects investors
        demanding a premium for locking up money for longer. A negative (inverted) yield curve,
        where short-term rates exceed long-term rates, has historically been one of the more
        reliable leading indicators of an upcoming recession, as it signals that markets expect
        the central bank to cut rates in response to a weakening economy.

        Both legs are decimal fractions (0.05 for 5%) whichever source is used, so the result
        is itself a decimal fraction (-0.0122 for an inversion of 1.22 percentage points),
        matching the convention used by get_misery_index and get_real_interest_rate.

        Also known as: term spread, 10Y-3M spread, curve inversion.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Defaults to None, which is quarterly when the Economics class was initialized with
                quarterly=True and monthly otherwise.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB),
                which is annual. Defaults to None, which uses the GMDB only for yearly data and only when the
                gmdb_source set on the Economics class is True (its default); monthly and quarterly data, or
                gmdb_source=False on the class, come from the OECD. True uses the GMDB whatever the period,
                False always the OECD.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Yield Curve Slope

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-01-01', end_date='2023-12-31')

        economics.get_yield_curve_slope(
            countries=['United States', 'Germany', 'Japan'],
            period='yearly'
        )
        ```

        Which returns:

        |      |   United States |   Germany |   Japan |
        |:-----|----------------:|----------:|--------:|
        | 2021 |          0.0139 |    0.0018 | -0      |
        | 2022 |          0.0071 |    0.008  |  0.0016 |
        | 2023 |         -0.0122 |   -0.01   |  0.0049 |
        """
        check_period_type(period)

        period = self._default_period(period, "monthly")
        # The GMDB is annual, so a monthly or quarterly request goes to the sources that
        # have that frequency unless the GMDB is asked for explicitly.
        gmdb_source = (
            gmdb_source
            if gmdb_source is not None
            else self._gmdb_source and period.lower() not in ("monthly", "quarterly")
        )

        if gmdb_source:

            long_term_interest_rate = gmdb_model.get_long_term_interest_rate(
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            )
            short_term_interest_rate = gmdb_model.get_short_term_interest_rate(
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            )
        else:
            long_term_interest_rate = oecd_model.get_long_term_interest_rate(
                period=period,
                start_date=self._start_date,
                end_date=self._end_date,
            )
            short_term_interest_rate = oecd_model.get_short_term_interest_rate(
                period=period,
                start_date=self._start_date,
                end_date=self._end_date,
            )

        yield_curve_slope = long_term_interest_rate - short_term_interest_rate

        return finalize_dataset(
            dataset=yield_curve_slope,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Yield Curve Slope",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_mortgage_rate_30_year(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the weekly average 30-Year Fixed Rate Mortgage from FRED (Freddie
        Mac's Primary Mortgage Market Survey).

        The 30-year fixed mortgage rate is the primary interest rate US households
        actually borrow at for home purchases, and is one of the clearest single
        transmission points from Federal Reserve policy to the real economy: it
        moves with (but is not identical to) the 10-year Treasury yield plus a
        credit/prepayment spread, and directly drives housing affordability and
        demand. It is the natural interest-rate complement to Housing Starts (see
        `get_housing_starts`) — rate moves here lead construction activity, since
        higher borrowing costs price marginal buyers out of the market before
        builders scale back new projects.

        The rate is weekly (week ending Thursday), returned as a decimal fraction per annum
        (0.0648 for 6.48%) and not seasonally adjusted. FRED publishes it in percentage
        points; it is rescaled here so that every rate the Finance Toolkit returns is a
        decimal fraction.

        Without a FRED API key the data comes from Freddie Mac's Primary Mortgage Market
        Survey directly, which gives the same figures. When a FRED API key is set (see the
        `fred_api_key` parameter of the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/MORTGAGE30US

        Also known as: 30-year mortgage rate, Freddie Mac PMMS rate.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of the weekly
            average 30-year fixed mortgage rate, as a decimal fraction per annum.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date="2025-12-31")

        economics.get_mortgage_rate_30_year()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-12-04 |          0.0619 |
        | 2025-12-11 |          0.0622 |
        | 2025-12-18 |          0.0621 |
        | 2025-12-24 |          0.0618 |
        | 2025-12-31 |          0.0615 |
        """
        # FRED republishes these figures, so it is used when a key is set and
        # Freddie Mac otherwise.
        mortgage_rate_30_year = (
            fred_model.get_mortgage_rate_30_year(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else freddie_mac_model.get_mortgage_rate_30_year()
        )

        return finalize_dataset(
            dataset=mortgage_rate_30_year,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    def _get_united_states_inflation_curve(
        self, curve: str, start_date: str
    ) -> pd.DataFrame:
        """
        Retrieves the US real yield curve or breakeven inflation curve from the U.S.
        Department of the Treasury, with the maturities labelled as elsewhere ("5Y" and
        "5Y5Y" for the 5-year, 5-year forward rate).

        Args:
            curve (str): "real" or "breakeven".
            start_date (str): The start date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: The curve, indexed by day with a column per maturity.
        """
        # The Treasury publishes the TIPS yields at the end of each business day; FRED
        # republishes them (DFII) a day later, so the latest day would be missing.
        values = (
            treasury_model.get_real_yield_curve(start_date, self._end_date)
            if curve == "real"
            else treasury_model.get_breakeven_inflation_expectations(
                start_date, self._end_date
            )
        )

        return values.rename(
            columns=lambda column: (
                "5Y5Y"
                if column == "5 Year, 5 Year Forward"
                else column.replace(" Year", "Y")
            )
        )

    def _get_bank_of_england_curve(self, curve: str, start_date: str) -> pd.DataFrame:
        """
        Retrieves a Bank of England spot curve at whole-year maturities.

        Args:
            curve (str): "real" or "inflation".
            start_date (str): The start date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: The curve, indexed by day with a column per whole-year maturity.
        """
        values = boe_model.get_spot_curve(curve, start_date, self._end_date)

        # The Bank of England estimates the curve in steps of half a year; the whole years
        # keep the columns comparable with the other countries.
        return values[
            [column for column in values.columns if float(column[:-1]).is_integer()]
        ]

    @handle_errors
    def get_real_yield_curve(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the real yield curve of a variety of countries: the yields of inflation-linked
        government bonds, which pay a return on top of inflation, by maturity. Together with
        the nominal curve (see `fixedincome.get_government_bond_yield_curve`) it gives the
        inflation the bond market expects, see `get_breakeven_inflation_expectations`.

        Every curve comes from the official source:
        - United States: the U.S. Department of the Treasury's real par yield curve of
          Treasury Inflation-Protected Securities (TIPS), 5, 7, 10, 20 and 30 years, from
          2003, the figures FRED republishes a day later as DFII.
        - United Kingdom: the Bank of England's real spot curve of index-linked gilts, 3 to
          40 years, from 1985. Index-linked gilts pay the Retail Prices Index (RPI).
        - Germany: the real yields of the inflation-linked German federal securities the
          Bundesbank publishes, interpolated to constant maturities between 3 and 30 years
          from 2014. Germany has four to six of these bonds outstanding at a time, so only
          the maturities between its shortest (at least two years) and longest bond are
          filled. The bonds are indexed to euro area inflation (HICP excluding tobacco).

        Yields are returned as a decimal fraction per annum (0.0174 for 1.74%), with one
        column per country and maturity. Weekly and monthly periods take the curve on the
        last trading day of each period.

        See definition: https://fred.stlouisfed.org/series/DFII10

        Also known as: TIPS yield curve, real Treasury yield curve, index-linked gilt curve,
        Bund linker real yields, inflation-linked bond yields.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "United States", "United Kingdom" and "Germany". Defaults to None, which
                retrieves every country.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The real yields as decimals, indexed by date with a column per
            country and maturity.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        real_yield_curve = economics.get_real_yield_curve(period='monthly')

        real_yield_curve.xs('10Y', axis=1, level='Maturity')
        ```

        Which returns:

        |         |   United States |   United Kingdom |   Germany |
        |:--------|----------------:|-----------------:|----------:|
        | 2026-04 |          0.0194 |           0.0154 |    0.0075 |
        | 2026-05 |          0.0207 |           0.0155 |    0.0084 |
        | 2026-06 |          0.022  |           0.0172 |    0.0095 |
        | 2026-07 |          0.0247 |           0.0188 |    0.0106 |
        | 2026-08 |          0.0244 |           0.0184 |    0.0108 |
        | 2026-09 |          0.0293 |           0.02   |    0.0132 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "real yield curve"
        )

        real_yield_curve = self._collect_country_curves(
            {
                "United States": lambda start: self._get_united_states_inflation_curve(
                    "real", start
                ),
                "United Kingdom": lambda start: self._get_bank_of_england_curve(
                    "real", start
                ),
                "Germany": lambda start: bundesbank_model.get_real_yield_curve(
                    start, self._end_date
                ),
            },
            countries,
            period,
            "real yield curve",
        )

        return finalize_dataset(
            dataset=real_yield_curve,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_breakeven_inflation_expectations(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the market-implied (Q-measure) breakeven inflation of a variety of countries by
        maturity: the nominal government bond yield minus the real yield of an
        inflation-linked bond of the same maturity, which is the inflation rate at which
        both earn the same. It is the bond market's expectation of inflation, including an
        inflation risk premium. See `get_inflation_expectations` for the expectations of
        professional forecasters instead.

        Every curve comes from the official source:
        - United States: the U.S. Department of the Treasury's nominal minus real (TIPS) par
          yields, 5, 7, 10, 20 and 30 years from 2003, plus the 5-year, 5-year forward rate
          ("5Y5Y"), the average inflation expected over the five years starting five years
          from now, with the formulas FRED uses for T5YIE, T10YIE and T5YIFR.
        - United Kingdom: the Bank of England's implied inflation spot curve, 3 to 40 years,
          from 1985. It is measured against the Retail Prices Index (RPI), which index-linked
          gilts pay and which has run above CPI inflation.
        - Germany: the nominal term structure of German federal securities at the remaining
          maturity of each inflation-linked federal security minus its real yield,
          interpolated to constant maturities between 3 and 30 years from 2014. The bonds are
          indexed to euro area inflation (HICP excluding tobacco), which makes this the
          euro area's market-implied inflation expectation, priced off its benchmark issuer,
          and a free proxy for euro area inflation swaps, which are licensed data. Only the
          maturities between the shortest (at least two years) and longest bond are filled.

        Rates are returned as a decimal fraction per annum (0.0221 for 2.21%), with one column
        per country and maturity. Weekly and monthly periods take the curve on the last
        trading day of each period.

        Investors compare breakeven inflation with their own inflation view to choose
        between nominal and inflation-linked bonds. Besides the inflation risk premium, it
        holds a liquidity premium: inflation-linked bonds trade less easily than nominal
        bonds, which lowers breakevens, most visibly in market stress (2008, March 2020),
        when they fell further than inflation expectations did.

        See definition: https://fred.stlouisfed.org/series/T10YIE

        Also known as: breakeven inflation rate, market-implied inflation expectations, UK
        implied inflation, euro area breakeven inflation, Bund linker breakeven.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "United States", "United Kingdom" and "Germany". Defaults to None, which
                retrieves every country.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The breakeven inflation as decimals, indexed by date with a column
            per country and maturity.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        breakeven_inflation = economics.get_breakeven_inflation_expectations(period='monthly')

        breakeven_inflation.xs('10Y', axis=1, level='Maturity')
        ```

        Which returns:

        |         |   United States |   United Kingdom |   Germany |
        |:--------|----------------:|-----------------:|----------:|
        | 2026-04 |          0.0246 |           0.0352 |    0.0234 |
        | 2026-05 |          0.0238 |           0.0331 |    0.0211 |
        | 2026-06 |          0.0224 |           0.0311 |    0.019  |
        | 2026-07 |          0.0228 |           0.0322 |    0.0208 |
        | 2026-08 |          0.0231 |           0.033  |    0.0221 |
        | 2026-09 |          0.0236 |           0.0341 |    0.0224 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "breakeven inflation"
        )

        breakeven_inflation = self._collect_country_curves(
            {
                "United States": lambda start: self._get_united_states_inflation_curve(
                    "breakeven", start
                ),
                "United Kingdom": lambda start: self._get_bank_of_england_curve(
                    "inflation", start
                ),
                "Germany": lambda start: bundesbank_model.get_breakeven_inflation(
                    start, self._end_date
                ),
            },
            countries,
            period,
            "breakeven inflation",
        )

        return finalize_dataset(
            dataset=breakeven_inflation,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_yield_decomposition(
        self,
        countries: list[str] | str | None = None,
        maturities: list[str] | str = "10Y",
        period: str = "daily",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Split the nominal government bond yield of a maturity into the real yield of the
        inflation-linked bond of that maturity and the breakeven inflation, the part the
        market demands for inflation, with the change of each part since the previous period:

        - Nominal Yield = Real Yield + Breakeven Inflation

        A rise of the nominal yield is either a rise of real rates (tighter monetary policy,
        stronger growth, more supply of bonds) or of inflation compensation (a higher
        expectation of inflation or a larger inflation risk premium), and the split says
        which: a 5-year US yield up 0.14 with an unchanged breakeven rose on real rates alone.

        The curves are those of `get_real_yield_curve` and
        `get_breakeven_inflation_expectations`, from the official sources:
        the U.S. Department of the Treasury (par yields, 5 to 30 years), the Bank of England
        (spot rates, 3 to 40 years) and, for Germany, the Bundesbank (constant maturities
        between 3 and 30 years). Each source's breakeven is its nominal yield minus its real
        yield, so the parts add up to the nominal yield in that source's own compounding. UK
        breakevens are measured against the Retail Prices Index (RPI) and German ones against
        euro area inflation (HICP excluding tobacco).

        The changes are the difference with the previous period the country has a curve for,
        in decimals (0.0014 for 14 basis points), so a market holiday is not a change. The first
        period of the date range has no change.

        Also known as: real yield and breakeven split, nominal yield decomposition,
        inflation compensation, real rate versus inflation.

        Args:
            countries (list[str] | str | None, optional): The countries, from the United
                States, the United Kingdom and Germany. Defaults to None, which is all three.
            maturities (list[str] | str, optional): The maturities, such as "5Y" or
                ["5Y", "10Y"], each one both curves of a country have. Defaults to "10Y".
            period (str, optional): Whether to return the daily, weekly or monthly data, the
                curve on the last trading day of each week or month. Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Nominal Yield, Real Yield, Breakeven Inflation and their Nominal
            Change, Real Change and Breakeven Change per country and maturity, with the dates
            as the index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-11-02', end_date='2026-11-10')

        economics.get_yield_decomposition(
            countries=['United States', 'Germany'], maturities='5Y'
        ).xs(('United States', '5Y'), axis=1)
        ```

        Which returns:

        |            |   Nominal Yield |   Real Yield |   Breakeven Inflation |   Nominal Change |   Real Change |   Breakeven Change |
        |:-----------|----------------:|-------------:|----------------------:|-----------------:|--------------:|-------------------:|
        | 2026-11-02 |          0.047  |       0.0231 |                0.0239 |         nan      |      nan      |           nan      |
        | 2026-11-03 |          0.0474 |       0.0235 |                0.0239 |           0.0004 |        0.0004 |             0      |
        | 2026-11-04 |          0.0469 |       0.0239 |                0.023  |          -0.0005 |        0.0004 |            -0.0009 |
        | 2026-11-05 |          0.0459 |       0.0233 |                0.0226 |          -0.001  |       -0.0006 |            -0.0004 |
        | 2026-11-06 |          0.0464 |       0.0235 |                0.0229 |           0.0005 |        0.0002 |             0.0003 |
        | 2026-11-09 |          0.0467 |       0.0242 |                0.0225 |           0.0003 |        0.0007 |            -0.0004 |
        | 2026-11-10 |          0.0481 |       0.0256 |                0.0225 |           0.0014 |        0.0014 |             0      |
        """
        maturities = [
            str(maturity).strip().upper()
            for maturity in helpers.split_values(maturities)
        ]
        # Unrounded, so the parts add up and the changes are exact; rounded once at the end.
        unrounded = 12
        real_yield = self.get_real_yield_curve(
            countries=countries, period=period, rounding=unrounded
        )
        breakeven_inflation = self.get_breakeven_inflation_expectations(
            countries=countries, period=period, rounding=unrounded
        )

        decompositions = {}
        for country, maturity in itertools.product(
            dict.fromkeys(real_yield.columns.get_level_values(0)), maturities
        ):
            if (country, maturity) not in real_yield.columns or (
                country,
                maturity,
            ) not in breakeven_inflation.columns:
                logger.warning(
                    "%s has no %s real yield and breakeven inflation, so it is left out.",
                    country,
                    maturity,
                )
                continue

            parts = pd.DataFrame(
                {
                    "Real Yield": real_yield[(country, maturity)],
                    "Breakeven Inflation": breakeven_inflation[(country, maturity)],
                }
            ).dropna()
            parts.insert(
                0, "Nominal Yield", parts["Real Yield"] + parts["Breakeven Inflation"]
            )
            changes = parts.diff()
            changes.columns = ["Nominal Change", "Real Change", "Breakeven Change"]
            decompositions[(country, maturity)] = pd.concat([parts, changes], axis=1)

        if not decompositions:
            raise ValueError(
                f"No country has a {', '.join(maturities)} real yield and breakeven inflation; the "
                "maturities run from 5 to 30 years for the United States and from 3 years "
                "for the United Kingdom and Germany."
            )

        decomposition = pd.concat(decompositions, axis=1)
        decomposition.columns.names = ["Country", "Maturity", None]

        return apply_rounding(
            decomposition, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_inflation_expectations(
        self,
        countries: list[str] | str | None = None,
        period: str = "monthly",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the inflation professional forecasters expect over the longer term, from surveys.
        Where `get_breakeven_inflation_expectations` is the bond market's expectation,
        priced daily and including risk premia, this is the forecasters' own expectation,
        with no inflation risk or liquidity premium in it and a longer history.

        Every series comes from the official source:
        - Germany: the inflation Consensus Economics' forecasters expect over 5 and 10 years,
          monthly from 1989. The Bundesbank publishes the expected real interest rate, the
          yield on debt securities outstanding issued by German residents with a residual
          maturity of 5 to 6 (or 9 to 10) years minus the expected inflation, so adding that
          yield back gives the expected inflation.
        - Euro Area: the average longer-term (five years ahead) HICP inflation forecast of the
          ECB's quarterly Survey of Professional Forecasters, from 1999. With
          period="monthly" each survey round is shown from the first month of its quarter
          until the next round.

        Rates are returned as a decimal fraction (0.0223 for 2.23%), with one column per
        country and horizon.

        See definition: https://www.ecb.europa.eu/stats/ecb_surveys/survey_of_professional_forecasters/html/index.en.html

        Also known as: survey-based inflation expectations, long-term inflation expectations,
        Consensus Economics inflation forecasts, Survey of Professional Forecasters.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "Germany" and "Euro Area". Defaults to None, which retrieves every country.
            period (str, optional): Whether to return the monthly or quarterly data. Quarterly
                data takes the last month of each quarter. Defaults to "monthly".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The expected inflation as decimals, indexed by month or quarter with
            a column per country and horizon.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        economics.get_inflation_expectations().xs('5Y', axis=1, level='Maturity')
        ```

        Which returns:

        |         |   Germany |   Euro Area |
        |:--------|----------:|------------:|
        | 2026-04 |    0.0218 |      0.0203 |
        | 2026-05 |    0.0216 |      0.0203 |
        | 2026-06 |    0.0212 |      0.0203 |
        | 2026-07 |    0.0218 |      0.0204 |
        | 2026-08 |    0.0219 |      0.0204 |
        | 2026-09 |    0.0223 |      0.0204 |
        """
        period = validate_period(
            period, ["monthly", "quarterly"], "inflation expectations"
        )
        start_date = buffered_start_date(self._start_date, "monthly")

        def euro_area() -> pd.DataFrame:
            survey = ecb_model.get_survey_inflation_expectations(
                start_date, self._end_date
            ).rename(columns={"Euro Area": "5Y"})

            if survey.empty or period == "quarterly":
                return survey

            # A survey round holds until the next one, a quarter later.
            survey.index = survey.index.asfreq("M", how="start")
            months = pd.period_range(survey.index[0], survey.index[-1] + 2, freq="M")

            return survey.reindex(months).ffill(limit=2)

        def germany() -> pd.DataFrame:
            expectations = bundesbank_model.get_survey_inflation_expectations(
                start_date, self._end_date
            )

            if expectations.empty or period == "monthly":
                return expectations

            return expectations.groupby(expectations.index.asfreq("Q")).last()

        sources = {"Germany": germany, "Euro Area": euro_area}

        if countries is not None and not isinstance(countries, str | list | tuple):
            raise TypeError(
                "The countries must be a country name or a list of country names, such as "
                f"'Germany' or ['Germany', 'Euro Area'], not a {type(countries).__name__} ({countries!r})."
            )

        requested = (
            list(sources)
            if countries is None
            else [countries] if isinstance(countries, str) else list(countries)
        )

        if unavailable := [country for country in requested if country not in sources]:
            # Worded as missing data, so the MCP server reports it as a note.
            logger.warning(
                "No inflation expectations data is available for %s. They cover %s.",
                ", ".join(unavailable),
                ", ".join(sources),
            )

        expectations = {
            country: values
            for country in requested
            if country in sources and not (values := sources[country]()).empty
        }

        if not expectations:
            return pd.DataFrame()

        inflation_expectations = pd.concat(expectations, axis=1).sort_index()
        inflation_expectations.index.name = None
        inflation_expectations.columns.names = ["Country", "Maturity"]

        return finalize_dataset(
            dataset=inflation_expectations,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )
