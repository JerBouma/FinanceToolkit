"""Prices Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit import helpers
from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    OVERNIGHT_RATE_DAY_COUNTS,
    average_monthly_index,
    buffered_start_date,
    change_over_periods,
    check_period_type,
    compound_overnight_rate,
    rebase_to_first,
    resample_to_period,
    validate_consumer_price_measure,
    validate_period,
    warn_about_long_run_prices,
)
from financetoolkit.economics.sources import (
    bis_model,
    bls_model,
    boj_model,
    eurostat_model,
    fred_model,
    gmdb_model,
    oecd_model,
    world_bank_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import (
    track_sources,
)
from financetoolkit.utilities.statistics_model import (
    EURO_AREA_MEMBERS,
    finalize_dataset,
)

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class PricesEconomics(EconomicsBase):
    """
    Prices: consumer prices and inflation, purchasing power, producer, export and import prices, house, rent and commercial real estate prices, and commodity prices.
    """

    def _get_monthly_industrial_prices(
        self, measure: str, countries: list[str] | str | None = None
    ) -> pd.DataFrame:
        """
        Combines a monthly industrial price index of every country from its official
        publisher: Eurostat (the euro area, the European
        Union and around 35 European countries), the Bureau of Labor Statistics (United
        States, through FRED when a FRED API key is set) and the Bank of Japan (Japan).

        Args:
            measure (str): "producer" for the producer price index, "export" for the
                export price index or "import" for the import price index.
            countries (list[str] | str | None, optional): The requested countries, so the
                sources of other countries are not retrieved. Defaults to None.

        Returns:
            pd.DataFrame: One column per country, indexed by month.
        """
        # Only the requested months are asked for, with room for growth and rolling.
        start_date = buffered_start_date(self._start_date, "monthly")
        end_date = self._end_date

        # FRED republishes the BLS series in full, so it is used when a key is set; the
        # keyless BLS API returns about the last three years.
        united_states = (
            (
                lambda: fred_model.get_price_index(
                    measure, start_date, end_date, self._fred_api_key
                )
            )
            if self._fred_api_key
            else (lambda: bls_model.get_price_index(measure))
        )

        prices = self._combine_sources_for(
            [
                (
                    lambda: eurostat_model.get_industrial_price_index(
                        measure, start_date, end_date
                    ),
                    None,
                ),
                (united_states, "United States"),
                (lambda: boj_model.collect_price_index(measure), "Japan"),
            ],
            countries,
        )

        if (
            not self._fred_api_key
            and "United States" in prices.columns
            and prices["United States"].first_valid_index()
            > pd.Period(self._start_date, freq="M")
        ):
            logger.warning(
                "Without a FRED API key, the US %s price index starts in %s: the keyless BLS "
                "API only returns about the last three years. Set a FRED API key for the "
                "full history.",
                measure,
                prices["United States"].first_valid_index().strftime("%Y-%m"),
            )

        return prices

    def _get_consumer_price_index(
        self,
        countries: list[str] | str | None,
        period: str,
        oecd_source: bool = False,
        gmdb_forecasts: bool = False,
        measure: str = "all",
    ) -> pd.DataFrame:
        """
        Retrieves the consumer price index, unrounded and for the full history retrieved,
        from the source the period and oecd_source select (see get_consumer_price_index).

        Args:
            countries (list[str] | str | None): The requested countries, so the sources of
                other countries are not retrieved.
            period (str): "monthly", "quarterly" or "yearly", already validated.
            oecd_source (bool, optional): Whether to retrieve the data from the OECD.
                Defaults to False.
            gmdb_forecasts (bool, optional): Whether to include the projections of the
                Global Macro Database. Defaults to False.
            measure (str, optional): All items, core or a component, already validated
                (monthly only). Defaults to "all".

        Returns:
            pd.DataFrame: The consumer price index, a column per country.
        """
        if oecd_source:
            return oecd_model.get_consumer_price_index(
                period=period, start_date=self._start_date, end_date=self._end_date
            )

        if period.lower() == "monthly":
            return self._get_monthly_price_data(
                "consumer_price_index", countries, measure
            )

        return gmdb_model.get_consumer_price_index(
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
        )

    @handle_errors
    def get_consumer_price_index(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        oecd_source: bool = False,
        measure: str = "all",
        cumulative: bool = False,
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
        Consumer Price Index (CPI) is a measure that examines the average change in prices
        paid by consumers for goods and services over time. It is a measure of inflation.

        By default the data is monthly, from the most current national source per country
        (see below). With period="yearly" the data comes from the Global Macro Database
        (GMDB), annual with 2015 = 100. Set `oecd_source=True` to retrieve monthly or
        quarterly data from the OECD instead (base year varies per country).

        The yearly index is chained across currency reforms and redenominations, in today's
        currency. A warning names the years in the date range where it repeats the value
        of the year before (carried forward by the source, such as Germany's 1913 and
        1914, so their inflation is 0 by construction) and the hyperinflations (prices
        rising more than tenfold in a year), after which the yearly average still rises
        sharply even when a currency reform ended them within the year.

        With period="monthly" and without oecd_source, every country comes from its most
        current source: Eurostat for the euro area and the European
        Economic Area (the HICP, 2015 = 100), the Office for National Statistics for the
        United Kingdom (CPI, 2015 = 100), the Statistics Bureau of Japan for Japan (CPI
        from 2015, on its latest base) and the Bank for International Settlements for
        every other country (2010 = 100). The base year differs between countries, so
        compare the changes rather than the levels.

        The measure parameter selects the core index (all items less energy and food;
        Eurostat also leaves out alcohol and tobacco) or a component: energy, food (Eurostat
        includes alcohol and tobacco, the BLS does not), goods, services and, for the United
        States, shelter. These are monthly, from Eurostat (the euro area and the European
        Economic Area) and the Bureau of Labor Statistics for the United States (CPI-U, not
        seasonally adjusted, 1982-84 = 100; the full history through FRED when a FRED API
        key is set, the last three years otherwise). Other countries have all items only.

        The US index of the BIS is the BLS's CPI-U rebased to 2010 = 100, which the BIS
        publishes a few days after the BLS. The months the BLS has published since are
        added to it, so a release appears the same day.

        With cumulative=True, every country's index is divided by its value in the first
        period of the date range: the growth of 1 in prices since the start_date (1.3460
        for prices 34.60% higher), which compares countries regardless of their base year.
        A country without data in the first period starts from its own first period. The
        purchasing power of money is its inverse (see get_purchasing_power).

        Further information about the Global Macro Database (GMDB) variable can be found
        within https://www.globalmacrodata.com/documentation.html

        Also known as: CPI, cost of living index, core CPI, CPI components, energy prices,
        food prices, shelter costs.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Without `oecd_source=True`, "monthly" uses the monthly sources described above and the
                annual data comes from the GMDB. Defaults to None, which is the monthly data (quarterly
                with `oecd_source=True` when the Economics class was initialized with quarterly=True).
            oecd_source (bool, optional): Whether to get the data from the OECD instead of the
                Global Macro Database (GMDB). Defaults to False.
            measure (str, optional): "all" for all items, "core" for all items less food and
                energy, or a component: "energy", "food", "goods", "services" or "shelter"
                (United States only). Other measures than "all" are monthly, without
                oecd_source. Defaults to "all".
            cumulative (bool, optional): Whether to return the growth of 1 in prices since the
                first period of the date range instead of the index. Defaults to False.
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
            pd.DataFrame: A DataFrame containing the Consumer Price Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2008-09-01', end_date='2020-03-01')

        economics.get_consumer_price_index(countries=['Germany', 'France', 'Portugal'], period='yearly')
        ```

        Which returns:

        |      |   Germany |   France |   Portugal |
        |:-----|----------:|---------:|-----------:|
        | 2008 |   91.9238 |  93.2046 |    92.9047 |
        | 2009 |   92.2236 |  93.2862 |    92.1285 |
        | 2010 |   93.2552 |  94.7165 |    93.4207 |
        | 2011 |   95.1772 |  96.7132 |    96.8332 |
        | 2012 |   97.064  |  98.604  |    99.5189 |
        | 2013 |   98.5364 |  99.4547 |    99.7919 |
        | 2014 |   99.4445 |  99.9597 |    99.5142 |
        | 2015 |  100      | 100      |   100      |
        | 2016 |  100.467  | 100.184  |   100.608  |
        | 2017 |  101.975  | 101.216  |   101.984  |
        | 2018 |  103.765  | 103.09   |   102.998  |
        | 2019 |  105.237  | 104.231  |   103.346  |
        | 2020 |  105.801  | 104.729  |   103.333  |
        """
        check_period_type(period)

        period = self._default_period(
            period, "monthly", quarterly_available=oecd_source
        )
        if not oecd_source:
            period = validate_period(
                period,
                ["monthly", "yearly"],
                "consumer price index (quarterly data needs oecd_source=True)",
            )

        measure = validate_consumer_price_measure(measure, period, oecd_source)
        consumer_price_index = self._get_consumer_price_index(
            countries, period, oecd_source, gmdb_forecasts, measure
        )

        if period == "yearly" and not oecd_source:
            warn_about_long_run_prices(
                consumer_price_index, countries, self._start_date, self._end_date
            )

        if cumulative:
            # Rebased only after slicing to the date range, so the growth starts at 1 in
            # the first period of the range.
            consumer_price_index = rebase_to_first(
                consumer_price_index.loc[self._start_date : self._end_date]
            )

        return finalize_dataset(
            dataset=consumer_price_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Consumer Price Index",
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
            dropna=not oecd_source
            and period is not None
            and period.lower() == "monthly",
        )

    @handle_errors
    def get_inflation_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        measure: str = "all",
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
        Inflation Rate is the percentage change in the Consumer Price Index (CPI) from one
        period to another. It is a measure of the rate of price increases in the economy.

        With period="yearly" the data comes from the Global Macro Database (GMDB). Its years
        that are IMF World Economic Outlook projections are only included with
        gmdb_forecasts=True; by default a series ends with its last observation. Further
        information about the variable can be found within
        https://www.globalmacrodata.com/documentation.html

        The yearly index is chained across currency reforms and redenominations, in today's
        currency. A warning names the years in the date range where it repeats the value
        of the year before (carried forward by the source, such as Germany's 1913 and
        1914, so their inflation is 0 by construction) and the hyperinflations (prices
        rising more than tenfold in a year), after which the yearly average still rises
        sharply even when a currency reform ended them within the year.

        With period="monthly", the default, the annual rate of change is returned for every
        month, the figure statistical offices publish, taken per country from the most
        current source: Eurostat for the euro area and the European
        Economic Area (the HICP, including the flash estimate at the end of the month
        itself), the Office for National Statistics for the United Kingdom (CPI), the
        Statistics Bureau of Japan for Japan (CPI, from 2016), the Bank for International
        Settlements for around sixty other countries, including the United States, China,
        India and Brazil, and the IMF for around a hundred more, all republishing the
        national consumer price indices.

        The measure parameter selects the inflation of the core index (all items less energy and food;
        Eurostat also leaves out alcohol and tobacco) or a component: energy, food (Eurostat
        includes alcohol and tobacco, the BLS does not), goods, services and, for the United
        States, shelter. These are monthly, from Eurostat (the euro area and the European
        Economic Area) and the Bureau of Labor Statistics for the United States (CPI-U, not
        seasonally adjusted, 1982-84 = 100; the full history through FRED when a FRED API
        key is set, the last three years otherwise). Other countries have all items only.

        The US index of the BIS is the BLS's CPI-U rebased to 2010 = 100, which the BIS
        publishes a few days after the BLS. The months the BLS has published since are
        added to it, so a release appears the same day.

        The result is a decimal fraction (0.041166 for 4.1166%), like every other rate in the
        Finance Toolkit.

        The rates are the sources' own figures (published in percent and divided by 100 here),
        except for Japan, which is calculated from the Statistics Bureau of Japan's CPI.

        The formula is as follows:

        - Monthly Inflation Rate (t) = CPI (t) / CPI (t-12 months) - 1, the annual rate of change for every month
        - Yearly Inflation Rate (t) = CPI (t) / CPI (t-1 year) - 1, the Global Macro Database's annual figure

        Headline inflation is what most central banks target, usually at 2% a year, but they
        read it together with core inflation, which excludes energy and food, to see past
        temporary price shocks. Year-on-year rates react to base effects: a large price jump
        a year earlier lowers today's rate once it drops out of the comparison. The Eurostat
        figures are the harmonised index (HICP), which differs from national CPIs, for
        example in leaving out the costs of owner-occupied housing.

        Also known as: CPI-based inflation, price increases, consumer prices, core
        inflation, energy inflation, food inflation, services inflation, shelter inflation.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly or the annual data.
                Defaults to None, which is the monthly data.
            measure (str, optional): "all" for all items, "core" for all items less food and
                energy, or a component: "energy", "food", "goods", "services" or "shelter"
                (United States only). Other measures than "all" are monthly. Defaults to "all".
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
            pd.DataFrame: A DataFrame containing the Inflation Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2003-01-01', end_date='2009-03-01')

        economics.get_inflation_rate(countries=['Germany', 'France', 'Portugal'], period='yearly')
        ```

        Which returns:

        |      |   Germany |   France |   Portugal |
        |:-----|----------:|---------:|-----------:|
        | 2003 |    0.0107 |   0.0217 |     0.0324 |
        | 2004 |    0.0177 |   0.0234 |     0.0251 |
        | 2005 |    0.0195 |   0.0189 |     0.0213 |
        | 2006 |    0.0177 |   0.0189 |     0.0305 |
        | 2007 |    0.0229 |   0.016  |     0.0242 |
        | 2008 |    0.0275 |   0.0316 |     0.0265 |
        | 2009 |    0.0023 |   0.001  |    -0.009  |
        """
        check_period_type(period)

        period = validate_period(
            self._default_period(period, "monthly", quarterly_available=False),
            ["monthly", "yearly"],
            "inflation rate",
        )
        measure = validate_consumer_price_measure(measure, period)

        if period == "monthly":
            inflation_rate = self._get_monthly_price_data(
                "inflation_rate", countries, measure
            )
        else:
            gmdb_dataset = self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            inflation_rate = gmdb_model.get_inflation_rate(gmd_dataset=gmdb_dataset)
            warn_about_long_run_prices(
                gmdb_model.get_consumer_price_index(gmd_dataset=gmdb_dataset),
                countries,
                self._start_date,
                self._end_date,
            )

        return finalize_dataset(
            dataset=inflation_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Inflation Rate",
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
            dropna=period == "monthly",
        )

    @handle_errors
    def get_purchasing_power(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        overnight_rate: bool = False,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Purchasing Power is what 100 units of a currency at the start of the date range
        buy in every later period, in the prices of the start: it falls as consumer prices
        rise, so 74.29 means that money has lost a quarter of its value. It is the inverse
        of the growth in the Consumer Price Index since the first period of the range (see
        get_consumer_price_index with cumulative=True), on the same monthly sources by
        default (period="yearly" for the Global Macro Database's annual averages).

        With overnight_rate=True, the 100 is not held as banknotes but deposited at the
        overnight rate of its currency, compounded day by day (see get_overnight_rate with
        cumulative=True): €STR for the Euro Area and its members, SONIA for the United
        Kingdom, TONA for Japan and SOFR for the United States. The result is the real
        value of cash, which stays above 100 only when the interest outpaced inflation.
        Countries without an overnight rate are left out with a note. A euro member earns
        €STR over the whole range, also before it adopted the euro.

        The formula is as follows:

        - Purchasing Power (t) = 100 × CPI (start) / CPI (t)
        - Purchasing Power of Cash (t) = 100 × Overnight Rate Growth (t) × CPI (start) / CPI (t)

        CPI (start) is each country's index in the first period of the date range (its own
        first period when it starts later). The cash growth starts at 1 on the first day of
        the range and every month takes it on the last day of the month, next to the index
        of that same month (the reference month). The index of a month is published some
        weeks after it ends, so the latest months of the cash growth are left out until
        their index is published rather than paired with older prices.

        Also known as: real value of money, value of money, real value of cash, real return
        on cash, inflation erosion.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data.
                Defaults to None, which is every country with a consumer price index, or the
                Euro Area, Japan, the United Kingdom and the United States with
                overnight_rate=True.
            period (str | None, optional): Whether to return the monthly or the annual data.
                Defaults to None, which is the monthly data. The overnight rate is only
                combined with the monthly data.
            overnight_rate (bool, optional): Whether the 100 earns the overnight rate of its
                currency instead of being held as banknotes. Defaults to False.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The purchasing power of 100 per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date='2026-09-30')

        economics.get_purchasing_power(
            countries=['Euro Area', 'Netherlands', 'United States'], overnight_rate=True
        ).iloc[::12]
        ```

        Which returns:

        |         |   Euro Area |   Netherlands |   United States |
        |:--------|------------:|--------------:|----------------:|
        | 2020-01 |     99.9567 |       99.9567 |        100.124  |
        | 2021-01 |     98.5071 |       97.8545 |         98.9891 |
        | 2022-01 |     93.1777 |       90.4128 |         92.1353 |
        | 2023-01 |     85.9347 |       83.578  |         88.3553 |
        | 2024-01 |     86.5277 |       83.8851 |         90.2447 |
        | 2025-01 |     87.5091 |       84.4527 |         92.2405 |
        | 2026-01 |     87.9163 |       84.4219 |         93.9886 |
        """
        check_period_type(period)

        period = validate_period(
            self._default_period(period, "monthly", quarterly_available=False),
            ["monthly", "yearly"],
            "purchasing power",
        )

        if overnight_rate and period != "monthly":
            raise ValueError(
                "The purchasing power of cash at the overnight rate is only available with "
                "period='monthly', as the interest is compounded to the end of every month."
            )

        purchasing_power = 100 / rebase_to_first(
            self._get_consumer_price_index(countries, period).loc[
                self._start_date : self._end_date
            ]
        )

        if overnight_rate:
            cash = resample_to_period(
                compound_overnight_rate(
                    self._get_overnight_rate(None).loc[
                        self._start_date : self._end_date
                    ],
                    OVERNIGHT_RATE_DAY_COUNTS,
                ),
                "monthly",
            )

            # The euro members hold euros, which earn €STR; they are only included when
            # asked for, so the default is one column per currency.
            areas = {
                country: country if country in cash.columns else "Euro Area"
                for country in purchasing_power.columns
                if country in cash.columns
                or (
                    countries is not None
                    and country in EURO_AREA_MEMBERS
                    and "Euro Area" in cash.columns
                )
            }
            purchasing_power = pd.DataFrame(
                {
                    country: purchasing_power[country]
                    * cash[area].reindex(purchasing_power.index)
                    for country, area in areas.items()
                },
                index=purchasing_power.index,
            )

        return finalize_dataset(
            dataset=purchasing_power,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name=(
                "Purchasing Power of Cash" if overnight_rate else "Purchasing Power"
            ),
            countries=countries,
            growth=growth,
            lag=lag,
            rounding=rounding,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_producer_price_index(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Producer Price Index (PPI) for a variety of countries over time. The PPI
        measures the average change over time in the prices domestic producers receive for
        their output. Because producers tend to pass rising input costs on to their
        customers with a lag, the PPI is generally seen as a leading, upstream indicator of
        cost pressure that later shows up in the Consumer Price Index (CPI).

        Each country's headline index comes monthly from its official publisher:

        - Eurostat for the euro area, the European Union and around 35 European countries:
          the industrial producer prices on the domestic market (industry except
          construction, sewerage and waste management), 2021 = 100.
        - The Bureau of Labor Statistics for the United States: the PPI for final demand,
          November 2009 = 100. The BLS API returns about the last three years; with a FRED
          API key the full history comes from FRED, which republishes it.
        - The Bank of Japan for Japan: the Corporate Goods Price Index (domestic), 2020 =
          100, from 1960.

        China is not included: the National Bureau of Statistics of China publishes its
        producer prices as changes rather than as an index, and its database refuses
        automated requests. Its monthly year-on-year figure is in get_economic_calendar.

        All indices are not seasonally adjusted, so changes are best taken year over year
        (growth=True with lag=12 on monthly data), which returns them as decimals (0.057 for
        5.7%). The base year differs per country, so compare growth rather than levels.
        Quarterly and yearly values are the average of the months, for complete periods only.

        See definition: https://ec.europa.eu/eurostat/cache/metadata/en/sts_inpp_esms.htm

        Also known as: PPI, wholesale prices, factory gate prices, upstream inflation.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Defaults to None, which is quarterly when the Economics class was initialized with
                quarterly=True and monthly otherwise.
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
            pd.DataFrame: A DataFrame containing the Producer Price Index per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-10-01', end_date='2026-09-30')

        economics.get_producer_price_index(
            countries=['United States', 'Euro Area', 'Japan'],
            period='monthly',
            growth=True,
            lag=12,
        )
        ```

        Which returns:

        |         |   United States |   Euro Area |   Japan |
        |:--------|----------------:|------------:|--------:|
        | 2025-10 |          0.0284 |     -0.004  |  0.0282 |
        | 2025-11 |          0.0314 |     -0.0126 |  0.0281 |
        | 2025-12 |          0.0314 |     -0.0204 |  0.0248 |
        | 2026-01 |          0.0308 |     -0.021  |  0.0247 |
        | 2026-02 |          0.0337 |     -0.0303 |  0.0223 |
        | 2026-03 |          0.0429 |      0.0198 |  0.0301 |
        | 2026-04 |          0.057  |      0.0493 |  0.0553 |
        | 2026-05 |          0.0592 |      0.0586 |  0.068  |
        | 2026-06 |          0.0559 |      0.046  |  0.0752 |
        | 2026-07 |          0.0494 |      0.0578 |  0.0773 |
        | 2026-08 |          0.0561 |      0.083  |  0.0759 |
        | 2026-09 |          0.0634 |      0.1113 |  0.0677 |
        """
        check_period_type(period)

        period = validate_period(
            self._default_period(period, "monthly"),
            ["monthly", "quarterly", "yearly"],
            "producer price index",
        )

        producer_price_index = average_monthly_index(
            self._get_monthly_industrial_prices("producer", countries), period
        )

        return finalize_dataset(
            dataset=producer_price_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Producer Price Index",
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
            dropna=True,
        )

    @handle_errors
    def get_export_price_index(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Export Price Index for a variety of countries over time. The index
        measures the average change over time in the prices a country's producers receive
        for the goods they sell abroad, in its own currency. Together with the Import Price
        Index it gives the terms of trade, and it shows how much of a currency move or a
        foreign price shock exporters pass on.

        Each country's index comes monthly from its official publisher:

        - Eurostat for the euro area, the European Union and around 35 European countries:
          the industrial producer prices on the non-domestic market (industry except
          construction, sewerage and waste management), 2021 = 100. For the euro area these
          include sales to other members.
        - The Bureau of Labor Statistics for the United States: the export price index of
          all commodities, 2000 = 100. The BLS API returns about the last three years; with
          a FRED API key the full history comes from FRED.
        - The Bank of Japan for Japan: the export price index in yen, 2020 = 100, from 1960.

        All indices are not seasonally adjusted, so changes are best taken year over year
        (growth=True with lag=12 on monthly data), which returns them as decimals (0.057 for
        5.7%). The base year differs per country, so compare growth rather than levels.
        Quarterly and yearly values are the average of the months, for complete periods only.

        See definition: https://ec.europa.eu/eurostat/cache/metadata/en/sts_inpp_esms.htm

        Also known as: export prices, XPI, trade prices.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Defaults to None, which is quarterly when the Economics class was initialized with
                quarterly=True and monthly otherwise.
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
            pd.DataFrame: A DataFrame containing the Export Price Index per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-10-01', end_date='2026-09-30')

        economics.get_export_price_index(
            countries=['United States', 'Euro Area', 'Japan'],
            period='monthly',
            growth=True,
            lag=12,
        )
        ```

        Which returns:

        |         |   United States |   Euro Area |   Japan |
        |:--------|----------------:|------------:|--------:|
        | 2025-10 |        nan      |     -0.0052 |  0.0224 |
        | 2025-11 |          0.0329 |     -0.0087 |  0.0205 |
        | 2025-12 |          0.0341 |     -0.0147 |  0.04   |
        | 2026-01 |          0.0251 |     -0.0129 |  0.0494 |
        | 2026-02 |          0.038  |     -0.018  |  0.0838 |
        | 2026-03 |          0.0564 |      0.0121 |  0.113  |
        | 2026-04 |          0.0931 |      0.0323 |  0.1809 |
        | 2026-05 |          0.1135 |      0.042  |  0.2057 |
        | 2026-06 |          0.0998 |      0.042  |  0.2126 |
        | 2026-07 |          0.0812 |      0.0525 |  0.1936 |
        | 2026-08 |          0.0928 |      0.0623 |  0.1796 |
        | 2026-09 |          0.1163 |      0.0747 |  0.1786 |
        """
        check_period_type(period)

        period = validate_period(
            self._default_period(period, "monthly"),
            ["monthly", "quarterly", "yearly"],
            "export price index",
        )

        export_price_index = average_monthly_index(
            self._get_monthly_industrial_prices("export", countries), period
        )

        return finalize_dataset(
            dataset=export_price_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Export Price Index",
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
            dropna=True,
        )

    @handle_errors
    def get_import_price_index(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Import Price Index for a variety of countries over time. The index measures
        the average change over time in the prices of the goods a country buys from abroad,
        in its own currency. It shows how foreign price shocks and currency moves reach the
        domestic economy, ahead of the Producer and Consumer Price Indices.

        Each country's index comes monthly from its official publisher:

        - Eurostat for the euro area, the European Union and around 20 European countries:
          the import prices of industrial products (mining, manufacturing and energy
          supply), 2021 = 100. For the euro area these include imports from other members.
        - The Bureau of Labor Statistics for the United States: the import price index of
          all commodities, 2000 = 100. The BLS API returns about the last three years; with
          a FRED API key the full history comes from FRED.
        - The Bank of Japan for Japan: the import price index in yen, 2020 = 100, from 1960.

        All indices are not seasonally adjusted, so changes are best taken year over year
        (growth=True with lag=12 on monthly data), which returns them as decimals (0.057 for
        5.7%). The base year differs per country, so compare growth rather than levels.
        Quarterly and yearly values are the average of the months, for complete periods only.

        See definition: https://ec.europa.eu/eurostat/cache/metadata/en/sts_inpi_esms.htm

        Also known as: import prices, MPI, trade prices.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Defaults to None, which is quarterly when the Economics class was initialized with
                quarterly=True and monthly otherwise.
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
            pd.DataFrame: A DataFrame containing the Import Price Index per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-10-01', end_date='2026-09-30')

        economics.get_import_price_index(
            countries=['United States', 'Euro Area', 'Japan'],
            period='monthly',
            growth=True,
            lag=12,
        )
        ```

        Which returns:

        |         |   United States |   Euro Area |   Japan |
        |:--------|----------------:|------------:|--------:|
        | 2025-10 |        nan      |     -0.0231 | -0.016  |
        | 2025-11 |         -0.0007 |     -0.0291 | -0.017  |
        | 2025-12 |          0      |     -0.036  |  0.0018 |
        | 2026-01 |          0.0021 |     -0.0357 |  0.0072 |
        | 2026-02 |          0.0099 |     -0.033  |  0.0283 |
        | 2026-03 |          0.0226 |      0.015  |  0.0831 |
        | 2026-04 |          0.0452 |      0.0617 |  0.217  |
        | 2026-05 |          0.0688 |      0.073  |  0.2642 |
        | 2026-06 |          0.0675 |      0.0604 |  0.3042 |
        | 2026-07 |          0.0602 |      0.064  |  0.2939 |
        | 2026-08 |          0.0674 |      0.0835 |  0.2506 |
        | 2026-09 |          0.081  |      0.1072 |  0.2447 |
        """
        check_period_type(period)

        period = validate_period(
            self._default_period(period, "monthly"),
            ["monthly", "quarterly", "yearly"],
            "import price index",
        )

        import_price_index = average_monthly_index(
            self._get_monthly_industrial_prices("import", countries), period
        )

        return finalize_dataset(
            dataset=import_price_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Import Price Index",
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
            dropna=True,
        )

    @handle_errors
    def get_house_prices(
        self,
        countries: list[str] | str | None = None,
        quarterly: bool | None = None,
        inflation_adjusted: bool = False,
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
        In most cases, the nominal house price index covers the sales of newly-built
        and existing dwellings, following the recommendations from the RPPI (Residential
        Property Prices Indices) manual.

        By default the data comes from the Global Macro Database (GMDB), annual with
        2015 = 100. Set gmdb_source=False to retrieve it from the OECD instead, which is
        the only source with quarterly data: asking for quarterly data takes it from the
        OECD unless gmdb_source=True is given.

        In the OECD data, the real house price index is given by the ratio of the nominal
        house price index to the consumers' expenditure deflator in each country from the
        OECD national accounts database. Both indices are seasonally adjusted.

        Both are an index based on 2015 = 100.

        See definition: https://data.oecd.org/price/housing-prices.htm

        Also known as: real estate prices, property prices, housing index.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            quarterly (bool | None, optional): Whether to return the quarterly data (True) or the
                annual data (False). Defaults to None, which returns the quarterly data, the
                highest frequency the OECD publishes.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data or the nominal data.
                Defaults to False.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB),
                which is annual. Defaults to None, which uses the OECD for quarterly data and the GMDB for
                annual data; asking for the GMDB together with quarterly data logs a warning.
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
            pd.DataFrame: A DataFrame containing the House Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2023-12-31')

        economics.get_house_prices(
            countries=['Japan', 'Netherlands', 'Ireland'],
            quarterly=False,
            inflation_adjusted=True
        )
        ```

        Which returns:

        |      |   Japan |   Netherlands |   Ireland |
        |:-----|--------:|--------------:|----------:|
        | 2015 | 100     |       100     |   100     |
        | 2016 | 102.645 |       106.044 |   108.503 |
        | 2017 | 104.009 |       113.945 |   120.785 |
        | 2018 | 105.604 |       122.047 |   128.926 |
        | 2019 | 106.087 |       126.665 |   128.744 |
        | 2020 | 107.773 |       135.898 |   130.088 |
        | 2021 | 115.407 |       156.892 |   144.681 |
        | 2022 | 121.069 |       150.691 |   145.71  |
        | 2023 | 119.632 |       145.341 |   141.362 |
        """
        # Quarterly is the highest frequency these indicators are published at.
        quarterly = True if quarterly is None else quarterly
        source_chosen = gmdb_source is not None
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        # The GMDB is yearly, so quarterly data comes from the OECD unless the GMDB was
        # asked for by name.
        if gmdb_source and quarterly:
            if source_chosen:
                logger.warning(
                    "The Global Macro Database publishes house prices yearly; set "
                    "gmdb_source=False for the quarterly OECD data."
                )
            else:
                gmdb_source = False

        if gmdb_source:
            # The real index is the nominal one deflated by consumer prices.
            house_prices = (
                self._get_gmdb_series("rHPI", gmdb_forecasts=gmdb_forecasts)
                if inflation_adjusted
                else gmdb_model.get_house_price_index(
                    gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
                )
            )
        else:
            house_prices = oecd_model.get_house_prices(
                quarterly=quarterly,
                inflation_adjusted=inflation_adjusted,
                start_date=self._start_date,
                end_date=self._end_date,
            )

        return finalize_dataset(
            dataset=house_prices,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="House Prices",
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
    def get_rent_prices(
        self,
        countries: list[str] | str | None = None,
        quarterly: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        The housing rent price index measures the prices paid for renting
        residential properties over time. Together with the house price index
        it is a key input into affordability and house ownership profitability
        measures such as the price to rent ratio.

        This is an index based on 2015 = 100.

        See definition: https://data.oecd.org/price/housing-prices.htm

        Also known as: rental prices, housing costs, rent index.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            quarterly (bool | None, optional): Whether to return the quarterly data (True) or the
                annual data (False). Defaults to None, which returns the quarterly data, the
                highest frequency the OECD publishes.
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
            pd.DataFrame: A DataFrame containing the Rent Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2023-12-31')

        economics.get_rent_prices(
            countries=['Turkey', 'United States', 'United Kingdom'],
            quarterly=False)
        ```

        Which returns:

        |      |   Turkey |   United States |   United Kingdom |
        |:-----|---------:|----------------:|-----------------:|
        | 2015 |  100     |         100     |          100     |
        | 2016 |  108.66  |         103.773 |          101.725 |
        | 2017 |  118.577 |         107.731 |          102.699 |
        | 2018 |  130.041 |         111.627 |          103.174 |
        | 2019 |  143.185 |         115.765 |          103.924 |
        | 2020 |  156.574 |         119.382 |          105.399 |
        | 2021 |  172.625 |         122.062 |          107.148 |
        | 2022 |  221.217 |         129.426 |          110.897 |
        | 2023 |  405.025 |         139.72  |          117.271 |
        """
        # Quarterly is the highest frequency these indicators are published at.
        quarterly = True if quarterly is None else quarterly

        rent_prices = oecd_model.get_rent_prices(
            quarterly=quarterly, start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=rent_prices,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Rent Prices",
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
    def get_commercial_real_estate_prices(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the quarterly growth of commercial real estate prices: the change in the price
        of commercial property (offices, retail, industrial and apartment buildings) on the
        same quarter a year earlier, for around twenty economies.

        This tracks commercial (office, retail, industrial, apartment) property
        prices, as distinct from residential house prices (see `get_house_prices`,
        which tracks a completely different asset class/market). It is a
        transaction-based index rather than the appraisal-smoothed methodology used
        by institutional benchmarks like the NCREIF Property Index -- which is not
        freely available anywhere -- so expect more volatility and less
        autocorrelation than an appraisal-based series would show.

        The prices come from the commercial property price indices
        the Bank for International Settlements collects from national sources, for the euro
        area, the United States (from 1945), Japan, Denmark, the Netherlands, Germany,
        France, Spain and around ten more, taking per country a nominal index of commercial
        property, preferably all commercial property in the whole country. When a FRED API
        key is set (see the `fred_api_key` parameter of the `Economics` class), the United
        States instead comes from FRED's series sourced from the IMF's Financial Soundness
        Indicators.

        See definitions: https://data.bis.org/topics/CPP and
        https://fred.stlouisfed.org/series/COMREPUSQ159N

        Also known as: commercial property price index, CRE price index.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
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
            pd.DataFrame: The quarterly change in commercial real estate prices on a year
            earlier, as a decimal fraction, with a column per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-01-01', end_date="2025-12-31")

        economics.get_commercial_real_estate_prices(countries=['United States', 'Euro Area', 'Japan', 'Germany'])
        ```

        Which returns:

        |        |   United States |   Euro Area |   Japan |   Germany |
        |:-------|----------------:|------------:|--------:|----------:|
        | 2025Q1 |         -0.027  |      0.0232 |  0.0289 |    0.0221 |
        | 2025Q2 |         -0.0494 |      0.0331 |  0.0197 |    0.0266 |
        | 2025Q3 |          0.0209 |      0.0173 |  0.0222 |    0.0264 |
        | 2025Q4 |          0.0109 |      0.0189 |  0.0104 |    0.0207 |
        """
        # The index is turned into its change on the same quarter a year earlier, the
        # measure FRED publishes, so both sources mean the same thing.
        index = bis_model.get_commercial_property_prices(
            buffered_start_date(self._start_date, "monthly"), self._end_date
        )
        commercial_real_estate_prices = change_over_periods(index, 4)

        if self._fred_api_key:
            united_states = fred_model.get_commercial_real_estate_prices(
                self._start_date, self._end_date, self._fred_api_key
            )
            if not united_states.empty:
                united_states.index = united_states.index.asfreq("Q")
                commercial_real_estate_prices = commercial_real_estate_prices.drop(
                    columns="United States", errors="ignore"
                ).join(united_states, how="outer")

        return finalize_dataset(
            dataset=commercial_real_estate_prices,
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
            countries=countries,
            dropna=True,
        )

    @handle_errors
    def get_commodity_prices(
        self,
        commodity: str | list[str] | None = None,
        deflate_with: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Retrieves the World Bank's monthly commodity prices from 1960 (The Pink Sheet): the
        average price of each month in US dollars of around seventy commodities, among which
        crude oil (Brent, WTI, Dubai), natural gas, coal, gold, silver, platinum, copper,
        aluminum, iron ore, nickel, zinc, grains, softs and fertilizers. Futures prices from
        exchanges reach back to 2000 at most, which leaves out the commodity booms and busts
        of the 1970s and 1980s, such as gold's peak of January 1980 and the oil shocks.

        With deflate_with, the prices are in real terms: in the money of the latest month of
        that country's consumer price index up to the end date (see get_consumer_price_index),
        so that the gold price of 1980 can be compared with today's. A month after the latest
        month of the consumer price index, which is published a few weeks late, keeps its
        nominal price, and a month the consumer price index skipped has no real price.

        The formula is as follows:

        - Real Price (t) = Price (t) * CPI (latest month) / CPI (t)

        The units are those of the World Bank: US dollars per barrel ($/bbl) for crude oil,
        per million British thermal units ($/mmbtu) for natural gas, per troy ounce ($/troy oz)
        for gold, silver and platinum, per metric ton ($/mt) for coal, base metals, grains, oils
        and fertilizers, per kilogram ($/kg) for cocoa, coffee, tea, sugar, meat, fruit, cotton
        and rubber, per dry metric ton unit ($/dmtu) for iron ore, per cubic meter for logs and
        sawnwood and cents per sheet for plywood. The Natural gas index is an index (2010=100).

        Also known as: Pink Sheet, commodity price history, long-run gold price, real gold
        price, real oil price, inflation-adjusted commodity prices.

        Args:
            commodity (str | list[str] | None, optional): The commodities, e.g. "Gold" or
                ["Gold", "Crude oil Brent"], as named by the World Bank without commas ("Crude
                oil, Brent" is "Crude oil Brent"), in any casing. Defaults to None, which
                returns all of them.
            deflate_with (str | None, optional): The country whose consumer price index the
                prices are deflated with, e.g. "United States" for US dollar prices. Defaults
                to None, which returns nominal prices.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
            growth (bool, optional): Whether to calculate the growth. Defaults to False.
            lag (int, optional): The lag in months for the growth, e.g. 12 for the change
                over a year. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result.
                Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The monthly prices with a column per commodity.

        Notes:
        - World Bank, "Commodity Markets: The Pink Sheet", monthly prices, licensed under
        CC BY 4.0.
        - The prices are monthly averages: a peak within a month, such as gold's $850 London
        fix of 21 January 1980, is higher than the average of its month.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="1979-01-01", end_date="2026-10-31")

        prices = economics.get_commodity_prices(commodity="Gold", deflate_with="United States")
        prices.loc[["1980-01", "2011-09", "2026-01"]]
        ```

        Which returns:

        |         |    Gold |
        |:--------|--------:|
        | 1980-01 | 2923.32 |
        | 2011-09 | 2631.5  |
        | 2026-01 | 4923.8  |
        """
        prices = world_bank_model.get_commodity_prices()

        if commodity is not None:
            requested = helpers.split_values(commodity)
            by_lowercase = {name.lower(): name for name in prices.columns}
            missing = [name for name in requested if name.lower() not in by_lowercase]
            if missing:
                logger.warning(
                    "The following commodities are not available: %s. Available are %s.",
                    ", ".join(f"'{name}'" for name in missing),
                    ", ".join(prices.columns),
                )
            prices = prices[
                [
                    by_lowercase[name.lower()]
                    for name in requested
                    if name not in missing
                ]
            ]

        if deflate_with is not None:
            consumer_price_index = self.get_consumer_price_index(
                countries=deflate_with, period="monthly", rounding=None
            )
            if consumer_price_index.empty:
                raise ValueError(
                    f"There is no monthly consumer price index for {deflate_with} to deflate "
                    "the prices with."
                )
            prices = helpers.deflate_with_price_index(
                prices, consumer_price_index.iloc[:, 0]
            )

        return finalize_dataset(
            dataset=prices,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Commodity Prices",
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )
