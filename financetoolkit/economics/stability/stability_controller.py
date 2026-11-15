"""Financial Stability Module"""

__docformat__ = "google"

from collections.abc import Callable

import pandas as pd

from financetoolkit import fmp_model, helpers
from financetoolkit.cache import policy_model
from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    buffered_start_date,
    validate_period,
)
from financetoolkit.economics.sources import (
    bis_model,
    cboe_model,
    esrb_model,
    fmp_model as economics_fmp_model,
    frb_model,
    gmdb_model,
    stoxx_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import (
    track_sources,
)
from financetoolkit.utilities.statistics_model import (
    finalize_dataset,
)

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class StabilityEconomics(EconomicsBase):
    """
    Financial stability and risk: sovereign debt, currency and banking crises, credit to GDP and its gap, the debt service ratio, stress test scenarios, implied volatility indices and the market risk premium.
    """

    @handle_errors
    def get_sovereign_debt_crisis(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Sovereign Debt Crisis dummy for a variety of countries over time from the Global
        Macro Database (GMDB). Unlike the other indicators in this module, this is a binary
        (0 = no crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating series rather than a
        continuous economic series: a value of 1 marks a year in which a country was undergoing a
        sovereign debt crisis (e.g. a default or restructuring of government debt), and 0 marks a
        year in which it was not.

        The crisis dating stops short of the present and ends in a different year per country
        (2017 for Argentina, for example), so the years after it are NaN rather than 0.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: sovereign default, debt crisis dummy.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
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
            pd.DataFrame: A DataFrame containing the Sovereign Debt Crisis dummy

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="1999-01-01", end_date="2005-12-31")

        economics.get_sovereign_debt_crisis(countries=["Argentina", "Ecuador", "Greece"])
        ```

        Which returns:

        |      |   Argentina |   Ecuador |   Greece |
        |:-----|------------:|----------:|---------:|
        | 1999 |           0 |         1 |        0 |
        | 2000 |           0 |         0 |        0 |
        | 2001 |           1 |         0 |        0 |
        | 2002 |           0 |         0 |        0 |
        | 2003 |           0 |         0 |        0 |
        | 2004 |           0 |         0 |        0 |
        | 2005 |           1 |         0 |        0 |
        """

        sovereign_debt_crisis = gmdb_model.get_sovereign_debt_crisis(
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
        )

        return finalize_dataset(
            dataset=sovereign_debt_crisis,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Sovereign Debt Crisis",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            rounding=rounding,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_currency_crisis(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Currency Crisis dummy for a variety of countries over time from the Global Macro
        Database (GMDB). Unlike the other indicators in this module, this is a binary (0 = no
        crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating series rather than a continuous
        economic series: a value of 1 marks a year in which a country was undergoing a currency
        crisis (e.g. a sharp, disorderly depreciation or collapse of the exchange rate), and 0
        marks a year in which it was not.

        The crisis dating stops short of the present and ends in a different year per country,
        so the years after it are NaN rather than 0.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: currency collapse, exchange rate crisis dummy.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
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
            pd.DataFrame: A DataFrame containing the Currency Crisis dummy

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='1990-01-01', end_date="2025-12-31")

        economics.get_currency_crisis(countries='Turkey')
        ```

        Which returns:

        |      |   Turkey |
        |:-----|---------:|
        | 2015 |        0 |
        | 2016 |        0 |
        | 2017 |        0 |
        | 2018 |      nan |
        | 2019 |      nan |
        """

        currency_crisis = gmdb_model.get_currency_crisis(
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
        )

        return finalize_dataset(
            dataset=currency_crisis,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Currency Crisis",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            rounding=rounding,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_banking_crisis(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        rounding: int | None = None,
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Banking Crisis dummy for a variety of countries over time from the Global Macro
        Database (GMDB). Unlike the other indicators in this module, this is a binary (0 = no
        crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating series rather than a continuous
        economic series: a value of 1 marks a year in which a country was undergoing a systemic
        banking crisis (e.g. bank runs, large-scale bank failures or government intervention to
        prevent them), and 0 marks a year in which it was not.

        The crisis dating stops short of the present and ends in a different year per country,
        so the years after it are NaN rather than 0.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: banking panic, financial crisis dummy, systemic banking crisis.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
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
            pd.DataFrame: A DataFrame containing the Banking Crisis dummy

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2005-01-01', end_date="2025-12-31")

        economics.get_banking_crisis(countries=['United States', 'United Kingdom'])
        ```

        Which returns:

        |      |   United States |   United Kingdom |
        |:-----|----------------:|-----------------:|
        | 2005 |               0 |                0 |
        | 2006 |               0 |                0 |
        | 2007 |               1 |                0 |
        | 2008 |               0 |                1 |
        | 2009 |               0 |                0 |
        | 2010 |               0 |                0 |
        | 2011 |               0 |                0 |
        | 2012 |               0 |                0 |
        | 2013 |               0 |                0 |
        | 2014 |               0 |                0 |
        | 2015 |               0 |                0 |
        | 2016 |               0 |                0 |
        | 2017 |               0 |                0 |
        | 2018 |               0 |                0 |
        | 2019 |               0 |                0 |
        | 2020 |               0 |                0 |
        | 2021 |               0 |                0 |
        | 2022 |               0 |                0 |
        | 2023 |               0 |                0 |
        | 2024 |               0 |                0 |
        | 2025 |               0 |                0 |
        """

        banking_crisis = gmdb_model.get_banking_crisis(
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
        )

        return finalize_dataset(
            dataset=banking_crisis,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Banking Crisis",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            rounding=rounding,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_credit_to_gdp_gap(
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
        Get the credit-to-GDP gap: how far the ratio of credit to the private non-financial
        sector to GDP lies above or below its long-term trend, in percentage points of GDP.
        The trend is a one-sided Hodrick-Prescott filter with a smoothing parameter of
        400,000, which only uses the quarters up to each point, so the gap is what was
        known at the time.

        A gap that stays above two percentage points of GDP has preceded most banking
        crises of the past decades, often by several years, which is why Basel III uses
        it as the guide for the countercyclical capital buffer: banks hold extra capital
        when the gap is wide (Drehmann and Tsatsaronis, 2014). A negative gap means credit
        grew slower than its trend, as it did in most advanced economies after the 2008
        crisis.

        The credit is all credit to households and non-financial corporations from all
        lenders, banks and non-banks, at market value, as compiled by the Bank for
        International Settlements for around 45 economies, quarterly and back to the
        1950s for some.

        See definition: https://data.bis.org/topics/CREDIT_GAPS and
        https://www.bis.org/publ/qtrpdf/r_qt1403g.htm

        Also known as: credit gap, Basel gap, credit-to-GDP deviation from trend.

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
            pd.DataFrame: The quarterly credit-to-GDP gap as a decimal fraction of GDP
            (-0.05 for five percentage points below the trend), with a column per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-01-01', end_date="2025-12-31")

        economics.get_credit_to_gdp_gap(countries=['United States', 'Euro Area', 'China', 'Japan'])
        ```

        Which returns:

        |        |   United States |   Euro Area |   China |   Japan |
        |:-------|----------------:|------------:|--------:|--------:|
        | 2025Q1 |         -0.126  |     -0.1909 | -0.0443 |  0.0461 |
        | 2025Q2 |         -0.1236 |     -0.1894 | -0.053  |  0.0335 |
        | 2025Q3 |         -0.12   |     -0.1873 | -0.0639 |  0.0418 |
        | 2025Q4 |         -0.1159 |     -0.1815 | -0.0769 |  0.0587 |
        """
        gap = bis_model.get_credit_statistics(
            "WS_CREDIT_GAP",
            # Credit to the private non-financial sector from all lenders, the gap (C)
            # rather than the ratio (A) or its trend (B).
            "Q..P.A.C",
            buffered_start_date(self._start_date, "monthly"),
            self._end_date,
        )

        return finalize_dataset(
            dataset=gap,
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

    # The borrowing sectors of the BIS debt service ratios.
    DEBT_SERVICE_BORROWERS = {"private": "P", "households": "H", "corporations": "N"}

    @handle_errors
    def get_debt_service_ratio(
        self,
        countries: list[str] | str | None = None,
        borrowers: str = "private",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the debt service ratio: the share of income that goes to interest payments and
        repayments of debt. It captures both how much is borrowed and at what rate, so it
        rises when rates go up even if the debt itself does not, and it is a leading
        indicator of financial stress: a high debt service ratio held back spending and
        preceded the downturns after several credit booms (Drehmann and Juselius, 2012).

        The ratio is computed by the Bank for International Settlements for the private
        non-financial sector of around 30 economies, and for households and non-financial
        corporations separately for about half of them, quarterly from 1999.
        It is the debt service (interest plus amortisation, assuming the average remaining
        maturity of the debt) divided by income: disposable income plus interest paid for
        households, and gross operating surplus plus interest paid for corporations. No API
        key is needed.

        See definition: https://data.bis.org/topics/DSR and
        https://www.bis.org/publ/qtrpdf/r_qt1509h.htm

        Also known as: DSR, debt service burden, debt service-to-income ratio.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            borrowers (str, optional): The borrowing sector: "private" (households and
                non-financial corporations together), "households" or "corporations"
                (non-financial corporations). Defaults to "private", the sector with the
                widest coverage.
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
            pd.DataFrame: The quarterly debt service ratio as a decimal fraction of income,
            with a column per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-01-01', end_date="2025-12-31")

        economics.get_debt_service_ratio(countries=['United States', 'Germany', 'China', 'Japan'])
        ```

        Which returns:

        |        |   United States |   Germany |   China |   Japan |
        |:-------|----------------:|----------:|--------:|--------:|
        | 2025Q1 |           0.14  |     0.123 |   0.189 |   0.15  |
        | 2025Q2 |           0.14  |     0.123 |   0.189 |   0.15  |
        | 2025Q3 |           0.139 |     0.123 |   0.188 |   0.152 |
        | 2025Q4 |           0.139 |     0.122 |   0.188 |   0.154 |
        """
        if borrowers not in self.DEBT_SERVICE_BORROWERS:
            raise ValueError(
                f"The borrowers must be one of {', '.join(map(repr, self.DEBT_SERVICE_BORROWERS))}, not {borrowers!r}."  # noqa: E501
            )

        debt_service_ratio = bis_model.get_credit_statistics(
            "WS_DSR",
            f"Q..{self.DEBT_SERVICE_BORROWERS[borrowers]}",
            buffered_start_date(self._start_date, "monthly"),
            self._end_date,
        )

        return finalize_dataset(
            dataset=debt_service_ratio,
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

    # The series keys of the BIS total credit statistics per borrowing sector, all from all
    # lenders and as a percentage of GDP. Government debt is taken at nominal value, the
    # valuation the BIS publishes for the most countries; the other sectors at market value.
    CREDIT_BORROWERS = {
        "non_financial": "Q..C.A.M.770.A",
        "private": "Q..P.A.M.770.A",
        "households": "Q..H.A.M.770.A",
        "corporations": "Q..N.A.M.770.A",
        "government": "Q..G.A.N.770.A",
    }

    @handle_errors
    def get_credit_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        borrowers: str = "private",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the total credit to a non-financial sector as a share of GDP: all loans and
        debt securities owed by households, non-financial corporations or the government,
        whoever the lender is (banks, other financial institutions, foreign lenders or
        bond holders). It measures how leveraged an economy is and where the debt sits,
        such as the high corporate debt in China or the high household debt in Australia
        and Canada.

        Unlike bank lending alone, it also counts market and cross-border funding, so it
        is comparable across countries where companies borrow from banks (the euro area,
        Japan) and where they issue bonds (the United States). The data is compiled by the
        Bank for International Settlements for around 45 economies, quarterly and back to
        the 1940s for some, from national financial accounts.

        See definition: https://data.bis.org/topics/TOTAL_CREDIT and
        https://www.bis.org/publ/qtrpdf/r_qt1303h.htm

        Also known as: total credit to GDP, debt to GDP by sector, private debt to GDP,
        leverage ratio of the economy.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            borrowers (str, optional): The borrowing sector: "private" (households and
                non-financial corporations), "households", "corporations" (non-financial
                corporations), "government" (at nominal value) or "non_financial" (all
                three together). Defaults to "private", the measure behind the
                credit-to-GDP gap.
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
            pd.DataFrame: The quarterly credit as a decimal fraction of GDP (1.5 for 150% of
            GDP), with a column per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-01-01', end_date="2025-12-31")

        economics.get_credit_to_gdp_ratio(countries=['United States', 'Euro Area', 'China', 'Japan'])
        ```

        Which returns:

        |        |   United States |   Euro Area |   China |   Japan |
        |:-------|----------------:|------------:|--------:|--------:|
        | 2025Q1 |           1.416 |       1.553 |   2.019 |   1.71  |
        | 2025Q2 |           1.41  |       1.545 |   2.018 |   1.701 |
        | 2025Q3 |           1.406 |       1.537 |   2.015 |   1.713 |
        | 2025Q4 |           1.402 |       1.533 |   2.008 |   1.734 |
        """
        if borrowers not in self.CREDIT_BORROWERS:
            raise ValueError(
                f"The borrowers must be one of {', '.join(map(repr, self.CREDIT_BORROWERS))}, not {borrowers!r}."  # noqa: E501
            )

        credit = bis_model.get_credit_statistics(
            "WS_TC",
            self.CREDIT_BORROWERS[borrowers],
            buffered_start_date(self._start_date, "monthly"),
            self._end_date,
        )

        return finalize_dataset(
            dataset=credit,
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
    def get_implied_volatility(
        self,
        markets: list[str] | str | None = None,
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
        Get the implied volatility the option markets price, by market and time to expiry:
        the volatility indices of Cboe, STOXX and ICE. Implied volatility is the market's
        expectation of how much prices will move, and its term structure (how it rises or
        falls with the time to expiry) is the market-consistent target for calibrating the
        volatility of equity, rate and commodity models.

        The markets and their maturities:
        - "US Equity": S&P 500 options, the Cboe VIX family: 9 days (VIX9D, from 2011), 1
          month (VIX, from 1990), 3 months (VIX3M), 6 months (VIX6M) and 1 year (VIX1Y, from
          2007).
        - "Euro Area Equity": EURO STOXX 50 options, the VSTOXX sub-indices from 1 to 24
          months, from 1999.
        - "Emerging Markets Equity": options on the MSCI Emerging Markets ETF (Cboe VXEEM),
          1 month, from 2011.
        - "US Equity Volatility": the volatility of the VIX itself (Cboe VVIX), 1 month, from
          2006.
        - "US Treasury Bonds": options on the 20+ year Treasury bond ETF (Cboe VXTLT), the
          price volatility of long Treasuries, 1 month, from 2004.
        - "Crude Oil" and "Gold": options on the oil and gold ETFs (Cboe OVX and GVZ), 1
          month, from 2009.
        - "US Treasury Rates": the ICE BofA MOVE index, the normal (basis point) volatility of
          Treasury yields implied by 1-month options on 2, 5, 10 and 30-year Treasuries, from
          2002. Requires a FinancialModelingPrep API key (the api_key of the Economics class).
        - "Japan Government Bonds": the S&P/JPX JGB VIX, from options on JGB futures, 1
          month, from 2015. Requires a FinancialModelingPrep API key.

        The volatility is returned as a decimal fraction per annum (0.155 for a VIX of 15.5),
        and the MOVE as a decimal yield change per annum (0.0105 for a MOVE of 105 basis
        points), with one column per market and maturity. Weekly and monthly periods take the
        value on the last trading day of each period. The Cboe SKEW index is not included
        since it is not a volatility.

        See definition: https://www.cboe.com/tradable_products/vix/

        Also known as: VIX, VSTOXX, MOVE index, volatility term structure, implied
        volatility index, fear index.

        Args:
            markets (list[str] | str | None, optional): The markets to retrieve, from
                "US Equity", "Euro Area Equity", "Emerging Markets Equity", "US Equity
                Volatility", "US Treasury Bonds", "Crude Oil", "Gold", "US Treasury Rates" and
                "Japan Government Bonds". Defaults to None, which retrieves every market, with
                "US Treasury Rates" (the MOVE) and "Japan Government Bonds" (the JGB VIX) only
                when an API key is set.
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
            pd.DataFrame: The implied volatility as decimals, indexed by date with a column per
            market and maturity.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        implied_volatility = economics.get_implied_volatility(
            markets=['US Equity', 'Euro Area Equity'], period='monthly'
        )

        implied_volatility['US Equity']
        ```

        Which returns:

        |         |     9D |     1M |     3M |     6M |     1Y |
        |:--------|-------:|-------:|-------:|-------:|-------:|
        | 2026-04 | 0.1437 | 0.1689 | 0.2008 | 0.2261 | 0.2365 |
        | 2026-05 | 0.1259 | 0.1532 | 0.1866 | 0.216  | 0.2306 |
        | 2026-06 | 0.1373 | 0.1645 | 0.19   | 0.215  | 0.2303 |
        | 2026-07 | 0.1305 | 0.1599 | 0.1902 | 0.2134 | 0.2294 |
        | 2026-08 | 0.1234 | 0.1492 | 0.1753 | 0.2017 | 0.2187 |
        | 2026-09 | 0.142  | 0.1634 | 0.1837 | 0.2035 | 0.218  |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "implied volatility"
        )

        def cboe(maturities: dict[str, str]) -> Callable[[str], pd.DataFrame]:
            return lambda start: pd.DataFrame(
                {
                    maturity: cboe_model.get_index(symbol)
                    for maturity, symbol in maturities.items()
                }
            )

        def fmp(symbol: str, scale: float) -> Callable[[str], pd.DataFrame]:
            return lambda start: (
                economics_fmp_model.get_index_history(
                    symbol, self._api_key, start, self._end_date
                )
                * scale
            ).to_frame("1M")

        sources: dict[str, Callable[[str], pd.DataFrame]] = {
            "US Equity": cboe(
                {
                    "9D": "VIX9D",
                    "1M": "VIX",
                    "3M": "VIX3M",
                    "6M": "VIX6M",
                    "1Y": "VIX1Y",
                }
            ),
            "Euro Area Equity": lambda start: stoxx_model.get_vstoxx_term_structure(),
            "Emerging Markets Equity": cboe({"1M": "VXEEM"}),
            "US Equity Volatility": cboe({"1M": "VVIX"}),
            "US Treasury Bonds": cboe({"1M": "VXTLT"}),
            "Crude Oil": cboe({"1M": "OVX"}),
            "Gold": cboe({"1M": "GVZ"}),
        }
        keyed: dict[str, Callable[[str], pd.DataFrame]] = {
            # The MOVE is in basis points and the JGB VIX in percent; both are divided by
            # 100 below with the other indices, so the MOVE is first divided by another 100.
            "US Treasury Rates": fmp("^MOVE", 0.01),
            "Japan Government Bonds": fmp("^SPJGBV", 1),
        }

        if markets is None and self._api_key:
            sources.update(keyed)
        elif markets is not None:
            requested = [markets] if isinstance(markets, str) else list(markets)
            if missing_key := [
                market for market in requested if market in keyed and not self._api_key
            ]:
                logger.warning(
                    "%s requires a FinancialModelingPrep API key, passed with the api_key "
                    "parameter of the Economics class.",
                    ", ".join(missing_key),
                )
            sources.update(
                {
                    market: source
                    for market, source in keyed.items()
                    if self._api_key and market in requested
                }
            )
            sources.update(
                {
                    market: lambda start: pd.DataFrame()
                    for market in keyed
                    if market in requested and market not in sources
                }
            )

        implied_volatility = self._collect_country_curves(
            sources, markets, period, "implied volatility"
        )

        if not implied_volatility.empty:
            # The indices are quoted in percent.
            implied_volatility = implied_volatility / 100
            implied_volatility.columns.names = ["Market", "Maturity"]

        return finalize_dataset(
            dataset=implied_volatility,
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
    def get_stress_test_scenario(
        self,
        scenario: str = "adverse",
        authority: str = "federal_reserve",
        countries: list[str] | str | None = None,
        year: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the scenarios regulators publish for their stress tests: documented paths of
        the economy and financial markets under a baseline and an adverse scenario, which
        serve as calibration and validation targets for scenario generators and as the
        stresses banks and insurers are tested against.

        Two authorities are available:
        - "federal_reserve": the Federal Reserve's supervisory stress test, quarterly over
          13 quarters. For the United States: real and nominal GDP and disposable income
          growth, unemployment, CPI inflation, the 3-month, 5-year and 10-year Treasury
          rates, the BBB corporate yield, the mortgage and prime rate, the Dow Jones Total
          Stock Market Index, house and commercial real estate price indices and the VIX;
          for the euro area, developing Asia, Japan and the United Kingdom: GDP growth,
          inflation and the dollar exchange rate. scenario="historic" returns the history
          of the same variables from 1976. year selects the year of the test (published in
          February), from 2025; by default the latest.
        - "esrb": the macro-financial scenario the European Systemic Risk Board designs for
          the EU-wide bank stress test of the EBA, yearly over three years: GDP growth,
          inflation, unemployment, residential and commercial real estate prices and
          long-term rates for every EU member and the main other economies, stock prices by
          region, commodity prices, iTraxx credit spreads and euro exchange rates.
          scenario="historic" returns the starting point. Always the latest exercise.

        The Federal Reserve's growth and inflation variables (GDP, disposable income and
        CPI, and the international GDP growth and inflation) are annualized
        quarter-on-quarter rates, as the Federal Reserve publishes them, so a "Real GDP
        growth" of -0.054 is a quarterly contraction of about 1.4%.

        Growth rates, inflation, interest rates, spreads and the VIX are decimal fractions
        (0.054 for 5.4%); price index levels, the stock market index and exchange rates are
        returned as published. The columns are per country (or region, index, commodity or
        currency pair) and variable.

        See definition: https://www.federalreserve.gov/supervisionreg/dfast-archive.htm

        Also known as: supervisory scenarios, CCAR, DFAST, EBA stress test, adverse
        scenario, severely adverse scenario.

        Args:
            scenario (str, optional): "baseline", "adverse" (the Federal Reserve's severely
                adverse scenario) or "historic". Defaults to "adverse".
            authority (str, optional): "federal_reserve" or "esrb". Defaults to
                "federal_reserve".
            countries (list[str] | str | None, optional): The countries (or regions) to
                include. Defaults to None, which includes all of them.
            year (int | None, optional): The year of the Federal Reserve's stress test, from
                2025. Defaults to None, which takes the latest.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The scenario, indexed by quarter or year with a column per country
            and variable.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(end_date="2025-12-31")

        stress_test = economics.get_stress_test_scenario(countries='United States')

        stress_test['United States'][
            ['Real GDP growth', 'Unemployment rate', '10-year Treasury yield', 'BBB corporate yield']
        ].head()
        ```

        Which returns:

        |        |   Real GDP growth |   Unemployment rate |   10-year Treasury yield |   BBB corporate yield |
        |:-------|------------------:|--------------------:|-------------------------:|----------------------:|
        | 2026Q1 |            -0.054 |               0.059 |                    0.031 |                 0.075 |
        | 2026Q2 |            -0.049 |               0.072 |                    0.027 |                 0.082 |
        | 2026Q3 |            -0.038 |               0.082 |                    0.024 |                 0.081 |
        | 2026Q4 |            -0.027 |               0.09  |                    0.023 |                 0.079 |
        | 2027Q1 |            -0.014 |               0.095 |                    0.023 |                 0.075 |
        """
        if scenario not in ("baseline", "adverse", "historic"):
            raise ValueError(
                f"The scenario must be 'baseline', 'adverse' or 'historic', not {scenario!r}."
            )

        if authority == "federal_reserve":
            stress_test = frb_model.get_supervisory_scenario(scenario, year)
        elif authority == "esrb":
            if year is not None:
                logger.info(
                    "The ESRB scenario is always the latest exercise, so year is not used."
                )
            stress_test = esrb_model.get_macro_financial_scenario(scenario)
        else:
            raise ValueError(
                f"The authority must be 'federal_reserve' or 'esrb', not {authority!r}."
            )

        if countries is not None and not stress_test.empty:
            requested = [countries] if isinstance(countries, str) else list(countries)
            available = stress_test.columns.get_level_values(0)

            if unavailable := [
                country for country in requested if country not in available
            ]:
                logger.warning(
                    "The %s scenario has no %s. It covers %s.",
                    authority,
                    ", ".join(unavailable),
                    ", ".join(dict.fromkeys(available)),
                )

            stress_test = stress_test.loc[:, available.isin(requested)]

        # A scenario runs into the future, so only the history is cut to the dates.
        historic = scenario == "historic"

        return finalize_dataset(
            dataset=stress_test,
            start_date=self._start_date if historic else None,
            end_date=self._end_date if historic else None,
            default_rounding=self._rounding,
            rounding=rounding,
            axis="rows",
            row_slice=historic,
            apply_slice=historic,
            dropna=True,
        )

    @handle_errors
    def get_market_risk_premium(
        self, show_columns: str | list[str] | None = None
    ) -> pd.DataFrame:
        """
        Returns the market risk premium per country: the extra return investors demand
        for holding equities over a risk-free investment. It consists of the country risk
        premium, which compensates for the additional risk of the country itself, and the
        total equity risk premium, which adds the premium of a mature market. The market
        risk premium is a key input of the Capital Asset Pricing Model (CAPM) and with
        that of the cost of equity and the WACC.

        In the CAPM the cost of equity is the risk-free rate plus beta times the total
        equity risk premium. Use the premium of the country where the company earns its
        revenue rather than where it is listed (weighted by revenue for a company active in
        several countries), and keep the risk-free rate consistent with the one the premium
        was estimated against.

        The same data is available as Toolkit.get_market_risk_premium; both share one
        cached copy. This requires a FinancialModelingPrep API key, passed via api_key
        when initializing.

        Also known as: equity risk premium, country risk premium, MRP.

        Args:
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The country and total equity risk premium per country, as decimals
            (0.0446 for 4.46%).

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(api_key="FINANCIAL_MODELING_PREP_KEY", end_date="2025-12-31")

        market_risk_premium = economics.get_market_risk_premium()

        market_risk_premium.loc[["United States", "Germany", "Japan", "Brazil", "India"]]
        ```

        Which returns:

        | Country       | Continent     |   Country Risk Premium |   Total Equity Risk Premium |
        |:--------------|:--------------|-----------------------:|----------------------------:|
        | United States | North America |                 0.0023 |                      0.0446 |
        | Germany       | Europe        |                 0      |                      0.0423 |
        | Japan         | Asia          |                 0.0091 |                      0.0514 |
        | Brazil        | South America |                 0.0324 |                      0.0747 |
        | India         | Asia          |                 0.0285 |                      0.0708 |
        """
        helpers.require_api_key(self._api_key, "the market risk premium")

        # Published per country rather than per ticker, so it is one cache entry, the
        # same one Toolkit.get_market_risk_premium uses.
        if self._cache is not None:
            cached_premium = self._cache.get(
                source=policy_model.FINANCIAL_MODELING_PREP,
                dataset="market_risk_premium",
                entity=policy_model.MARKET_RISK_PREMIUM_ENTITY,
            )
            if cached_premium is not None:
                return cached_premium

        market_risk_premium = fmp_model.get_market_risk_premium(api_key=self._api_key)

        if self._cache is not None and not market_risk_premium.empty:
            self._cache.set(
                source=policy_model.FINANCIAL_MODELING_PREP,
                dataset="market_risk_premium",
                entity=policy_model.MARKET_RISK_PREMIUM_ENTITY,
                data=market_risk_premium,
            )

        return market_risk_premium
