"""Government Finances Module"""

__docformat__ = "google"

from financetoolkit import helpers
from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    validate_period,
)
from financetoolkit.economics.sources import imf_model, oecd_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import (
    track_sources,
)
from financetoolkit.utilities.statistics_model import (
    finalize_dataset,
)

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class GovernmentEconomics(EconomicsBase):
    """
    Government finances: debt, revenue, tax revenue, expenditure (also by function), the deficit, interest expenditure and trust in government.
    """

    @staticmethod
    def _government_variable(variable: str, level: str) -> str:
        """
        Returns the Global Macro Database variable of a government series at a level of
        government: "govdebt" is the consolidated debt, "gen_govdebt" the general government
        debt and "cgovdebt" the central government debt.

        Args:
            variable (str): The consolidated variable, e.g. "govdebt_GDP".
            level (str): "consolidated", "general" or "central".

        Returns:
            str: The variable at that level.

        Raises:
            ValueError: When the level is not one of the three.
        """
        prefixes = {"consolidated": "", "general": "gen_", "central": "c"}

        if level not in prefixes:
            raise ValueError(
                f"The level must be one of {', '.join(map(repr, prefixes))}, not {level!r}."
            )

        return f"{prefixes[level]}{variable}"

    @handle_errors
    def get_government_debt(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Debt for a variety of countries over time from the Global Macro Database (GMDB).
        Government Debt is the total amount of money that a government owes to creditors.

        The stock is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: national debt, sovereign debt.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Debt

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_debt(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |        Canada |        Mexico |
        |:-----|----------------:|--------------:|--------------:|
        | 2015 |     1.92907e+07 |   1.83683e+06 |   9.80508e+06 |
        | 2016 |     2.01975e+07 |   1.87318e+06 |   1.14065e+07 |
        | 2017 |     2.08678e+07 |   1.94774e+06 |   1.184e+07   |
        | 2018 |     2.22374e+07 |   2.02901e+06 |   1.26199e+07 |
        | 2019 |     2.34322e+07 |   2.08475e+06 |   1.30327e+07 |
        | 2020 |     2.83333e+07 |   2.62165e+06 |   1.40652e+07 |
        | 2021 |     2.96684e+07 |   2.85548e+06 |   1.51339e+07 |
        | 2022 |     3.10429e+07 |   2.96557e+06 |   1.58852e+07 |
        | 2023 |     3.33635e+07 |   3.11556e+06 |   1.68118e+07 |
        | 2024 |     3.58226e+07 |   3.41999e+06 |   1.97928e+07 |
        | 2025 |     3.811e+07   | nan           | nan           |
        """

        government_debt = self._get_gmdb_series(
            self._government_variable("govdebt", level), gmdb_forecasts=gmdb_forecasts
        )

        return finalize_dataset(
            dataset=government_debt,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Debt",
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
    def get_government_debt_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Debt to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Debt to GDP Ratio is the ratio of the total amount of money that a government owes to creditors
        to the Gross Domestic Product (GDP).

        Debt to GDP is the standard measure of a government's indebtedness, used by rating agencies, the IMF and the
        European Union, whose fiscal rules refer to 60% of GDP. Whether a ratio is sustainable depends less on its level
        than on the gap between the interest rate on the debt and the growth rate of nominal GDP, and on the primary
        balance. Compare countries on the same level of government (the level parameter), since central government debt
        can differ substantially from general government debt.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (1.3173 for 131.73% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Government Debt to GDP Ratio = Gross Government Debt / Nominal GDP,
          using the level of government selected with `level` ("consolidated" by default)

        Also known as: debt-to-GDP ratio, fiscal sustainability.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Debt to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_debt_to_gdp_ratio(countries=['Netherlands', 'Germany', 'China'])
        ```

        Which returns:

        |      |   Netherlands |   Germany |    China |
        |:-----|--------------:|----------:|---------:|
        | 2015 |        0.6382 |    0.7118 |   0.4078 |
        | 2016 |        0.6087 |    0.683  |   0.5003 |
        | 2017 |        0.5597 |    0.6399 |   0.5389 |
        | 2018 |        0.5155 |    0.6076 |   0.5588 |
        | 2019 |        0.4765 |    0.5868 |   0.5984 |
        | 2020 |        0.5341 |    0.6804 |   0.7013 |
        | 2021 |        0.505  |    0.6794 |   0.7213 |
        | 2022 |        0.4841 |    0.644  |   0.7729 |
        | 2023 |        0.4585 |    0.6235 |   0.8408 |
        | 2024 |        0.4378 |    0.6223 |   0.9042 |
        | 2025 |        0.4439 |    0.635  | nan      |
        """

        government_debt_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govdebt_GDP", level),
            in_percent=True,
            gmdb_forecasts=gmdb_forecasts,
        )

        return finalize_dataset(
            dataset=government_debt_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Debt to GDP Ratio",
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
    def get_government_revenue(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Revenue for a variety of countries over time from the Global Macro Database (GMDB).
        Government Revenue is the total amount of money that a government collects from taxes and other sources.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: government income, public revenue.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Revenue

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2019-01-01', end_date="2025-12-31")

        economics.get_government_revenue(countries=['United Kingdom', 'Canada', 'Japan'])
        ```

        Which returns:

        |      |   United Kingdom |           Canada |       Japan |
        |:-----|-----------------:|-----------------:|------------:|
        | 2019 | 856511           | 938667           | 1.97133e+08 |
        | 2020 | 825025           | 919593           | 1.97357e+08 |
        | 2021 | 917948           |      1.07553e+06 | 2.07143e+08 |
        | 2022 |      1.04634e+06 |      1.17518e+06 | 2.16838e+08 |
        | 2023 |      1.10774e+06 |      1.23487e+06 | 2.24713e+08 |
        | 2024 |      1.1523e+06  |      1.31248e+06 | 2.33044e+08 |
        | 2025 |      1.23387e+06 |      1.36827e+06 | 2.44853e+08 |
        """

        government_revenue = self._get_gmdb_series(
            self._government_variable("govrev", level), gmdb_forecasts=gmdb_forecasts
        )

        return finalize_dataset(
            dataset=government_revenue,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Revenue",
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
    def get_government_revenue_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Revenue to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Revenue to GDP Ratio is the ratio of the total amount of money that a government collects from taxes
        and other sources to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.3104 for 31.04% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Government Revenue to GDP Ratio = Government Revenue / Nominal GDP,
          using the level of government selected with `level` ("consolidated" by default)

        Also known as: revenue to GDP ratio.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Revenue to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_revenue_to_gdp_ratio(countries=['United States', 'Canada', 'Russian Federation'])
        ```

        Which returns:

        |      |   United States |   Canada |   Russian Federation |
        |:-----|----------------:|---------:|---------------------:|
        | 2015 |          0.3324 |   0.3996 |               0.3189 |
        | 2016 |          0.3275 |   0.403  |               0.3292 |
        | 2017 |          0.3343 |   0.4034 |               0.3336 |
        | 2018 |          0.3169 |   0.4102 |               0.3554 |
        | 2019 |          0.3167 |   0.4057 |               0.3568 |
        | 2020 |          0.3239 |   0.4141 |               0.3516 |
        | 2021 |          0.3351 |   0.4241 |               0.3572 |
        | 2022 |          0.348  |   0.4103 |               0.338  |
        | 2023 |          0.3146 |   0.4165 |               0.339  |
        | 2024 |          0.3174 |   0.4222 |               0.3506 |
        | 2025 |          0.3253 |   0.4219 |               0.3523 |
        """

        government_revenue_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govrev_GDP", level),
            in_percent=True,
            gmdb_forecasts=gmdb_forecasts,
        )

        return finalize_dataset(
            dataset=government_revenue_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Revenue to GDP Ratio",
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
    def get_government_tax_revenue(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Tax Revenue for a variety of countries over time from the Global Macro Database (GMDB).
        Government Tax Revenue is the total amount of money that a government collects from taxes.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: tax revenue, fiscal revenue.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Tax Revenue

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_tax_revenue(countries=['Kenya', 'Nigeria', 'South Africa'] )
        ```

        Which returns:

        |      |         Kenya |   South Africa |
        |:-----|--------------:|---------------:|
        | 2015 |   1.0216e+06  |    1.16848e+06 |
        | 2016 |   1.14692e+06 |    1.24935e+06 |
        | 2017 |   1.28657e+06 |    1.33015e+06 |
        | 2018 |   1.35099e+06 |    1.41653e+06 |
        | 2019 |   1.55782e+06 |    1.48503e+06 |
        | 2020 |   1.54288e+06 |    1.38495e+06 |
        | 2021 |   1.64048e+06 |    1.70199e+06 |
        | 2022 |   1.99797e+06 |    1.83491e+06 |
        | 2023 |   2.11082e+06 |    1.89676e+06 |
        | 2024 | nan           |    2.02766e+06 |
        | 2025 | nan           |    2.03308e+06 |
        """

        government_tax_revenue = self._get_gmdb_series(
            self._government_variable("govtax", level), gmdb_forecasts=gmdb_forecasts
        )

        return finalize_dataset(
            dataset=government_tax_revenue,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Tax Revenue",
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
    def get_government_tax_revenue_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Tax Revenue to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Tax Revenue to GDP Ratio is the ratio of the total amount of money that a government collects from taxes
        to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.1022 for 10.22% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Government Tax Revenue to GDP Ratio = Government Tax Revenue / Nominal GDP,
          using the level of government selected with `level` ("consolidated" by default)

        Also known as: tax burden, tax to GDP ratio.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Tax Revenue to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_tax_revenue_to_gdp_ratio(
            countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |   Canada |   Mexico |
        |:-----|----------------:|---------:|---------:|
        | 2015 |          0.1994 |   0.2815 |   0.1323 |
        | 2016 |          0.1958 |   0.2855 |   0.1401 |
        | 2017 |          0.2031 |   0.2863 |   0.1358 |
        | 2018 |          0.1874 |   0.2903 |   0.136  |
        | 2019 |          0.1887 |   0.2857 |   0.1369 |
        | 2020 |          0.1928 |   0.2976 |   0.1484 |
        | 2021 |          0.2073 |   0.3021 |   0.1443 |
        | 2022 |          0.2203 |   0.2938 |   0.1449 |
        | 2023 |          0.1932 |   0.2958 |   0.1538 |
        | 2024 |          0.1942 |   0.2959 | nan      |
        | 2025 |          0.202  |   0.2925 | nan      |
        """

        government_tax_revenue_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govtax_GDP", level),
            in_percent=True,
            gmdb_forecasts=gmdb_forecasts,
        )

        return finalize_dataset(
            dataset=government_tax_revenue_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Tax Revenue to GDP Ratio",
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
    def get_government_expenditure(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Expenditure for a variety of countries over time from the Global Macro Database (GMDB).
        Government Expenditure is the total spending of the government: goods and services and the wages of its
        employees, but also social benefits and other transfers, subsidies and interest on its debt.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: government spending, public expenditure.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Expenditure

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_expenditure(countries=['Japan', 'China', 'India'])
        ```

        Which returns:

        |      |       Japan |       China |       India |
        |:-----|------------:|------------:|------------:|
        | 2015 | 2.06691e+08 | 2.1786e+07  | 3.65765e+07 |
        | 2016 | 2.08867e+08 | 2.41238e+07 | 4.12501e+07 |
        | 2017 | 2.09092e+08 | 2.71281e+07 | 4.4235e+07  |
        | 2018 | 2.10865e+08 | 3.05534e+07 | 4.91601e+07 |
        | 2019 | 2.14548e+08 | 3.38067e+07 | 5.35377e+07 |
        | 2020 | 2.47428e+08 | 3.6053e+07  | 6.12564e+07 |
        | 2021 | 2.43415e+08 | 3.74577e+07 | 7.03345e+07 |
        | 2022 | 2.41313e+08 | 4.02806e+07 | 7.82555e+07 |
        | 2023 | 2.39074e+08 | 4.23663e+07 | 8.60877e+07 |
        | 2024 | 2.43693e+08 | 4.44027e+07 | 9.31256e+07 |
        | 2025 | 2.51507e+08 | 4.44552e+07 | 9.84739e+07 |
        """

        government_expenditure = self._get_gmdb_series(
            self._government_variable("govexp", level), gmdb_forecasts=gmdb_forecasts
        )

        return finalize_dataset(
            dataset=government_expenditure,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Expenditure",
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
    def get_government_expenditure_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Expenditure to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Expenditure to GDP Ratio is the total spending of the government (goods and services and the wages
        of its employees, but also social benefits and other transfers, subsidies and interest on its debt) as a share of
        the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.3708 for 37.08% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Government Expenditure to GDP Ratio = Government Expenditure / Nominal GDP,
          using the level of government selected with `level` ("consolidated" by default)

        Also known as: government spending to GDP.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Expenditure to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_expenditure_to_gdp_ratio(
            countries=['United States', 'Japan', 'Netherlands'])
        ```

        Which returns:

        |      |   United States |   Japan |   Netherlands |
        |:-----|----------------:|--------:|--------------:|
        | 2015 |          0.3791 |  0.3766 |        0.4525 |
        | 2016 |          0.3818 |  0.3757 |        0.4391 |
        | 2017 |          0.3788 |  0.37   |        0.4275 |
        | 2018 |          0.3785 |  0.3705 |        0.4244 |
        | 2019 |          0.3836 |  0.3752 |        0.421  |
        | 2020 |          0.4736 |  0.4466 |        0.4781 |
        | 2021 |          0.4542 |  0.4244 |        0.4591 |
        | 2022 |          0.3849 |  0.4126 |        0.4328 |
        | 2023 |          0.3931 |  0.3881 |        0.4401 |
        | 2024 |          0.3966 |  0.3839 |        0.4429 |
        | 2025 |          0.3981 |  0.3791 |        0.4486 |
        """

        government_expenditure_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govexp_GDP", level),
            in_percent=True,
            gmdb_forecasts=gmdb_forecasts,
        )

        return finalize_dataset(
            dataset=government_expenditure_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Expenditure to GDP Ratio",
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
    def get_government_deficit(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Deficit for a variety of countries over time from the Global Macro Database (GMDB).
        Government Deficit is the total amount of money that a government spends more than it collects from taxes
        and other sources. A government deficit is usually financed by borrowing money.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The series is signed as a fiscal balance rather than as a deficit, in millions of
        national currency: a negative value is a deficit (spending exceeding revenue) and a
        positive value is a surplus.

        Also known as: budget deficit, fiscal deficit.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Deficit

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_deficit(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |     United States |   Canada |            Mexico |
        |:-----|------------------:|---------:|------------------:|
        | 2015 | -853792           |    -1231 | -742271           |
        | 2016 |      -1.02094e+06 |    -9182 | -556067           |
        | 2017 | -872728           |    -2401 | -233109           |
        | 2018 |      -1.27096e+06 |     8046 | -517006           |
        | 2019 |      -1.44074e+06 |     -400 | -569358           |
        | 2020 |      -3.20097e+06 |  -243127 |      -1.03224e+06 |
        | 2021 |      -2.82551e+06 |   -77523 |      -1.00013e+06 |
        | 2022 | -961028           |    17672 |      -1.27619e+06 |
        | 2023 |      -2.18266e+06 |    -4808 |      -1.36965e+06 |
        | 2024 |      -2.31875e+06 |   -65919 |      -1.93368e+06 |
        | 2025 |      -2.23721e+06 |   -57263 |     nan           |
        """

        government_deficit = self._get_gmdb_series(
            self._government_variable("govdef", level), gmdb_forecasts=gmdb_forecasts
        )

        return finalize_dataset(
            dataset=government_deficit,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Deficit",
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
    def get_government_deficit_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
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
        Get the Government Deficit to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Deficit to GDP Ratio is the ratio of the total amount of money that a government spends more than it
        collects from taxes and other sources to the Gross Domestic Product (GDP). A government deficit is usually financed
        by borrowing money.

        The budget balance as a share of GDP is the headline fiscal indicator, with the European Union's 3% of GDP as the
        best-known reference for deficits. It includes interest payments and moves with the business cycle, so analysts
        also look at the primary balance (excluding interest) and the cyclically adjusted balance to judge the fiscal
        stance. A deficit that stays above the nominal growth rate times the debt ratio keeps pushing debt to GDP up.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The series is signed as a fiscal balance rather than as a deficit, and expressed as a
        decimal fraction of GDP: -0.07068 means a deficit of 7.068% of GDP, and a positive
        value is a surplus.

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Government Deficit to GDP Ratio = (Government Revenue - Government Expenditure) / Nominal GDP,
          using the level of government selected with `level` ("consolidated" by default)

        Also known as: deficit-to-GDP, fiscal balance.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
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
            pd.DataFrame: A DataFrame containing the Government Deficit to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        government_deficit_to_gdp_ratio = economics.get_government_deficit_to_gdp_ratio()

        government_deficit_to_gdp_ratio.loc[:, ['New Zealand', 'Australia', 'United Kingdom']]
        ```

        Which returns:

        |      |   New Zealand |   Australia |   United Kingdom |
        |:-----|--------------:|------------:|-----------------:|
        | 2015 |        0.0012 |     -0.0104 |          -0.045  |
        | 2016 |        0.0115 |     -0.0155 |          -0.0327 |
        | 2017 |        0.0162 |     -0.0063 |          -0.0246 |
        | 2018 |        0.0083 |     -0.0056 |          -0.0219 |
        | 2019 |       -0.0028 |     -0.0121 |          -0.0246 |
        | 2020 |       -0.0758 |     -0.1142 |          -0.1291 |
        | 2021 |       -0.0388 |     -0.0417 |          -0.0774 |
        | 2022 |       -0.0292 |     -0.0193 |          -0.0473 |
        | 2023 |       -0.0348 |     -0.0135 |          -0.0604 |
        | 2024 |       -0.0299 |     -0.0277 |          -0.0613 |
        | 2025 |       -0.0333 |     -0.0393 |          -0.0546 |
        """

        government_deficit_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govdef_GDP", level),
            in_percent=True,
            gmdb_forecasts=gmdb_forecasts,
        )

        return finalize_dataset(
            dataset=government_deficit_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Deficit to GDP Ratio",
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
    def get_government_interest_expenditure(
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
        Get the Government Interest Expenditure for a variety of countries over time from the
        Government Finance Statistics of the International Monetary Fund (IMF). It is the
        interest the general government (the central, state and local governments and the
        social security funds together) owes on its debt: GFS item 24, interest payable, which
        is D.41 in the national accounts.

        The interest is recorded when it accrues rather than when it is paid, and before the
        interest the government receives is netted against it. It is therefore higher than
        the net interest of a national budget: for the United States, the general government
        interest also covers the states and local governments, and the federal net interest
        the Congressional Budget Office reports subtracts the interest the federal government
        receives.

        The figures are quarterly (period="quarterly", the default) or yearly and expressed in
        millions of national currency, so levels are not comparable across countries with
        different currencies, but growth rates are. Quarterly figures are not seasonally
        adjusted; trailing=4 gives the interest over the last four quarters. The IMF's
        quarterly government finance statistics cover around 40 economies, among which the
        United States, the euro area and its members, the United Kingdom, Canada, Australia,
        Brazil, Turkey and Indonesia; countries outside them are left out with a warning.
        For those, get_government_expenditure_by_function with
        functions="Public Debt Transactions" has the yearly public debt transactions (COFOG
        01.7, mostly interest) of around 180 countries.

        Data comes from the IMF's Quarterly Government Finance Statistics, further information
        can be found within https://data.imf.org/en/datasets/IMF.STA:QGFS

        Also known as: interest payments, debt service cost, interest outlays, cost of
        government debt.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            period (str | None, optional): "quarterly" or "yearly". Defaults to None, which is
                quarterly, the highest frequency published.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Interest Expenditure

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_interest_expenditure(
            countries=['United States', 'Germany', 'Italy'], period='yearly')
        ```

        Which returns:

        |      |    United States |   Germany |   Italy |
        |:-----|-----------------:|----------:|--------:|
        | 2015 | 428286           |     42913 |   68116 |
        | 2016 | 468186           |     37888 |   66258 |
        | 2017 | 494638           |     34489 |   65333 |
        | 2018 | 571996           |     31787 |   64366 |
        | 2019 | 606869           |     28052 |   60214 |
        | 2020 | 541832           |     22390 |   57125 |
        | 2021 | 631404           |     21756 |   63003 |
        | 2022 | 815793           |     28018 |   81614 |
        | 2023 | 997791           |     36848 |   77780 |
        | 2024 |      1.17905e+06 |     45828 |   85484 |
        | 2025 |      1.24322e+06 |     49544 |   87145 |
        """
        period = validate_period(
            self._default_period(period, "quarterly"),
            list(imf_model.PERIOD_CODES),
            "government interest expenditure",
        )

        government_interest_expenditure = imf_model.get_government_interest(
            period, self._start_date, self._end_date
        )

        return finalize_dataset(
            dataset=government_interest_expenditure,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Interest Expenditure",
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
    def get_government_interest_expenditure_to_gdp_ratio(
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
        Get the Government Interest Expenditure to GDP Ratio for a variety of countries over
        time from the International Monetary Fund (IMF). It is the interest the general
        government owes on its debt (GFS item 24, interest payable, D.41 in the national
        accounts) as a share of the Gross Domestic Product (GDP).

        The interest burden as a share of GDP shows how much of the economy's output goes to
        servicing government debt. It rises with the debt ratio and with the interest rate the
        debt is refinanced at, and is the item that turns a primary balance into the overall
        balance. Comparing it with the spending on a function, such as defence with
        get_government_expenditure_by_function_to_gdp_ratio, uses the same general government
        and the same GDP.

        The interest is general government interest payable, on an accrual basis and before
        interest received is netted, so it is higher than the net interest of a national budget
        (see get_government_interest_expenditure). Both the interest and the GDP come from the
        IMF: its Quarterly Government Finance Statistics and its National Economic Accounts
        (quarterly GDP not seasonally adjusted, like the interest). Around 40 economies are
        covered; countries outside them are left out with a warning.

        The ratio is expressed as a decimal fraction (0.0402 for 4.02% of GDP).

        The formula is as follows:

        - Government Interest Expenditure to GDP Ratio = Interest Payable / Nominal GDP,
          both summed over the last four quarters for period="quarterly" (an annual ratio
          without seasonal swings) or over the year for period="yearly"

        Also known as: interest burden, interest to GDP, debt service to GDP.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            period (str | None, optional): "quarterly" (the ratio over the four quarters up to
                each quarter) or "yearly". Defaults to None, which is quarterly, the highest
                frequency published.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Interest Expenditure to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_interest_expenditure_to_gdp_ratio(
            countries=['United States', 'Germany', 'Italy', 'United Kingdom'], period='yearly')
        ```

        Which returns:

        |      |   United States |   Germany |   Italy |   United Kingdom |
        |:-----|----------------:|----------:|--------:|-----------------:|
        | 2015 |          0.0234 |    0.0139 |  0.041  |           0.0244 |
        | 2016 |          0.0249 |    0.0118 |  0.0389 |           0.0256 |
        | 2017 |          0.0252 |    0.0103 |  0.0375 |           0.028  |
        | 2018 |          0.0277 |    0.0092 |  0.0363 |           0.0258 |
        | 2019 |          0.0282 |    0.0079 |  0.0334 |           0.0229 |
        | 2020 |          0.0253 |    0.0064 |  0.0343 |           0.0208 |
        | 2021 |          0.0266 |    0.0059 |  0.0343 |           0.0277 |
        | 2022 |          0.0313 |    0.007  |  0.0408 |           0.0426 |
        | 2023 |          0.0359 |    0.0087 |  0.0363 |           0.0322 |
        | 2024 |          0.0402 |    0.0104 |  0.0387 |           0.0292 |
        | 2025 |          0.0404 |    0.0109 |  0.0385 |           0.0332 |
        """
        period = validate_period(
            self._default_period(period, "quarterly"),
            list(imf_model.PERIOD_CODES),
            "government interest expenditure to GDP ratio",
        )

        government_interest_expenditure_to_gdp_ratio = (
            imf_model.get_government_interest(
                period, self._start_date, self._end_date, denominator="gdp"
            )
        )

        return finalize_dataset(
            dataset=government_interest_expenditure_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Interest Expenditure to GDP Ratio",
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
    def get_government_interest_expenditure_to_revenue_ratio(
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
        Get the Government Interest Expenditure to Revenue Ratio for a variety of countries
        over time from the Government Finance Statistics of the International Monetary Fund
        (IMF). It is the interest the general government owes on its debt (GFS item 24,
        interest payable) as a share of its revenue (GFS item 1: taxes, social contributions,
        grants and other revenue).

        The share of revenue that goes to interest is the measure rating agencies use for debt
        affordability: it says how much of what the government collects is committed to its
        creditors before anything else is paid for. Both come from the same statistics of the
        same general government, on an accrual basis, with the interest before interest
        received is netted. Around 40 economies are covered; countries outside them are left
        out with a warning.

        The ratio is expressed as a decimal fraction (0.1351 for 13.51% of revenue).

        The formula is as follows:

        - Government Interest Expenditure to Revenue Ratio = Interest Payable / Government
          Revenue, both summed over the last four quarters for period="quarterly" (an annual
          ratio without the seasonal swings of tax receipts) or over the year for
          period="yearly"

        Also known as: interest to revenue, interest burden on revenue, debt affordability.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            period (str | None, optional): "quarterly" (the ratio over the four quarters up to
                each quarter) or "yearly". Defaults to None, which is quarterly, the highest
                frequency published.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Interest Expenditure to Revenue Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2024-01-01', end_date="2025-12-31")

        economics.get_government_interest_expenditure_to_revenue_ratio(
            countries=['United States', 'France', 'Canada'])
        ```

        Which returns:

        |        |   United States |   France |   Canada |
        |:-------|----------------:|---------:|---------:|
        | 2024Q1 |          0.1258 |   0.038  |   0.0788 |
        | 2024Q2 |          0.1301 |   0.0386 |   0.0805 |
        | 2024Q3 |          0.1335 |   0.0402 |   0.0822 |
        | 2024Q4 |          0.1351 |   0.04   |   0.082  |
        | 2025Q1 |          0.1354 |   0.0416 |   0.0812 |
        | 2025Q2 |          0.1347 |   0.0418 |   0.0808 |
        | 2025Q3 |          0.1334 |   0.0423 |   0.0792 |
        | 2025Q4 |          0.1325 |   0.0427 |   0.08   |
        """
        period = validate_period(
            self._default_period(period, "quarterly"),
            list(imf_model.PERIOD_CODES),
            "government interest expenditure to revenue ratio",
        )

        government_interest_expenditure_to_revenue_ratio = (
            imf_model.get_government_interest(
                period, self._start_date, self._end_date, denominator="revenue"
            )
        )

        return finalize_dataset(
            dataset=government_interest_expenditure_to_revenue_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Interest Expenditure to Revenue Ratio",
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

    @staticmethod
    def _expenditure_functions(functions: list[str] | str | None) -> list[str]:
        """
        Returns the government functions asked for, by their names in
        imf_model.EXPENDITURE_FUNCTIONS, in any casing.

        Args:
            functions (list[str] | str | None): The functions, a comma-separated string, or
                None for the ten COFOG divisions.

        Returns:
            list[str]: The functions.

        Raises:
            ValueError: When a function is not known.
        """
        names = {name.lower(): name for name in imf_model.EXPENDITURE_FUNCTIONS}
        requested = helpers.split_values(functions) or imf_model.DIVISIONS

        if unknown := [name for name in requested if str(name).lower() not in names]:
            raise ValueError(
                f"Unknown government function(s) {', '.join(map(repr, unknown))}. Choose from "
                f"{', '.join(map(repr, imf_model.EXPENDITURE_FUNCTIONS))}."
            )

        return list(dict.fromkeys(names[str(name).lower()] for name in requested))

    @handle_errors
    def get_government_expenditure_by_function(
        self,
        countries: list[str] | str | None = None,
        functions: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Government Expenditure by Function for a variety of countries over time from
        the Government Finance Statistics of the International Monetary Fund (IMF). The
        spending of the general government (the central, state and local governments and the
        social security funds together) is split by purpose according to the United Nations'
        Classification of the Functions of Government (COFOG), the classification Eurostat,
        the OECD and the IMF all publish.

        The ten COFOG divisions are General Public Services, Defence, Public Order and Safety,
        Economic Affairs, Environmental Protection, Housing and Community Amenities, Health,
        Recreation, Culture and Religion, Education and Social Protection, which add up to the
        Total. General Public Services include the Public Debt Transactions (group 01.7),
        which are mostly the interest on government debt, also available as a function of its
        own where a country reports it. Defence is military and civil defence, foreign military
        aid and defence research; military pensions are Social Protection.

        The data is yearly, in millions of national currency, for around 180 countries; the
        most recent year is usually one or two years back. Levels are not comparable across
        countries with different currencies, but growth rates are, and
        get_government_expenditure_by_function_to_gdp_ratio returns the shares of GDP.

        The columns are a country and a function each; a country without data for a function
        has no column for it, and countries without data are left out with a warning.

        Data comes from the IMF's Government Finance Statistics, further information can be
        found within https://data.imf.org/en/datasets/IMF.STA:GFS_COFOG

        Also known as: COFOG, government spending by function, defence spending, military
        spending, health spending, education spending, social spending.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            functions (list[str] | str | None, optional): The functions to return, a list or a
                comma-separated string of the ten divisions above, "Public Debt Transactions"
                or "Total". Defaults to None, which returns the ten divisions.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Expenditure by Function, with a
            (Country, Function) column

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date="2025-12-31")

        expenditure = economics.get_government_expenditure_by_function(
            countries=['United States', 'Germany'], functions='Defence,Health,Education')

        expenditure['United States']
        ```

        Which returns:

        |      |   Defence |      Health |   Education |
        |:-----|----------:|------------:|------------:|
        | 2018 |    660147 | 1.91712e+06 | 1.22016e+06 |
        | 2019 |    712831 | 2.0491e+06  | 1.26193e+06 |
        | 2020 |    744556 | 2.26781e+06 | 1.28909e+06 |
        | 2021 |    762090 | 2.45445e+06 | 1.37456e+06 |
        | 2022 |    775165 | 2.64297e+06 | 1.47586e+06 |
        | 2023 |    843209 | 2.79374e+06 | 1.54819e+06 |
        | 2024 |    911017 | 3.01146e+06 | 1.63922e+06 |
        """
        government_expenditure_by_function = imf_model.get_expenditure_by_function(
            self._expenditure_functions(functions), self._start_date, self._end_date
        )

        return finalize_dataset(
            dataset=government_expenditure_by_function,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Expenditure by Function",
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
    def get_government_expenditure_by_function_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        functions: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Government Expenditure by Function to GDP Ratio for a variety of countries over
        time from the Government Finance Statistics of the International Monetary Fund (IMF).
        It is the spending of the general government on each function of the United Nations'
        Classification of the Functions of Government (COFOG) as a share of the Gross Domestic
        Product (GDP).

        Shares of GDP make the functions comparable across countries and over time, such as
        NATO's 2% of GDP guideline for defence. The ten divisions, the Public Debt Transactions
        (group 01.7, mostly interest) and the Total are described in
        get_government_expenditure_by_function. Defence covers military and civil defence,
        foreign military aid and defence research, not military pensions, so it differs from
        NATO's definition and from a national defence budget.

        For the interest itself, get_government_interest_expenditure_to_gdp_ratio uses the
        same general government and the same GDP, so the two together answer whether a
        government pays more in interest than it spends on defence.

        The data is yearly, for around 180 countries; the most recent year is usually one or
        two years back. The ratio is the IMF's own figure, published in percent and divided by
        100 here, so it is a decimal fraction (0.0311 for 3.11% of GDP).

        The formula is as follows:

        - Government Expenditure by Function to GDP Ratio = Expenditure on the Function /
          Nominal GDP

        Also known as: defence spending to GDP, military spending to GDP, health spending to
        GDP, education spending to GDP, social spending to GDP, COFOG to GDP.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            functions (list[str] | str | None, optional): The functions to return, a list or a
                comma-separated string of the ten COFOG divisions, "Public Debt Transactions"
                or "Total". Defaults to None, which returns the ten divisions.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Expenditure by Function to GDP
            Ratio, with a (Country, Function) column

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        defence = economics.get_government_expenditure_by_function_to_gdp_ratio(
            countries=['United States', 'Japan', 'Germany', 'United Kingdom'], functions='Defence')

        defence.xs('Defence', axis=1, level='Function')
        ```

        Which returns:

        |      |   United States |   Japan |   Germany |   United Kingdom |
        |:-----|----------------:|--------:|----------:|-----------------:|
        | 2015 |          0.0327 |  0.0086 |    0.0094 |           0.0198 |
        | 2016 |          0.0318 |  0.0079 |    0.0096 |           0.0189 |
        | 2017 |          0.0314 |  0.0085 |    0.0099 |           0.0185 |
        | 2018 |          0.032  |  0.009  |    0.0105 |           0.0183 |
        | 2019 |          0.0331 |  0.0089 |    0.0106 |           0.0196 |
        | 2020 |          0.0348 |  0.0092 |    0.0112 |           0.0214 |
        | 2021 |          0.0321 |  0.0099 |    0.0114 |           0.0212 |
        | 2022 |          0.0298 |  0.0092 |    0.0115 |           0.0214 |
        | 2023 |          0.0303 |  0.0105 |    0.0125 |           0.022  |
        | 2024 |          0.0311 |  0.013  |    0.0135 |           0.0225 |
        """
        government_expenditure_by_function_to_gdp_ratio = (
            imf_model.get_expenditure_by_function(
                self._expenditure_functions(functions),
                self._start_date,
                self._end_date,
                to_gdp=True,
            )
        )

        return finalize_dataset(
            dataset=government_expenditure_by_function_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Expenditure by Function to GDP Ratio",
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
    def get_trust_in_government(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Trust in government refers to the share of people who report having confidence
        in the national government. The data shown reflect the share of respondents
        answering “yes” (the other response categories being “no”, and “don’t know”)
        to the survey question: “In this country, do you have confidence in… national government?”

        Due to small sample sizes, country averages for horizontal inequalities (by age,
        gender and education) are pooled between 2010-18 to improve the accuracy of the
        estimates.

        The sample is ex ante designed to be nationally representative of the population
        aged 15 and over. The population-wide figure is returned -- both sexes, all education
        levels -- as an annual decimal fraction of that population (0.3933 for 39.33%). The
        breakdowns by sex and education level that the OECD publishes alongside it are not
        returned here.

        See definition: https://data.oecd.org/gga/trust-in-government.htm

        Also known as: political trust, institutional trust.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
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
            pd.DataFrame: A DataFrame containing the Trust in Government.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(end_date="2025-12-31")

        trust_in_government = economics.get_trust_in_government()

        trust_in_government.loc[:, ['United States', 'Greece', 'Japan']]
        ```

        Which returns:

        |      |   United States |   Greece |   Japan |
        |:-----|----------------:|---------:|--------:|
        | 2006 |          0.4756 |   0.4345 |  0.2952 |
        | 2007 |          0.4756 |   0.4345 |  0.2952 |
        | 2008 |          0.4335 |   0.2764 |  0.2478 |
        | 2009 |          0.4335 |   0.2764 |  0.2478 |
        | 2010 |          0.4335 |   0.2764 |  0.2478 |
        | 2011 |          0.34   |   0.1483 |  0.2528 |
        | 2012 |          0.34   |   0.1483 |  0.2528 |
        | 2013 |          0.34   |   0.1483 |  0.2528 |
        | 2014 |          0.3309 |   0.2527 |  0.3649 |
        | 2015 |          0.3309 |   0.2527 |  0.3649 |
        | 2016 |          0.3309 |   0.2527 |  0.3649 |
        | 2017 |          0.3544 |   0.2311 |  0.4029 |
        | 2018 |          0.3544 |   0.2311 |  0.4029 |
        | 2019 |          0.3544 |   0.2311 |  0.4029 |
        | 2020 |          0.3933 |   0.3518 |  0.3819 |
        | 2021 |          0.3933 |   0.3518 |  0.3819 |
        | 2022 |          0.3933 |   0.3518 |  0.3819 |
        | 2023 |          0.2944 |   0.2732 |  0.3215 |
        | 2024 |          0.2944 |   0.2732 |  0.3215 |
        | 2025 |          0.2944 |   0.2732 |  0.3215 |
        """
        trust_in_government = oecd_model.get_trust_in_government(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=trust_in_government,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Trust in Government",
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
