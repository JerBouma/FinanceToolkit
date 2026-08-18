"""GMBD Model"""

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import Cache
from financetoolkit.utilities.requests_model import get_request

GMD_LOCATION = "https://github.com/KMueller-Lab/Global-Macro-Database/blob/main/data/final/data_final.dta?raw=True"


def collect_global_macro_database_dataset(
    gmd_location: str = GMD_LOCATION,
    cache: Cache | None = None,
) -> pd.DataFrame:
    """
    Collect and transform the Global Macro Database dataset.
    Reads a Stata file, processes it by converting 'year' to integers, removing 'ISO3' if present,
    and setting a multi-index of 'year' and 'countryname'. The dataset is then unstacked by 'countryname'.

    The Global Macro Database is published as one annual Stata file covering every
    country at once, so there is no per-country request to make and nothing to append
    to incrementally. It is therefore cached whole, which still removes a multi-megabyte
    download from every run that happens within the cache's freshness window.

    Note on units: the Global Macro Database quotes every rate, share and ratio variable in
    percentage points (3.625 for a 3.625% policy rate, 81.0 for consumption worth 81% of
    GDP). Each getter below that returns such a series divides it by 100, so the Finance
    Toolkit returns decimal fractions throughout and the GMDB series line up with the OECD
    and FRED ones. Levels (currency amounts, head counts), price indices (CPI, HPI, REER,
    the GDP deflator), exchange rates and the binary crisis dummies are returned unchanged.

    Args:
        gmd_location (str): The file path to the Stata dataset. Defaults to GMD_LOCATION.
        cache (Cache | None): An optional cache to serve the dataset from and store it in.

    Returns:
        pd.DataFrame: A transformed DataFrame indexed by 'year' with country-wise columns.
    """
    if cache is not None and cache.enabled:
        cached_dataset = cache.get(
            source=policy_model.GLOBAL_MACRO_DATABASE,
            dataset="dataset",
            entity="global",
        )

        if cached_dataset is not None:
            return cached_dataset

    response = get_request(gmd_location, timeout=30)
    response.raise_for_status()

    gmd_dataset = pd.read_stata(filepath_or_buffer=io.BytesIO(response.content))
    gmd_dataset["year"] = pd.PeriodIndex(gmd_dataset["year"].astype(int), freq="Y")
    gmd_dataset = gmd_dataset.set_index(["year", "countryname"])
    gmd_dataset.index.names = [None] * gmd_dataset.index.nlevels
    gmd_dataset = gmd_dataset.unstack(level=1)

    gmd_dataset = gmd_dataset.sort_index(axis=1)

    if cache is not None and cache.enabled:
        cache.set(
            source=policy_model.GLOBAL_MACRO_DATABASE,
            dataset="dataset",
            entity="global",
            data=gmd_dataset,
        )

    return gmd_dataset


def get_nominal_gross_domestic_product(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves nominal GDP ('nGDP'), removing rows with all NaNs."""
    return gmd_dataset["nGDP"].dropna(axis="rows", how="all")


def get_real_gross_domestic_product(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves real GDP ('rGDP'), removing rows with all NaNs."""
    return gmd_dataset["rGDP"].dropna(axis="rows", how="all")


def get_real_gross_domestic_product_usd(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves real GDP expressed in cross-country comparable USD ('rGDP_USD'), removing rows with all NaNs."""
    return gmd_dataset["rGDP_USD"].dropna(axis="rows", how="all")


def get_real_gross_domestic_product_per_capita(
    gmd_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Retrieves real GDP per capita ('rGDP_pc'), removing rows with all NaNs."""
    return gmd_dataset["rGDP_pc"].dropna(axis="rows", how="all")


def get_gross_domestic_product_deflator(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves GDP deflator ('deflator'), removing rows with all NaNs.

    This is a price index (100 in the base year) rather than a rate, so it is returned
    on the scale the Global Macro Database publishes it on.
    """
    return gmd_dataset["deflator"].dropna(axis="rows", how="all")


def get_population(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves population data ('pop'), removing rows with all NaNs.

    This is a head count in millions rather than a rate, so it is returned on the scale
    the Global Macro Database publishes it on.
    """
    return gmd_dataset["pop"].dropna(axis="rows", how="all")


def get_total_consumption(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves total consumption ('cons'), removing rows with all NaNs."""
    return gmd_dataset["cons"].dropna(axis="rows", how="all")


def get_total_consumption_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the total consumption to GDP ratio ('cons_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.81 for consumption worth 81% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    total_consumption_to_gdp_ratio = gmd_dataset["cons_GDP"].dropna(
        axis="rows", how="all"
    )

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return total_consumption_to_gdp_ratio / 100


def get_real_total_consumption(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves real total consumption ('rcons'), removing rows with all NaNs."""
    return gmd_dataset["rcons"].dropna(axis="rows", how="all")


def get_investment(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves investment data ('inv'), removing rows with all NaNs."""
    return gmd_dataset["inv"].dropna(axis="rows", how="all")


def get_investment_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the investment to GDP ratio ('inv_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.2248 for investment worth 22.48% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    investment_to_gdp_ratio = gmd_dataset["inv_GDP"].dropna(axis="rows", how="all")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return investment_to_gdp_ratio / 100


def get_fixed_investment(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves fixed investment ('finv'), removing rows with all NaNs."""
    return gmd_dataset["finv"].dropna(axis="rows", how="all")


def get_fixed_investment_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the fixed investment to GDP ratio ('finv_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.2179 for fixed investment worth 21.79% of
    GDP), matching the convention used everywhere else in the Finance Toolkit.
    """
    fixed_investment_to_gdp_ratio = gmd_dataset["finv_GDP"].dropna(
        axis="rows", how="all"
    )

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return fixed_investment_to_gdp_ratio / 100


def get_exports(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves exports data ('exports'), removing rows with all NaNs."""
    return gmd_dataset["exports"].dropna(axis="rows", how="all")


def get_exports_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the exports to GDP ratio ('exports_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.1016 for exports worth 10.16% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    exports_to_gdp_ratio = gmd_dataset["exports_GDP"].dropna(axis="rows", how="all")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return exports_to_gdp_ratio / 100


def get_imports(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves imports data ('imports'), removing rows with all NaNs."""
    return gmd_dataset["imports"].dropna(axis="rows", how="all")


def get_imports_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the imports to GDP ratio ('imports_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.1213 for imports worth 12.13% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    imports_to_gdp_ratio = gmd_dataset["imports_GDP"].dropna(axis="rows", how="all")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return imports_to_gdp_ratio / 100


def get_current_account_balance(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts current account balance ('CA') from the dataset, removing NaN rows."""
    return gmd_dataset["CA"].dropna(axis="rows", how="all")


def get_current_account_balance_to_gdp(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the current account balance to GDP ratio ('CA_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (-0.0211 for a deficit worth 2.11% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    current_account_balance_to_gdp = gmd_dataset["CA_GDP"].dropna(
        axis="rows", how="all"
    )

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return current_account_balance_to_gdp / 100


def get_usd_exchange_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the USD exchange rate ('USDfx'), removing rows with all NaNs.

    This is a price (units of local currency per US dollar) rather than a rate, so it is
    returned on the scale the Global Macro Database publishes it on.
    """
    return gmd_dataset["USDfx"].dropna(axis="rows", how="all")


def get_real_effective_exchange_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the real effective exchange rate ('REER'), removing rows with all NaNs.

    This is an index (100 in the base year) rather than a rate, so it is returned on the
    scale the Global Macro Database publishes it on.
    """
    return gmd_dataset["REER"].dropna(axis="rows", how="all")


def get_government_debt(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts government debt ('govdebt'), removing rows with all NaNs."""
    return gmd_dataset["govdebt"].dropna(axis="rows", how="all")


def get_government_debt_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the government debt to GDP ratio ('govdebt_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (1.3173 for debt worth 131.73% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    government_debt_to_gdp_ratio = gmd_dataset["govdebt_GDP"].dropna(
        axis="rows", how="all"
    )

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_debt_to_gdp_ratio / 100


def get_government_revenue(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government revenue ('govrev'), removing rows with all NaNs."""
    return gmd_dataset["govrev"].dropna(axis="rows", how="all")


def get_government_revenue_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the government revenue to GDP ratio ('govrev_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.3104 for revenue worth 31.04% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    government_revenue_to_gdp_ratio = gmd_dataset["govrev_GDP"].dropna(
        axis="rows", how="all"
    )

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_revenue_to_gdp_ratio / 100


def get_government_tax_revenue(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government tax revenue ('govtax'), removing rows with all NaNs."""
    return gmd_dataset["govtax"].dropna(axis="rows", how="all")


def get_government_tax_revenue_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the government tax revenue to GDP ratio ('govtax_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.1022 for tax revenue worth 10.22% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    government_tax_revenue_to_gdp_ratio = gmd_dataset["govtax_GDP"].dropna(
        axis="rows", how="all"
    )

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_tax_revenue_to_gdp_ratio / 100


def get_government_expenditure(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government expenditure ('govexp'), removing rows with all NaNs."""
    return gmd_dataset["govexp"].dropna(axis="rows", how="all")


def get_government_expenditure_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the government expenditure to GDP ratio ('govexp_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.3708 for spending worth 37.08% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    government_expenditure_to_gdp_ratio = gmd_dataset["govexp_GDP"].dropna(
        axis="rows", how="all"
    )

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_expenditure_to_gdp_ratio / 100


def get_government_deficit(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government deficit ('govdef'), removing rows with all NaNs."""
    return gmd_dataset["govdef"].dropna(axis="rows", how="all")


def get_government_deficit_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the government deficit to GDP ratio ('govdef_GDP'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (-0.0604 for a deficit worth 6.04% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.
    """
    government_deficit_to_gdp_ratio = gmd_dataset["govdef_GDP"].dropna(
        axis="rows", how="all"
    )

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_deficit_to_gdp_ratio / 100


def get_money_supply(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts money supply data ('M0', 'M1', 'M2', 'M3', 'M4'), dropping rows with all NaNs."""
    return gmd_dataset[["M0", "M1", "M2", "M3", "M4"]].dropna(axis="rows", how="all")


def get_central_bank_policy_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the central bank policy rate ('cbrate'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0538 for a 5.375% policy rate), matching
    the convention used everywhere else in the Finance Toolkit.
    """
    central_bank_policy_rate = gmd_dataset["cbrate"].dropna(axis="rows", how="all")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return central_bank_policy_rate / 100


def get_short_term_interest_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the short-term interest rate ('strate'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0408 for 4.08%), putting it on exactly
    the same scale as the OECD short-term interest rate.
    """
    short_term_interest_rate = gmd_dataset["strate"].dropna(axis="rows", how="all")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return short_term_interest_rate / 100


def get_long_term_interest_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the long-term interest rate ('ltrate'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0416 for 4.16%), putting it on exactly
    the same scale as the OECD long-term interest rate.
    """
    long_term_interest_rate = gmd_dataset["ltrate"].dropna(axis="rows", how="all")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return long_term_interest_rate / 100


def get_consumer_price_index(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves consumer price index ('CPI') data, removing rows with all NaNs.

    This is a price index (100 in the base year) rather than a rate, so it is returned
    on the scale the Global Macro Database publishes it on.
    """
    return gmd_dataset["CPI"].dropna(axis="rows", how="all")


def get_house_price_index(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves house price index ('HPI'), removing rows with all NaNs.

    This is a price index (100 in the base year) rather than a rate, so it is returned
    on the scale the Global Macro Database publishes it on.
    """
    return gmd_dataset["HPI"].dropna(axis="rows", how="all")


def get_inflation_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the inflation rate ('infl'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0214 for 2.14% inflation), matching the
    convention used everywhere else in the Finance Toolkit.
    """
    inflation_rate = gmd_dataset["infl"].dropna(axis="rows", how="all")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return inflation_rate / 100


def get_unemployment_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the unemployment rate ('unemp'), removing rows with all NaNs.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0397 for 3.97% unemployment), putting it
    on exactly the same scale as the OECD unemployment rate.
    """
    unemployment_rate = gmd_dataset["unemp"].dropna(axis="rows", how="all")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return unemployment_rate / 100


def get_sovereign_debt_crisis(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the sovereign debt crisis dummy ('SovDebtCrisis'), removing rows with all NaNs.

    The values are binary (0 = no crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating
    indicators, not a continuous economic series.
    """
    return gmd_dataset["SovDebtCrisis"].dropna(axis="rows", how="all")


def get_currency_crisis(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the currency crisis dummy ('CurrencyCrisis'), removing rows with all NaNs.

    The values are binary (0 = no crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating
    indicators, not a continuous economic series.
    """
    return gmd_dataset["CurrencyCrisis"].dropna(axis="rows", how="all")


def get_banking_crisis(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the banking crisis dummy ('BankingCrisis'), removing rows with all NaNs.

    The values are binary (0 = no crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating
    indicators, not a continuous economic series.
    """
    return gmd_dataset["BankingCrisis"].dropna(axis="rows", how="all")
