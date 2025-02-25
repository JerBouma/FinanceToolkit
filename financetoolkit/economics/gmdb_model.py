"""GMBD Model"""

import pandas as pd

GMD_LOCATION = "https://github.com/KMueller-Lab/Global-Macro-Database/blob/main/data/final/data_final.dta?raw=True"


def collect_global_macro_database_dataset(
    gmd_location: str = GMD_LOCATION,
) -> pd.DataFrame:
    """
    Collect and transform the Global Macro Database dataset.
    Reads a Stata file, processes it by converting 'year' to integers, removing 'ISO3' if present,
    and setting a multi-index of 'year' and 'countryname'. The dataset is then unstacked by 'countryname'.

    Args:
        gmd_location (str): The file path to the Stata dataset. Defaults to GMD_LOCATION.

    Returns:
        pd.DataFrame: A transformed DataFrame indexed by 'year' with country-wise columns.
    """
    gmd_dataset = pd.read_stata(filepath_or_buffer=gmd_location)
    gmd_dataset["year"] = pd.PeriodIndex(gmd_dataset["year"].astype(int), freq="Y")
    gmd_dataset = gmd_dataset.set_index(["year", "countryname"])
    gmd_dataset.index.names = [None] * gmd_dataset.index.nlevels
    gmd_dataset = gmd_dataset.unstack(level=1)

    gmd_dataset = gmd_dataset.sort_index(axis=1)

    return gmd_dataset


def get_nominal_gross_domestic_product(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves nominal GDP ('nGDP'), removing rows with all NaNs."""
    return gmd_dataset["nGDP"].dropna(axis="rows", how="all")


def get_real_gross_domestic_product(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves real GDP ('rGDP'), removing rows with all NaNs."""
    return gmd_dataset["rGDP"].dropna(axis="rows", how="all")


def get_gross_domestic_product_deflator(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves GDP deflator ('deflator'), removing rows with all NaNs."""
    return gmd_dataset["deflator"].dropna(axis="rows", how="all")


def get_population(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves population data ('pop'), removing rows with all NaNs."""
    return gmd_dataset["pop"].dropna(axis="rows", how="all")


def get_total_consumption(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves total consumption ('cons'), removing rows with all NaNs."""
    return gmd_dataset["cons"].dropna(axis="rows", how="all")


def get_total_consumption_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts total consumption to GDP ratio ('cons_GDP'), removing rows with all NaNs."""
    return gmd_dataset["cons_GDP"].dropna(axis="rows", how="all")


def get_real_total_consumption(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves real total consumption ('rcons'), removing rows with all NaNs."""
    return gmd_dataset["rcons"].dropna(axis="rows", how="all")


def get_investment(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves investment data ('inv'), removing rows with all NaNs."""
    return gmd_dataset["inv"].dropna(axis="rows", how="all")


def get_investment_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts investment to GDP ratio ('inv_GDP'), removing rows with all NaNs."""
    return gmd_dataset["inv_GDP"].dropna(axis="rows", how="all")


def get_fixed_investment(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves fixed investment ('finv'), removing rows with all NaNs."""
    return gmd_dataset["finv"].dropna(axis="rows", how="all")


def get_fixed_investment_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts fixed investment to GDP ratio ('finv_GDP'), removing rows with all NaNs."""
    return gmd_dataset["finv_GDP"].dropna(axis="rows", how="all")


def get_exports(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves exports data ('exports'), removing rows with all NaNs."""
    return gmd_dataset["exports"].dropna(axis="rows", how="all")


def get_exports_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts exports to GDP ratio ('exports_GDP'), removing rows with all NaNs."""
    return gmd_dataset["exports_GDP"].dropna(axis="rows", how="all")


def get_imports(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves imports data ('imports'), removing rows with all NaNs."""
    return gmd_dataset["imports"].dropna(axis="rows", how="all")


def get_imports_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts imports to GDP ratio ('imports_GDP'), removing rows with all NaNs."""
    return gmd_dataset["imports_GDP"].dropna(axis="rows", how="all")


def get_current_account_balance(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts current account balance ('CA') from the dataset, removing NaN rows."""
    return gmd_dataset["CA"].dropna(axis="rows", how="all")


def get_current_account_balance_to_gdp(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts current account balance to GDP ('CA_GDP') from the dataset, removing NaN rows."""
    return gmd_dataset["CA_GDP"].dropna(axis="rows", how="all")


def get_real_effective_exchange_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves real effective exchange rate ('REER'), removing rows with all NaNs."""
    return gmd_dataset["REER"].dropna(axis="rows", how="all")


def get_usd_exchange_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves the USD exchange rate ('USDfx'), removing rows with all NaNs."""
    return gmd_dataset["USDfx"].dropna(axis="rows", how="all")


def get_government_debt(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts government debt ('govdebt'), removing rows with all NaNs."""
    return gmd_dataset["govdebt"].dropna(axis="rows", how="all")


def get_government_debt_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts government debt to GDP ratio ('govdebt_GDP'), removing rows with all NaNs."""
    return gmd_dataset["govdebt_GDP"].dropna(axis="rows", how="all")


def get_government_revenue(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government revenue ('govrev'), removing rows with all NaNs."""
    return gmd_dataset["govrev"].dropna(axis="rows", how="all")


def get_government_revenue_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts government revenue to GDP ratio ('govrev_GDP'), removing rows with all NaNs."""
    return gmd_dataset["govrev_GDP"].dropna(axis="rows", how="all")


def get_government_tax_revenue(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government tax revenue ('govtax'), removing rows with all NaNs."""
    return gmd_dataset["govtax"].dropna(axis="rows", how="all")


def get_government_tax_revenue_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts government tax revenue to GDP ratio ('govtax_GDP'), removing rows with all NaNs."""
    return gmd_dataset["govtax_GDP"].dropna(axis="rows", how="all")


def get_government_expenditure(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government expenditure ('govexp'), removing rows with all NaNs."""
    return gmd_dataset["govexp"].dropna(axis="rows", how="all")


def get_government_expenditure_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts government expenditure to GDP ratio ('govexp_GDP'), removing rows with all NaNs."""
    return gmd_dataset["govexp_GDP"].dropna(axis="rows", how="all")


def get_government_deficit(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government deficit ('govdef'), removing rows with all NaNs."""
    return gmd_dataset["govdef"].dropna(axis="rows", how="all")


def get_government_deficit_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves government deficit to GDP ('govdef_GDP'), removing rows with all NaNs."""
    return gmd_dataset["govdef_GDP"].dropna(axis="rows", how="all")


def get_money_supply(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts money supply data ('M0', 'M1', 'M2', 'M3', 'M4'), dropping rows with all NaNs."""
    return gmd_dataset[["M0", "M1", "M2", "M3", "M4"]].dropna(axis="rows", how="all")


def get_central_bank_policy_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves central bank policy rate ('cbrate'), removing rows with all NaNs."""
    return gmd_dataset["cbrate"].dropna(axis="rows", how="all")


def get_short_term_interest_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves short-term interest rate ('strate'), removing rows with all NaNs."""
    return gmd_dataset["strate"].dropna(axis="rows", how="all")


def get_long_term_interest_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves long-term interest rate ('ltrate'), removing rows with all NaNs."""
    return gmd_dataset["ltrate"].dropna(axis="rows", how="all")


def get_consumer_price_index(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves consumer price index ('CPI') data, removing rows with all NaNs."""
    return gmd_dataset["CPI"].dropna(axis="rows", how="all")


def get_house_price_index(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves house price index ('houseprice'), removing rows with all NaNs."""
    return gmd_dataset["HPI"].dropna(axis="rows", how="all")


def get_inflation_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves inflation rate ('infl'), removing rows with all NaNs."""
    return gmd_dataset["infl"].dropna(axis="rows", how="all")


def get_unemployment_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieves unemployment rate ('unemp'), removing rows with all NaNs."""
    return gmd_dataset["unemp"].dropna(axis="rows", how="all")
