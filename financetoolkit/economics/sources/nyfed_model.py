"""Federal Reserve Bank of New York Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.excel_model import read_excel
from financetoolkit.utilities.requests_model import get_request

# The Treasury term premia of Adrian, Crump and Moench (2013), which the New York Fed
# updates daily from 1961 in one workbook with a daily and a month-end worksheet.
TERM_PREMIUM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
TERM_PREMIUM_SHEET = "ACM Daily"

# Per maturity of 1 to 10 years, the workbook holds the fitted yield (ACMY), the term
# premium (ACMTP) and the risk-neutral yield (ACMRNY): the yield if investors were
# indifferent to risk, i.e. the average expected short rate over the maturity.
TERM_PREMIUM_COMPONENTS = {
    "ACMTP": "Term Premium",
    "ACMRNY": "Expected Short Rate",
    "ACMY": "Fitted Yield",
}
TERM_PREMIUM_MATURITIES = range(1, 11)

# The Global Supply Chain Pressure Index, which the New York Fed updates monthly from
# 1998 in a workbook with a single monthly worksheet.
SUPPLY_CHAIN_PRESSURE_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
SUPPLY_CHAIN_PRESSURE_SHEET = "GSCPI Monthly Data"


def get_term_premium() -> pd.DataFrame:
    """
    Retrieves the daily Treasury term premia of the Adrian, Crump and Moench (2013) model
    from 1961: per maturity of 1 to 10 years the term premium, the expected short rate
    (the risk-neutral yield) and the fitted zero-coupon yield, which is their sum.

    Returns:
        pd.DataFrame: The components as decimals (continuously compounded), indexed by day
        with a column per component and maturity ("1Y" to "10Y").

    Raises:
        ValueError: When the workbook misses one of the expected columns.
    """
    description = "ACM term premium"

    def fetch() -> pd.DataFrame:
        response = get_request(TERM_PREMIUM_URL, timeout=120)
        data = read_excel(io.BytesIO(response.content), sheet_name=TERM_PREMIUM_SHEET)

        columns = {
            f"{code}{maturity:02d}": (component, f"{maturity}Y")
            for code, component in TERM_PREMIUM_COMPONENTS.items()
            for maturity in TERM_PREMIUM_MATURITIES
        }
        require_columns(data, {"DATE", *columns}, description)

        term_premium = data[list(columns)].apply(pd.to_numeric, errors="coerce")
        term_premium.columns = pd.MultiIndex.from_tuples(list(columns.values()))
        term_premium.index = pd.PeriodIndex(
            pd.to_datetime(data["DATE"], format="%d-%b-%Y"), freq="D"
        )
        term_premium.index.name = None

        # The workbook is in percent, while rates are decimals.
        return (term_premium / 100).dropna(how="all").sort_index()

    return collect_cached_data(
        source=policy_model.FEDERAL_RESERVE,
        dataset="series",
        entity="acm_term_premium",
        fetch=fetch,
        description=description,
    )


def get_supply_chain_pressure_index() -> pd.DataFrame:
    """
    Retrieves the monthly Global Supply Chain Pressure Index (GSCPI) from 1998: the
    pressure on global supply chains in standard deviations from its average, combining
    shipping costs (the Baltic Dry Index, the Harpex index and air freight rates) with the
    delivery times, backlogs and purchased stocks of the manufacturing surveys of the
    largest economies.

    Returns:
        pd.DataFrame: The index in standard deviations, indexed by month with a single
        "World" column.

    Raises:
        ValueError: When the workbook misses one of the expected columns.
    """
    description = "Global Supply Chain Pressure Index"

    def fetch() -> pd.DataFrame:
        response = get_request(SUPPLY_CHAIN_PRESSURE_URL, timeout=60)
        data = read_excel(
            io.BytesIO(response.content), sheet_name=SUPPLY_CHAIN_PRESSURE_SHEET
        )
        require_columns(data, {"Date", "GSCPI"}, description)

        # Empty rows sit between the header and the first month.
        data = data.dropna(subset=["Date", "GSCPI"])
        index = pd.PeriodIndex(pd.to_datetime(data["Date"], format="mixed"), freq="M")
        index.name = None

        return pd.DataFrame(
            {"World": pd.to_numeric(data["GSCPI"], errors="coerce").to_numpy()},
            index=index,
        ).sort_index()

    return collect_cached_data(
        source=policy_model.FEDERAL_RESERVE,
        dataset="series",
        entity="gscpi",
        fetch=fetch,
        description=description,
    )
