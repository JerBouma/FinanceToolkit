"""FED Model"""

__docformat__ = "google"

import pandas as pd

BASE_URL = "https://markets.newyorkfed.org/read"
EXTENSIONS_1 = "?startDt=2000-12-01&eventCodes="
CODES = {
    "EFFR": "500",
    "OBFR": "505",
    "TGCR": "510",
    "BGCR": "515",
    "SOFR": "520",
    "SOFRAI": "525",
}

EXTENSIONS_2 = "&productCode=50&sort=postDt:-1,eventCode:1&format=csv"

COLUMN_NAMES = {
    "Rate (%)": "Rate",
    "1st Percentile (%)": "1st Percentile",
    "25th Percentile (%)": "25th Percentile",
    "75th Percentile (%)": "75th Percentile",
    "99th Percentile (%)": "99th Percentile",
    "Target Rate From (%)": "Target Rate From",
    "Target Rate To (%)": "Target Rate To",
}


def collect_fed_data(fed_code: str) -> pd.DataFrame:
    """
    Collect the data from the Federal Reserve Bank of New York.

    Args:
        fed_code (str): The code for the data to be collected.

    Returns:
       pd.DataFrame: A DataFrame containing the data from the Federal
       Reserve Bank of New York.
    """
    fed_data = pd.read_csv(f"{BASE_URL}{EXTENSIONS_1}{fed_code}{EXTENSIONS_2}")

    fed_data = fed_data.set_index("Effective Date")

    fed_data.index = pd.PeriodIndex(data=fed_data.index, freq="D")

    fed_data = fed_data.sort_index()

    fed_data = fed_data[
        [
            "Rate (%)",
            "1st Percentile (%)",
            "25th Percentile (%)",
            "75th Percentile (%)",
            "99th Percentile (%)",
            "Volume ($Billions)",
            "Target Rate From (%)",
            "Target Rate To (%)",
        ]
    ]

    fed_data = fed_data.rename(columns=COLUMN_NAMES)

    return fed_data


def get_effective_federal_funds_rate() -> pd.DataFrame:
    """
    Get the Effective Federal Funds Rate from the Federal Reserve Bank of New York over
    time. The Effective Federal Funds Rate is the rate at which banks can
    borrow money from each other for the duration of one day.

    Returns:
       pd.DataFrame: A DataFrame containing the Effective Federal Funds Rate over time.
    """
    fed_code = CODES["EFFR"]

    effective_federal_funds_rate = collect_fed_data(fed_code)

    # Convert to percentages
    effective_federal_funds_rate.loc[
        :, effective_federal_funds_rate.columns != "Volume ($Billions)"
    ] = (
        effective_federal_funds_rate.loc[
            :, effective_federal_funds_rate.columns != "Volume ($Billions)"
        ]
        / 100
    )

    return effective_federal_funds_rate


def get_overnight_banking_funding_rate() -> pd.DataFrame:
    """
    Get the overnight banking funding rate from the Federal Reserve Bank of New York over
    time. The overnight banking funding rate is the rate at which depository institutions
    can borrow money that are required to meet overnight balances.

    Returns:
       pd.DataFrame: A DataFrame containing the Main Refinancing Operations over time.
    """
    fed_code = CODES["OBFR"]

    overnight_banking_funding_rate = collect_fed_data(fed_code)

    # Convert to percentages
    overnight_banking_funding_rate.loc[
        :, overnight_banking_funding_rate.columns != "Volume ($Billions)"
    ] = (
        overnight_banking_funding_rate.loc[
            :, overnight_banking_funding_rate.columns != "Volume ($Billions)"
        ]
        / 100
    )

    return overnight_banking_funding_rate


def get_tri_party_general_collateral_rate() -> pd.DataFrame:
    """
    Get the tri-party general collateral rate from the Federal Reserve Bank of New York over
    time. The tri-party general collateral rate covers specific-counterparty tri-party
    general collateral repo transactions.

    Returns:
       pd.DataFrame: A DataFrame containing the tri-party general collateral rate over time.
    """
    fed_code = CODES["TGCR"]

    tri_party_general_collateral_rate = collect_fed_data(fed_code)

    # Convert to percentages
    tri_party_general_collateral_rate.loc[
        :, tri_party_general_collateral_rate.columns != "Volume ($Billions)"
    ] = (
        tri_party_general_collateral_rate.loc[
            :, tri_party_general_collateral_rate.columns != "Volume ($Billions)"
        ]
        / 100
    )

    return tri_party_general_collateral_rate


def get_broad_general_collateral_rate() -> pd.DataFrame:
    """
    Get the Broad General Collateral Rate from the Federal Reserve Bank of New York over
    time. The Broad General Collateral Rate is a measure of rates on
    overnight Treasury general collateral repo transactions.

    Returns:
       pd.DataFrame: A DataFrame containing the Broad General Collateral Rate over time.
    """
    fed_code = CODES["BGCR"]

    broad_general_collateral_rate = collect_fed_data(fed_code)

    # Convert to percentages
    broad_general_collateral_rate.loc[
        :, broad_general_collateral_rate.columns != "Volume ($Billions)"
    ] = (
        broad_general_collateral_rate.loc[
            :, broad_general_collateral_rate.columns != "Volume ($Billions)"
        ]
        / 100
    )

    return broad_general_collateral_rate


def get_secured_overnight_financing_rate() -> pd.DataFrame:
    """
    Get the Secured Overnight Financing Rate (SOFR) from the Federal Reserve Bank of
    New York over time. The Secured Overnight Financing Rate is broad measure of
    the cost of borrowing cash overnight collateralized by Treasury securities.

    Returns:
       pd.DataFrame: A DataFrame containing the Secured Overnight Financing Rate over time.
    """
    fed_code = CODES["SOFR"]

    secured_overnight_financing_rate = collect_fed_data(fed_code)

    # Convert to percentages
    secured_overnight_financing_rate.loc[
        :, secured_overnight_financing_rate.columns != "Volume ($Billions)"
    ] = (
        secured_overnight_financing_rate.loc[
            :, secured_overnight_financing_rate.columns != "Volume ($Billions)"
        ]
        / 100
    )

    return secured_overnight_financing_rate
