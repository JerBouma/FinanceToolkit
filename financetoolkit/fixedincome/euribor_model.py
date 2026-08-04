"""Euribor Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.fixedincome.helpers import collect_ecb_data


def get_euribor_rate(maturity: str, nominal: bool = True) -> pd.DataFrame:
    """
    Get the Euribor rate for a given maturity from the European Central Bank over
    time. The Euribor (Euro Interbank Offered Rate) is the rate at which a panel of
    European banks lend unsecured funds to one another in the interbank market.

    Args:
        maturity (str): The maturity of the Euribor rate to collect. Must be one of
            '1M', '3M', '6M' or '1Y'.
        nominal (bool): Selects which ECB series is queried. True uses the standard
            published Euribor series (``RT.MM.EURIBOR<maturity>D_``); False uses the
            ECB's ``4F.MM.R_EURIBOR<maturity>D_`` variant. Defaults to True.

    Returns:
       pd.DataFrame: A DataFrame containing the Euribor rate over time.

    Raises:
        ValueError: If maturity is not one of '1M', '3M', '6M' or '1Y'.
    """
    if maturity not in ["1M", "3M", "6M", "1Y"]:
        raise ValueError("Invalid maturity. Please choose from 1M, 3M, 6M, 1Y.")

    if nominal:
        ecb_data_string = f"M.U2.EUR.RT.MM.EURIBOR{maturity}D_.HSTA"
    else:
        ecb_data_string = f"M.U2.EUR.4F.MM.R_EURIBOR{maturity}D_.HSTA"

    euribor_rate = collect_ecb_data(
        ecb_data_string=ecb_data_string, dataset="FM", frequency="M"
    )

    # Divide by 100 to convert the percentage into a decimal
    euribor_rate = euribor_rate / 100

    return euribor_rate
