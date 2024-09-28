"""ECB Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.fixedincome.helpers import collect_ecb_data


def get_euribor_rate(maturity: str, nominal: bool = True) -> pd.DataFrame:
    """
    Get the Main Refinancing Operations from the European Central Bank over
    time. The Main Refinancing Operations are the rate at which banks can
    borrow money from the central bank for the duration of one week.

    Returns:
       pd.DataFrame: A DataFrame containing the Main Refinancing Operations over time.
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

    # Convert to percentage
    euribor_rate = euribor_rate / 100

    return euribor_rate
