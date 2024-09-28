"""ECB Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.fixedincome.helpers import collect_ecb_data


def get_main_refinancing_operations() -> pd.DataFrame:
    """
    Get the Main Refinancing Operations from the European Central Bank over
    time. The Main Refinancing Operations are the rate at which banks can
    borrow money from the central bank for the duration of one week.

    Returns:
       pd.DataFrame: A DataFrame containing the Main Refinancing Operations over time.
    """
    ecb_data_string = "D.U2.EUR.4F.KR.MRR_RT.LEV"

    main_refinancing_operations = collect_ecb_data(
        ecb_data_string=ecb_data_string, dataset="FM", frequency="D"
    )

    # Convert to percentage
    main_refinancing_operations = main_refinancing_operations / 100

    return main_refinancing_operations


def get_marginal_lending_facility() -> pd.DataFrame:
    """
    Get the Marginal Lending Facility from the European Central Bank over
    time. The Marginal Lending Facility is the interest rate which banks may
    use to obtain overnight liquidity with the Eurosystem.

    Args:
        per_capita (bool): Whether to return the per capita data or the total data.

    Returns:
       pd.DataFrame: A DataFrame containing the Marginal Lending Facility over time.
    """
    ecb_data_string = "D.U2.EUR.4F.KR.MLFR.LEV"

    marginal_lending_facility = collect_ecb_data(
        ecb_data_string=ecb_data_string, dataset="FM", frequency="D"
    )

    # Convert to percentage
    marginal_lending_facility = marginal_lending_facility / 100

    return marginal_lending_facility


def get_deposit_facility() -> pd.DataFrame:
    """
    Get the Deposit Facility from the European Central Bank over
    time. The Deposit Facility is the interest rate which banks may
    use to make overnight deposits with the Eurosystem.

    Returns:
       pd.DataFrame: A DataFrame containing the Deposit Facility over time.
    """
    ecb_data_string = "D.U2.EUR.4F.KR.DFR.LEV"

    deposit_facility = collect_ecb_data(
        ecb_data_string=ecb_data_string, dataset="FM", frequency="D"
    )

    # Convert to percentage
    deposit_facility = deposit_facility / 100

    return deposit_facility
