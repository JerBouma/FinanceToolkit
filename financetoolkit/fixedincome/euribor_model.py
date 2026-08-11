"""Euribor Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.fixedincome.helpers import collect_ecb_data


def get_euribor_rate(maturity: str, nominal: bool = True) -> pd.DataFrame:
    """
    Get the Euribor rate for a given maturity from the European Central Bank over
    time. The Euribor (Euro Interbank Offered Rate) is the rate at which a panel of
    European banks lend unsecured funds to one another in the interbank market.

    The ECB publishes these as monthly observations in percent per annum (the series'
    own unit is PCPA), which this function converts into decimals. Euribor fixings went
    negative between 2015 and 2022 and the returned series reflects that faithfully, so
    nothing downstream may assume the rate is positive.

    Args:
        maturity (str): The maturity of the Euribor rate to collect. Must be one of
            '1M', '3M', '6M' or '1Y'. Only '3M' is available when nominal is False.
        nominal (bool): Selects which ECB series is queried. True uses the standard
            published Euribor series (``RT.MM.EURIBOR<maturity>D_``, titled e.g.
            "Euribor 3-month - Historical close, average of observations through
            period"). False uses the ECB's inflation-adjusted counterpart
            (``4F.MM.R_EURIBOR3MD_``, titled "Real 3-month Euribor"), which the ECB only
            publishes for the 3-month maturity. Defaults to True.

    Returns:
       pd.DataFrame: A DataFrame containing the Euribor rate over time.

    Raises:
        ValueError: If maturity is not one of '1M', '3M', '6M' or '1Y', or if a real
            rate is requested for a maturity other than '3M'.
    """
    if maturity not in ["1M", "3M", "6M", "1Y"]:
        raise ValueError("Invalid maturity. Please choose from 1M, 3M, 6M, 1Y.")

    # The ECB only publishes a real (inflation-adjusted) Euribor for the 3-month tenor;
    # the other three series do not exist and would come back as a bare HTTP 404.
    if not nominal and maturity != "3M":
        raise ValueError(
            f"The European Central Bank only publishes a real Euribor rate for the 3M "
            f"maturity, so no real rate is available for {maturity}. Request the 3M "
            "maturity instead, or set nominal=True."
        )

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
