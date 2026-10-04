"""National Bureau of Economic Research (NBER) Model"""

__docformat__ = "google"

from datetime import datetime

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

# The business cycle peaks and troughs the NBER's Business Cycle Dating Committee
# determined, the chronology FRED's USREC series is built from.
URL = "https://data.nber.org/data/cycles/business_cycle_dates.json"

# The monthly series starts with the first full cycle the chronology covers.
FIRST_MONTH = "1854-12"

COUNTRY = "United States"


def get_recession_indicator() -> pd.DataFrame:
    """
    Builds the monthly US recession indicator from the NBER business cycle dates, the
    way FRED builds USREC: 1 from the month after a peak through the trough month, and 0
    otherwise. A recession without a trough yet runs through the current month, and the
    first trough, whose peak predates the chronology, counts from the first month.

    Returns:
        pd.DataFrame: A single "United States" column of 1 and 0, indexed by the first day
        of each month ("Date") as FRED dates it.

    Raises:
        ValueError: When the response is not a list of peaks and troughs.
    """
    description = "NBER business cycle dates"

    def fetch() -> pd.DataFrame:
        cycles = get_request(URL, timeout=60).json()

        if not isinstance(cycles, list) or not all(
            isinstance(cycle, dict) and {"peak", "trough"}.issubset(cycle)
            for cycle in cycles
        ):
            raise ValueError(
                f"The {description} are not a list of peaks and troughs, which means the NBER "
                "changed the format."
            )

        months = pd.period_range(
            FIRST_MONTH, datetime.now().strftime("%Y-%m"), freq="M"
        )
        indicator = pd.Series(0, index=months)

        for cycle in cycles:
            # The recession starts in the month after the peak. The chronology opens with
            # a trough whose peak lies before it, so that recession starts with the series.
            first = (
                pd.Timestamp(cycle["peak"]) + pd.DateOffset(months=1)
                if cycle["peak"]
                else months[0].start_time
            )
            last = (
                pd.Timestamp(cycle["trough"])
                if cycle["trough"]
                else months[-1].start_time
            )
            indicator.loc[first.strftime("%Y-%m") : last.strftime("%Y-%m")] = 1

        index = pd.PeriodIndex(indicator.index.asfreq("D", how="start"), name="Date")

        return pd.DataFrame({COUNTRY: indicator.to_numpy()}, index=index)

    return collect_cached_data(
        source=policy_model.NATIONAL_BUREAU_OF_ECONOMIC_RESEARCH,
        dataset="business_cycle_dates",
        entity="usrec",
        fetch=fetch,
        description=description,
    )
