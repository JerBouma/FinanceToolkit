"""Deutsche Bundesbank Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_ranged_data, require_columns
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://api.statistiken.bundesbank.de/rest/data/BBSIS/"

# The yields of the term structure the Bundesbank estimates daily from listed federal
# securities, one series per residual maturity from 1 to 30 years (R01XX to R30XX).
MATURITIES = {f"R{years:02d}XX": f"{years}Y" for years in range(1, 31)}
SERIES_KEY = "D.I.ZST.ZI.EUR.S1311.B.A604.{maturities}.R.A.A._Z._Z.A"


def get_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily German government bond yield curve, the term structure the
    Bundesbank estimates from listed federal securities, for maturities of 1 to 30 years,
    in a single request. Only the days that are not cached yet are requested.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The yields as decimals, indexed by business day with a column per
        maturity ("1Y" to "30Y").
    """
    description = "German government bond yield curve"
    series_key = SERIES_KEY.format(maturities="+".join(MATURITIES))

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{BASE_URL}{series_key}?format=sdmx_csv&lang=en"
            f"&startPeriod={fetch_start}&endPeriod={fetch_end}",
            timeout=120,
            extra_headers={"Accept": "text/csv"},
        )
        text = response.content.decode("utf-8-sig")

        if not text.strip():
            return pd.DataFrame()

        # The separator follows the language: a comma in English, a semicolon in German.
        header = text.splitlines()[0]
        data = pd.read_csv(
            io.StringIO(text), sep=";" if header.count(";") > header.count(",") else ","
        )
        require_columns(
            data, {"BBK_SEIS_MATURITY", "TIME_PERIOD", "OBS_VALUE"}, description
        )

        data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
        yield_curve = data.pivot(
            index="TIME_PERIOD", columns="BBK_SEIS_MATURITY", values="OBS_VALUE"
        )
        yield_curve = yield_curve.rename(columns=MATURITIES)
        yield_curve = yield_curve[
            [label for label in MATURITIES.values() if label in yield_curve]
        ]
        yield_curve.index = pd.PeriodIndex(yield_curve.index, freq="D")
        yield_curve.index.name = None
        yield_curve.columns.name = None

        # The Bundesbank publishes the yields in percent; weekends are listed without a value.
        return yield_curve.dropna(how="all").sort_index() / 100

    return collect_ranged_data(
        source=policy_model.BUNDESBANK,
        dataset="series",
        entity="term_structure",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
