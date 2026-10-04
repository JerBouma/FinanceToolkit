"""European Central Bank (ECB) Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import (
    COUNTRY_CODES,
    collect_ranged_data,
    require_columns,
)
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://data-api.ecb.europa.eu/service/data/"

SERIES = {
    # Euro short-term rate (€STR), the overnight rate of the euro area.
    "overnight_rate": "EST/B.EU000A2X2A25.WT",
}

# The maturities of the euro area yield curve (all central government bonds), keyed by
# the ECB's code for the spot rate at that maturity.
YIELD_CURVE_MATURITIES = {
    "SR_3M": "3M",
    "SR_6M": "6M",
    "SR_1Y": "1Y",
    "SR_2Y": "2Y",
    "SR_3Y": "3Y",
    "SR_5Y": "5Y",
    "SR_7Y": "7Y",
    "SR_10Y": "10Y",
    "SR_15Y": "15Y",
    "SR_20Y": "20Y",
    "SR_30Y": "30Y",
}
YIELD_CURVE_KEY = "YC/B.U2.EUR.4F.G_N_C.SV_C_YM." + "+".join(YIELD_CURVE_MATURITIES)

COUNTRY = "Euro Area"


def collect_ecb_series(
    series_key: str,
    description: str,
    start_date: str,
    end_date: str,
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Retrieves the days between two dates of one or more ECB Data Portal series. Only the
    days that are not cached yet are requested.

    The portal answers roughly one in five queries it has not seen before with a gateway
    timeout, while a query it has answered recently comes back at once. Every request
    therefore starts on the first day of a month and has no end date, so the same query
    is repeated all month, by every user, and is usually served from the portal's cache.
    Several series are combined into one request with "+" in the series key.

    Args:
        series_key (str): The dataset and series key, e.g. "EST/B.EU000A2X2A25.WT", with
            "+" between the values of a dimension to request several series at once.
        description (str): What is retrieved, used in the log and error messages.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        columns (dict[str, str] | None): For a request of several series, the column name
            per series, keyed by the last part of the series key. Defaults to None, which
            returns a single "Euro Area" column.

    Returns:
        pd.DataFrame: The values as decimals, indexed by day.
    """

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        month_start = pd.Timestamp(fetch_start).strftime("%Y-%m-01")

        # Without detail=dataonly every observation carries its attributes, which makes
        # the €STR request run into the portal's 30-second gateway timeout.
        response = get_request(
            f"{BASE_URL}{series_key}?format=csvdata&detail=dataonly&startPeriod={month_start}",
            timeout=120,
        )

        # A range without observations is answered with an empty body.
        if not response.text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(response.text))
        require_columns(data, {"KEY", "TIME_PERIOD", "OBS_VALUE"}, description)

        data["DATE"] = pd.PeriodIndex(pd.to_datetime(data["TIME_PERIOD"]), freq="D")
        data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")

        if columns is None:
            frame = data.set_index("DATE")[["OBS_VALUE"]].rename(
                columns={"OBS_VALUE": COUNTRY}
            )
        else:
            data["COLUMN"] = data["KEY"].str.rsplit(".", n=1).str[-1].map(columns)
            frame = data.dropna(subset=["COLUMN"]).pivot(
                index="DATE", columns="COLUMN", values="OBS_VALUE"
            )
            frame = frame[
                [column for column in columns.values() if column in frame.columns]
            ]

        frame.index.name = None
        frame.columns.name = None
        frame = frame.sort_index()

        # The ECB publishes the rates in percent.
        return frame.loc[pd.Period(fetch_start, "D") : pd.Period(fetch_end, "D")] / 100

    return collect_ranged_data(
        source=policy_model.EUROPEAN_CENTRAL_BANK,
        dataset="economics_series",
        entity=series_key,
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


def get_long_term_interest_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily 10-year spot rate of the euro area government bond yield curve,
    estimated by the ECB from the bonds of all euro area central governments. It is
    taken from the yield curve request, so both share one query.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The yield as a decimal, indexed by business day.
    """
    yield_curve = get_yield_curve(start_date, end_date)

    if yield_curve.empty or "10Y" not in yield_curve.columns:
        return pd.DataFrame()

    return yield_curve[["10Y"]].rename(columns={"10Y": COUNTRY}).dropna()


def get_overnight_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily euro short-term rate (€STR).

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The rate as a decimal, indexed by business day.
    """
    return collect_ecb_series(
        SERIES["overnight_rate"], "euro short-term rate", start_date, end_date
    )


def get_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily euro area yield curve, the spot rates the ECB estimates from the
    bonds of all euro area central governments, from 3 months to 30 years, since 2004.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The spot rates as decimals, indexed by business day with a column per
        maturity ("3M" to "30Y").
    """
    return collect_ecb_series(
        YIELD_CURVE_KEY,
        "euro area yield curve",
        start_date,
        end_date,
        columns=YIELD_CURVE_MATURITIES,
    )


def get_long_term_convergence_yields(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the monthly long-term interest rates for convergence purposes, the average
    yield of a 10-year government bond of every European Union member, which the ECB
    publishes for the Maastricht criteria. Only the months that are not cached yet are
    requested.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The yields as decimals, indexed by month with a column per country.
    """
    description = "long-term interest rates for convergence purposes"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        # Monthly, so a request that starts at the beginning of a year repeats all year. The
        # currency is left open, since members outside the euro area borrow in their own.
        response = get_request(
            f"{BASE_URL}IRS/M..L.L40.CI.0000..N.Z?format=csvdata&detail=dataonly"
            f"&startPeriod={fetch_start[:4]}-01",
            timeout=120,
        )

        if not response.text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(response.text))
        require_columns(
            data,
            {"REF_AREA", "CURRENCY_TRANS", "TIME_PERIOD", "OBS_VALUE"},
            description,
        )

        # "EU" would be the euro area in the shared country names; it is not an issuer.
        data = data[
            data["REF_AREA"].isin(COUNTRY_CODES.keys()) & (data["REF_AREA"] != "EU")
        ]
        data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")

        # A member that adopted the euro, such as Croatia or Bulgaria, has a series in its
        # former currency next to the euro series for the months they overlap; the euro
        # series is kept, so every country is one column.
        data = data.sort_values(
            "CURRENCY_TRANS", key=lambda currency: currency != "EUR"
        ).drop_duplicates(subset=["TIME_PERIOD", "REF_AREA"], keep="first")
        yields = data.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
        yields = yields.rename(columns=COUNTRY_CODES)
        yields.index = pd.PeriodIndex(yields.index, freq="M")
        yields.index.name = None
        yields.columns.name = None
        yields = yields.sort_index()

        # The ECB publishes the yields in percent.
        return yields.loc[pd.Period(fetch_start, "M") : pd.Period(fetch_end, "M")] / 100

    return collect_ranged_data(
        source=policy_model.EUROPEAN_CENTRAL_BANK,
        dataset="convergence_yields",
        entity="IRS",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
