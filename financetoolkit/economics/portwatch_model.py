"""IMF PortWatch Chokepoint Transits Model"""

__docformat__ = "google"

from urllib.parse import urlencode

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

# IMF PortWatch publishes its daily chokepoint data as an ArcGIS feature service, which
# answers at most 1,000 rows per request, so a chokepoint's history (one row per day from
# 2019) is read in pages.
QUERY_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)
PAGE_SIZE = 1000

# The ship categories, as PortWatch counts them (n_*) and estimates their trade volume
# (capacity_*), with the names the Finance Toolkit gives them. Cargo is every category
# except tankers, Total all five.
VESSEL_TYPES = {
    "container": "Container",
    "dry_bulk": "Dry Bulk",
    "general_cargo": "General Cargo",
    "roro": "Ro-Ro",
    "tanker": "Tanker",
    "cargo": "Cargo",
    "total": "Total",
}


def _query(parameters: dict) -> dict:
    """
    Answers a query of the chokepoint feature service.

    Args:
        parameters (dict): The query parameters, without the format.

    Returns:
        dict: The JSON answer.

    Raises:
        ValueError: When the service answers with an error, which means it changed.
    """
    response = get_request(
        f"{QUERY_URL}?{urlencode({**parameters, 'f': 'json'})}", timeout=60
    ).json()

    if "error" in response:
        raise ValueError(
            f"IMF PortWatch answered the chokepoint query with an error: "
            f"{response['error'].get('message', response['error'])}"
        )

    return response


def get_chokepoint_names() -> list[str]:
    """
    Returns the names of the chokepoints IMF PortWatch covers, such as "Strait of Hormuz",
    "Suez Canal" and "Panama Canal".

    Returns:
        list[str]: The names, sorted.

    Raises:
        ValueError: When the service lists no chokepoints, which means it changed.
    """
    description = "IMF PortWatch chokepoint names"

    def fetch() -> pd.DataFrame:
        response = _query(
            {
                "where": "1=1",
                "outFields": "portname",
                "returnDistinctValues": "true",
                "returnGeometry": "false",
            }
        )
        names = sorted(
            feature["attributes"]["portname"]
            for feature in response.get("features", [])
        )

        if not names:
            raise ValueError(
                f"The {description} came back empty, which means the service changed."
            )

        return pd.DataFrame({"Chokepoint": names})

    names = collect_cached_data(
        source=policy_model.IMF_PORTWATCH,
        dataset="chokepoints",
        entity="names",
        fetch=fetch,
        description=description,
    )

    return names["Chokepoint"].tolist() if not names.empty else []


def get_chokepoint_transits(chokepoint: str) -> pd.DataFrame:
    """
    Retrieves IMF PortWatch's daily transits of one chokepoint from 2019: the number of
    ships that passed it each day (UTC) and the estimated trade volume they carried in
    metric tons, by ship category.

    A ship is counted when it crosses the chokepoint's boundary, once per transit, and not
    again within 48 hours. The trade volume is each ship's deadweight tonnage times its
    estimated load factor, read from its draft in the AIS signals.

    Args:
        chokepoint (str): The chokepoint as PortWatch names it, e.g. "Strait of Hormuz".

    Returns:
        pd.DataFrame: Indexed by day, with the columns "Transits <category>" and "Trade
        Volume <category>" for each category of VESSEL_TYPES.

    Raises:
        ValueError: When the answer lacks the expected fields, which means it changed.
    """
    description = f"IMF PortWatch transits of the {chokepoint}"
    fields = (
        ["date"]
        + [
            f"{prefix}_{category}"
            for prefix in ("n", "capacity")
            for category in VESSEL_TYPES
            if not (prefix == "capacity" and category == "total")
        ]
        + ["capacity"]
    )

    def fetch() -> pd.DataFrame:
        rows: list[dict] = []
        escaped = chokepoint.replace("'", "''")

        while True:
            response = _query(
                {
                    "where": f"portname='{escaped}'",
                    "outFields": ",".join(fields),
                    "orderByFields": "date",
                    "resultOffset": len(rows),
                    "resultRecordCount": PAGE_SIZE,
                    "returnGeometry": "false",
                }
            )
            features = response.get("features", [])
            rows.extend(feature["attributes"] for feature in features)

            if not features or not response.get("exceededTransferLimit"):
                break

        if not rows:
            return pd.DataFrame()

        data = pd.DataFrame(rows)
        missing = [field for field in fields if field not in data.columns]

        if missing:
            raise ValueError(
                f"The {description} lack the fields {missing}, which means the service "
                "changed."
            )

        data.index = pd.PeriodIndex(data.pop("date"), freq="D")
        data.index.name = None
        data = data.rename(columns={"capacity": "capacity_total"})
        data.columns = [
            ("Transits " if column.startswith("n_") else "Trade Volume ")
            + VESSEL_TYPES[column.split("_", 1)[1]]
            for column in data.columns
        ]

        return data[~data.index.duplicated(keep="last")].sort_index().astype(float)

    return collect_cached_data(
        source=policy_model.IMF_PORTWATCH,
        dataset="chokepoint_transits",
        entity=chokepoint.lower().replace(" ", "_"),
        fetch=fetch,
        description=description,
    )
