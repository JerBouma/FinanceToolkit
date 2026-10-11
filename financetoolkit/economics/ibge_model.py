"""Brazilian Institute of Geography and Statistics (IBGE) Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

# The SIDRA API of the IBGE: table 6381 is the monthly release of the continuous
# national household sample survey (PNAD Contínua), variable 4099 its unemployment rate.
UNEMPLOYMENT_URL = "https://apisidra.ibge.gov.br/values/t/6381/n1/all/v/4099/p/all"

COUNTRY = "Brazil"


def get_unemployment_rate() -> pd.DataFrame:
    """
    Retrieves the Brazilian unemployment rate for people aged 14 and over from the
    PNAD Contínua survey, from 2012. Each month is the average of the three months ending
    in it, as the IBGE reports it.

    Returns:
        pd.DataFrame: A single "Brazil" column with the rate as a decimal fraction of the
        labour force (0.053 for 5.3%), indexed by month.

    Raises:
        ValueError: When the response has no observations in the expected format.
    """
    description = "Brazilian unemployment rate"

    def fetch() -> pd.DataFrame:
        # SIDRA negotiates the format and answers the default browser Accept header with XML.
        rows = get_request(
            UNEMPLOYMENT_URL, timeout=60, extra_headers={"Accept": "application/json"}
        ).json()

        # The first row holds the column names; every other row is one rolling quarter,
        # identified by its last month in D3C (202608 for June to August 2026).
        observations = rows[1:] if isinstance(rows, list) else []

        if not observations or not all(
            isinstance(row, dict) and {"D3C", "V"}.issubset(row) for row in observations
        ):
            raise ValueError(
                f"The IBGE response for the {description} has no observations in the "
                "expected format, which means the SIDRA API changed."
            )

        index = pd.PeriodIndex(
            pd.to_datetime([row["D3C"] for row in observations], format="%Y%m"),
            freq="M",
        )
        values = pd.to_numeric(
            pd.Series([row["V"] for row in observations]), errors="coerce"
        )

        # The IBGE publishes the rate in percent of the labour force.
        return pd.DataFrame(
            {COUNTRY: values.to_numpy() / 100}, index=index
        ).sort_index()

    return collect_cached_data(
        source=policy_model.IBGE,
        dataset="series",
        entity="6381/4099",
        fetch=fetch,
        description=description,
    )
