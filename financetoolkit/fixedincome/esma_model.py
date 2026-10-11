"""European Securities and Markets Authority (ESMA) Model"""

__docformat__ = "google"

import io
import json

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import HEADERS, SESSION

# CEREP, ESMA's central repository of the ratings of the credit rating agencies registered
# in the EU, publishes their rating activity, default rates and transition matrices twice a
# year, from 1989. Its web page queries these endpoints with a JSON body of filters.
BASE_URL = "https://registers.esma.europa.eu/cerep-publication/"

# The rating agencies by common name; the CEREP code of any other agency is accepted too.
AGENCIES = {
    "S&P": "STPGB",
    "Moody's": "MDYGB",
    "Fitch": "FITGB",
    "DBRS": "DBRGB",
    "Scope": "PSRDE",
    "Kroll": "KBRUS",
    "Creditreform": "CRRDE",
    "A.M. Best": "AMBGB",
    "Japan Credit Rating": "JCRFR",
}

RATING_TYPES = {
    "corporate": "C",
    "sovereign": "S",
    "structured_finance": "T",
    "covered_bonds": "B",
}

# The statistics count ratings by category (AAA, AA, ...) rather than notch (AA+, AA-).
CATEGORY_FILTERS = {
    "filterList": [{"code": "C", "selected": True}, {"code": "N", "selected": True}]
}

# The CEREP returns this text instead of a file when a query is not valid.
INVALID_QUERY = "ERROR"


def _period_code(date: pd.Timestamp) -> str:
    """
    Converts the start or end of a period into the code CEREP uses for it: the date's
    midnight in UTC in milliseconds since 1970.

    Args:
        date (pd.Timestamp): The date.

    Returns:
        str: The code.
    """
    return str(int(pd.Timestamp(date.date(), tz="UTC").timestamp() * 1000))


def _query(
    export: str,
    agency: str,
    rating_type: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    region: str | None,
) -> pd.DataFrame:
    """
    Retrieves one CEREP statistic as a table.

    Args:
        export (str): "Transition-Matrix" or "Default-Rate".
        agency (str): The CEREP code of the rating agency, e.g. "STPGB".
        rating_type (str): The CEREP code of the rating type, e.g. "C".
        period_start (pd.Timestamp): The start of the period (1 January or 1 July).
        period_end (pd.Timestamp): The end of the period (30 June or 31 December).
        region (str | None): The geographical area, e.g. "Europe", or None for all.

    Returns:
        pd.DataFrame: The statistic as published.
    """
    filters = {
        "categories": CATEGORY_FILTERS,
        "number": CATEGORY_FILTERS,
        "cra": {"filterList": [{"code": agency, "selected": True}]},
        "ratingType": {"filterList": [{"code": rating_type, "selected": True}]},
        "timeHorizon": {"filterList": [{"code": "L", "selected": True}]},
        "begOfPrd": {
            "filterList": [{"code": _period_code(period_start), "selected": True}]
        },
        "endOfPrd": {
            "filterList": [{"code": _period_code(period_end), "selected": True}]
        },
    }
    if region:
        filters["geoArea"] = {"filterList": [{"code": region, "selected": True}]}

    description = f"ESMA CEREP {export.replace('-', ' ').lower()} of {agency}"

    def fetch() -> pd.DataFrame:
        response = SESSION.post(
            f"{BASE_URL}exportCsv/{export}",
            data=json.dumps({"filters": filters}),
            headers={**HEADERS, "Content-Type": "application/json", "Accept": "*/*"},
            timeout=60,
        )
        response.raise_for_status()
        text = response.content.decode("utf-8-sig", errors="replace")

        # A period without ratings, or a combination CEREP does not publish.
        if not text.strip() or text.startswith(INVALID_QUERY):
            return pd.DataFrame()

        return pd.read_csv(io.StringIO(text), sep=";", index_col=0)

    return collect_cached_data(
        source=policy_model.ESMA,
        dataset="statistics",
        entity=(
            f"{export}/{agency}/{rating_type}/{region or 'all'}/"
            f"{period_start.date()}/{period_end.date()}"
        ),
        fetch=fetch,
        description=description,
        # The periods before an agency reported to CEREP have no statistics.
        expect_empty=True,
    )


def resolve_agency(agency: str) -> str:
    """
    Returns the CEREP code of a rating agency given by common name or code.

    Args:
        agency (str): E.g. "S&P", "Moody's" or "STPGB".

    Returns:
        str: The CEREP code.

    Raises:
        ValueError: When the agency is not known by that name and is not a code.
    """
    if agency in AGENCIES:
        return AGENCIES[agency]
    if agency.isalnum() and agency.isupper() and len(agency) == len("STPGB"):
        return agency

    raise ValueError(
        f"The agency must be one of {', '.join(map(repr, AGENCIES))} or a CEREP code such as "
        f"'STPGB', not {agency!r}."
    )


def get_transition_matrix(
    agency: str,
    rating_type: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    region: str | None = None,
) -> pd.DataFrame:
    """
    Retrieves the number of ratings that moved from each rating category at the start of a
    period to each category at its end, including defaults and withdrawals.

    Args:
        agency (str): The CEREP code of the rating agency, e.g. "STPGB".
        rating_type (str): A key of RATING_TYPES, e.g. "corporate".
        period_start (pd.Timestamp): The start of the period (1 January or 1 July).
        period_end (pd.Timestamp): The end of the period (30 June or 31 December).
        region (str | None): The geographical area, e.g. "Europe". Defaults to None, which
            covers all areas.

    Returns:
        pd.DataFrame: The counts, with a row per rating at the start of the period and a
        column per rating (and "Withdrawals") at its end.
    """
    matrix = _query(
        "Transition-Matrix",
        agency,
        RATING_TYPES[rating_type],
        period_start,
        period_end,
        region,
    )

    if not matrix.empty:
        matrix.index.name = None
        matrix = matrix.apply(pd.to_numeric, errors="coerce")

    return matrix


def get_default_rates(
    agency: str,
    rating_type: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    region: str | None = None,
) -> pd.DataFrame:
    """
    Retrieves the number of defaults and the default rate per rating category over a
    period.

    Args:
        agency (str): The CEREP code of the rating agency, e.g. "STPGB".
        rating_type (str): A key of RATING_TYPES, e.g. "corporate".
        period_start (pd.Timestamp): The start of the period (1 January or 1 July).
        period_end (pd.Timestamp): The end of the period (30 June or 31 December).
        region (str | None): The geographical area, e.g. "Europe". Defaults to None, which
            covers all areas.

    Returns:
        pd.DataFrame: The "Defaults" and the "Default Rate" as a decimal, per rating.
    """
    rates = _query(
        "Default-Rate",
        agency,
        RATING_TYPES[rating_type],
        period_start,
        period_end,
        region,
    )

    if rates.empty:
        return rates

    rates.index.name = None
    rates.columns = ["Defaults", "Default Rate"]
    rates["Defaults"] = pd.to_numeric(rates["Defaults"], errors="coerce")
    rates["Default Rate"] = (
        pd.to_numeric(
            rates["Default Rate"].astype(str).str.rstrip("%"), errors="coerce"
        )
        / 100
    )

    return rates
