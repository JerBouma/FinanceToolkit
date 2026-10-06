"""European Insurance and Occupational Pensions Authority (EIOPA) Model"""

__docformat__ = "google"

import io
import re
import zipfile

import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.requests_model import get_request

logger = get_logger()

BASE_URL = "https://www.eiopa.europa.eu"

# The page that links every monthly release published since January 2023. A release is a
# zip file whose link carries a random identifier, so the links are read from this page.
RELEASES_PAGE = f"{BASE_URL}/tools-and-data/risk-free-interest-rate-term-structures_en"
RELEASE_PATTERN = re.compile(r'href="([^"]*?filename=EIOPA_RFR_(\d{8})\.zip)"')

# The worksheets of the term structures workbook, per curve.
CURVES = {
    "spot_no_va": "RFR_spot_no_VA",
    "spot_with_va": "RFR_spot_with_VA",
    "shock_up": "Spot_NO_VA_shock_UP",
    "shock_down": "Spot_NO_VA_shock_DOWN",
}

# The names EIOPA gives the currencies and countries that differ from the ones used
# elsewhere in the Finance Toolkit.
NAMES = {"Euro": "Euro Area", "Czechia": "Czech Republic"}

# The row of the worksheet with the names, and the first row with a maturity.
NAME_ROW = 1
FIRST_MATURITY_ROW = 10


def get_releases() -> pd.DataFrame:
    """
    Lists the monthly releases EIOPA publishes, with the link to each.

    Returns:
        pd.DataFrame: The link per release, indexed by the month the release is for.

    Raises:
        ValueError: When the page links no releases, which means it changed.
    """
    description = "EIOPA risk-free rate releases"

    def fetch() -> pd.DataFrame:
        page = get_request(RELEASES_PAGE, timeout=60).text
        releases = {
            pd.Period(pd.Timestamp(date), freq="M"): link
            for link, date in RELEASE_PATTERN.findall(page)
        }

        if not releases:
            raise ValueError(
                f"The {description} page links no releases, which means EIOPA changed it."
            )

        frame = pd.DataFrame({"Link": list(releases.values())}, index=list(releases))
        frame.index = pd.PeriodIndex(frame.index, freq="M")

        return frame.sort_index()

    return collect_cached_data(
        source=policy_model.EIOPA,
        dataset="releases",
        entity="releases",
        fetch=fetch,
        description=description,
    )


def _parse_term_structures(
    workbook: bytes, sheet: str, description: str
) -> pd.DataFrame:
    """
    Reads one curve from the term structures workbook of a release.

    Args:
        workbook (bytes): The workbook.
        sheet (str): The worksheet of the curve, e.g. "RFR_spot_no_VA".
        description (str): What is read, used in the error message.

    Returns:
        pd.DataFrame: The rates as decimals, indexed by maturity in years with a column per
        currency or country.

    Raises:
        ValueError: When the worksheet is missing or has no maturities where expected.
    """
    try:
        data = pd.read_excel(io.BytesIO(workbook), sheet_name=sheet, header=None)
    except ValueError as error:
        raise ValueError(
            f"The {description} has no worksheet '{sheet}', which means EIOPA changed its layout."
        ) from error

    names = data.iloc[NAME_ROW, 2:]
    rates = data.iloc[FIRST_MATURITY_ROW:, 1:]
    rates = rates[pd.to_numeric(rates.iloc[:, 0], errors="coerce").notna()]

    if rates.empty:
        raise ValueError(
            f"The {description} has no maturities where they are expected, which means EIOPA "
            "changed its layout."
        )

    curve = rates.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    curve.index = rates.iloc[:, 0].astype(int).to_numpy()
    curve.columns = [NAMES.get(str(name).strip(), str(name).strip()) for name in names]

    return curve.loc[:, [column for column in curve.columns if column != "nan"]]


def _get_release(month: pd.Period, link: str, curve: str) -> pd.DataFrame:
    """
    Retrieves one curve of one monthly release. A published release does not change, so
    it is cached for a year.

    Args:
        month (pd.Period): The month of the release.
        link (str): The link to the release's zip file.
        curve (str): The curve, a key of CURVES.

    Returns:
        pd.DataFrame: The rates, indexed by maturity in years with a column per currency or
        country.
    """
    description = f"EIOPA risk-free rates of {month}"

    def fetch() -> pd.DataFrame:
        archive = zipfile.ZipFile(
            io.BytesIO(get_request(f"{BASE_URL}{link}", timeout=180).content)
        )
        workbook = next(
            (
                name
                for name in archive.namelist()
                if name.endswith("_Term_Structures.xlsx")
            ),
            None,
        )

        if workbook is None:
            raise ValueError(
                f"The {description} have no term structures workbook, which means EIOPA "
                "changed the release."
            )

        return _parse_term_structures(
            archive.read(workbook), CURVES[curve], description
        )

    return collect_cached_data(
        source=policy_model.EIOPA,
        dataset="release",
        entity=f"{month}/{curve}",
        fetch=fetch,
        description=description,
    )


def get_risk_free_rate_term_structures(
    curve: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Retrieves the risk-free rate term structures EIOPA publishes monthly for Solvency II,
    for every currency and country it covers and maturities of 1 to 150 years. Only the
    releases between the start and end date are downloaded.

    Args:
        curve (str): "spot_no_va", "spot_with_va", "shock_up" or "shock_down".
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The rates as decimals, indexed by the month of the release with a
        column per currency or country and maturity.
    """
    releases = get_releases()

    if releases.empty:
        return pd.DataFrame()

    selected = releases.loc[pd.Period(start_date, "M") : pd.Period(end_date, "M")]

    if selected.empty:
        logger.warning(
            "EIOPA publishes its risk-free rates monthly from %s; there is no release between "
            "%s and %s.",
            releases.index[0],
            start_date,
            end_date,
        )
        return pd.DataFrame()

    curves = helpers.run_in_parallel(
        _get_release,
        [(month, link, curve) for month, link in selected["Link"].items()],
        max_workers=4,
    )

    frames = {
        month: frame.stack()
        for month, frame in zip(selected.index, curves, strict=True)
        if not frame.empty
    }

    if not frames:
        return pd.DataFrame()

    term_structures = pd.DataFrame(frames).T
    term_structures.columns = term_structures.columns.swaplevel(0, 1)
    term_structures = term_structures.sort_index(
        axis=1, level=[0, 1], sort_remaining=False
    )

    # Maturities in years, labelled like the other curves of the Finance Toolkit.
    term_structures.columns = pd.MultiIndex.from_tuples(
        [(country, f"{maturity}Y") for country, maturity in term_structures.columns],
        names=["Country", "Maturity"],
    )
    term_structures.index = pd.PeriodIndex(term_structures.index, freq="M")

    return term_structures
