"""De Nederlandsche Bank (DNB) Model"""

__docformat__ = "google"

import io
import re
import zipfile

import numpy as np
import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request

# DNB publishes, every quarter, the scenario sets Dutch pension funds must use for their
# feasibility test and the transition to the new pension contract: 20,000 scenarios over
# 100 years of the economic scenario generator the Commissie Parameters 2022 specified,
# under the real-world measure (P) and the market-consistent measure (Q).
BASE_URL = "https://www.dnb.nl"
LISTING_URL = (
    f"{BASE_URL}/voor-de-sector/open-boek-toezicht/sectoren/pensioenfondsen/"
    "dnb-publiceert-definitieve-scenariosets-bij-wet-toekomst-pensioenen/"
)
SCENARIO_SET_PATTERN = (
    r'href="(/media/[^"]+/cp2022-{measure}-scenarioset-20k-(\d{{4}})q(\d)\.xlsx)"'
)

MEASURES = {"real_world": "p", "market_consistent": "q"}

# The worksheets: the three state variables of the model (from which the term structures
# follow), the equity return and the price inflation of the euro area and the Netherlands,
# and the parameters of the nominal and euro area real term structures.
STATE_SHEETS = [
    "1_Toestandsvariabele_1",
    "2_Toestandsvariabele_2",
    "3_Toestandsvariabele_3",
]
RETURN_SHEETS = {
    "Equity return": "4_Aandelenrendement",
    "Inflation EU": "5_Prijsinflatie_EU",
    "Inflation NL": "6_Prijsinflatie_NL",
}
NOMINAL_PHI, NOMINAL_PSI = "7_Renteparameter_phi_N", "8_Renteparameter_Psi_N"
REAL_PHI, REAL_PSI = "9_Renteparameter_phi_R_EU", "10_Renteparameter_Psi_R"

# The maturities of the zero rates derived from the term structures.
NOMINAL_MATURITIES = [1, 5, 10, 20, 30]
REAL_MATURITIES = [10]

ROW_PATTERN = re.compile(rb"<row [^>]*>(.*?)</row>", re.S)
VALUE_PATTERN = re.compile(rb"<v>([^<]*)</v>")


def get_scenario_set_link(measure: str) -> tuple[str, str]:
    """
    Finds the latest scenario set of a measure on DNB's page.

    Args:
        measure (str): "real_world" or "market_consistent".

    Returns:
        tuple[str, str]: The link and the quarter it is for (e.g. "2026Q3").

    Raises:
        ValueError: When the page links no scenario set, which means it changed.
    """
    page = get_request(LISTING_URL, timeout=60, extra_headers=TOOLKIT_HEADERS).text
    links = re.findall(SCENARIO_SET_PATTERN.format(measure=MEASURES[measure]), page)

    if not links:
        raise ValueError("DNB's page links no scenario set, which means it changed.")

    link, year, quarter = max(links, key=lambda found: (found[1], found[2]))

    return f"{BASE_URL}{link}", f"{year}Q{quarter}"


def _sheet_paths(workbook: zipfile.ZipFile) -> dict[str, str]:
    """
    Maps the worksheet names of a workbook to their files within it.

    Args:
        workbook (zipfile.ZipFile): The workbook.

    Returns:
        dict[str, str]: The file per worksheet name.
    """
    # The attributes of a sheet can come in any order.
    sheets = [
        (name.group(1), identifier.group(1))
        for tag in re.findall(
            r"<sheet\b[^>]*>", workbook.read("xl/workbook.xml").decode()
        )
        if (name := re.search(r'\bname="([^"]+)"', tag))
        and (identifier := re.search(r'\br:id="(rId\d+)"', tag))
    ]
    targets = {
        identifier.group(1): target.group(1)
        for tag in re.findall(
            r"<Relationship\b[^>]*>",
            workbook.read("xl/_rels/workbook.xml.rels").decode(),
        )
        if (identifier := re.search(r'\bId="(rId\d+)"', tag))
        and (target := re.search(r'\bTarget="([^"]+)"', tag))
    }

    return {
        name: "xl/" + targets[identifier].lstrip("/").removeprefix("xl/")
        for name, identifier in sheets
        if identifier in targets
    }


def _read_sheet(workbook: zipfile.ZipFile, path: str) -> np.ndarray:
    """
    Reads a worksheet of numbers only, row by row. The worksheets of the scenario sets
    are around 90 MB each, which a spreadsheet reader would take minutes over, while every
    cell holds a plain number, so the values are read straight from the XML.

    Args:
        workbook (zipfile.ZipFile): The workbook.
        path (str): The file of the worksheet within the workbook.

    Returns:
        np.ndarray: The values, a row per worksheet row.
    """
    xml = workbook.read(path)
    rows = [
        np.array(VALUE_PATTERN.findall(match.group(1)), dtype=float)
        for match in ROW_PATTERN.finditer(xml)
    ]

    return np.vstack(rows) if rows else np.empty((0, 0))


def _zero_rates(
    phi: np.ndarray, psi: np.ndarray, states: np.ndarray, maturity: int
) -> np.ndarray:
    """
    Computes the annually compounded zero rate of a maturity in every scenario and year
    from the affine term structure of the model: the discount factor is exp(phi + psi x).

    Args:
        phi (np.ndarray): The phi parameter per maturity (rows) and year (columns).
        psi (np.ndarray): The psi loadings per maturity (rows) and state variable.
        states (np.ndarray): The state variables, per state, scenario and year.
        maturity (int): The maturity in years.

    Returns:
        np.ndarray: The zero rates per scenario (rows) and year (columns).
    """
    exponent = phi[maturity - 1][None, :] + np.tensordot(
        psi[maturity - 1], states, axes=1
    )

    return np.exp(-exponent / maturity) - 1


def get_scenario_set(measure: str) -> pd.DataFrame:
    """
    Retrieves the latest scenario set of a measure: for every one of the 20,000 scenarios
    and every year of the 100-year horizon, the nominal zero rates, the euro area real zero
    rate, the equity return and the price inflation of the euro area and the Netherlands.

    Args:
        measure (str): "real_world" or "market_consistent".

    Returns:
        pd.DataFrame: The variables as decimals, indexed by the year of the horizon (0 is
        the end of the quarter the set is for) with a (variable, scenario) column.
    """
    link, quarter = get_scenario_set_link(measure)
    description = f"DNB {measure.replace('_', '-')} scenario set of {quarter}"

    def fetch() -> pd.DataFrame:
        workbook = zipfile.ZipFile(
            io.BytesIO(
                get_request(link, timeout=600, extra_headers=TOOLKIT_HEADERS).content
            )
        )
        paths = _sheet_paths(workbook)

        if missing := [
            sheet
            for sheet in [
                *STATE_SHEETS,
                *RETURN_SHEETS.values(),
                NOMINAL_PHI,
                NOMINAL_PSI,
            ]
            if sheet not in paths
        ]:
            raise ValueError(
                f"The {description} has no worksheet {', '.join(missing)}, which means its "
                "layout changed."
            )

        states = np.stack(
            [_read_sheet(workbook, paths[sheet]) for sheet in STATE_SHEETS]
        )
        horizon = states.shape[2]
        variables = {}

        phi, psi = (
            _read_sheet(workbook, paths[NOMINAL_PHI]),
            _read_sheet(workbook, paths[NOMINAL_PSI]),
        )
        for maturity in NOMINAL_MATURITIES:
            variables[f"Nominal rate {maturity}Y"] = _zero_rates(
                phi, psi, states, maturity
            )

        if REAL_PHI in paths and REAL_PSI in paths:
            phi, psi = (
                _read_sheet(workbook, paths[REAL_PHI]),
                _read_sheet(workbook, paths[REAL_PSI]),
            )
            for maturity in REAL_MATURITIES:
                variables[f"Real rate EU {maturity}Y"] = _zero_rates(
                    phi, psi, states, maturity
                )

        # Returns and inflation are over the years 1 to 100, so year 0 has none.
        for name, sheet in RETURN_SHEETS.items():
            values = _read_sheet(workbook, paths[sheet])
            variables[name] = np.hstack(
                [np.full((values.shape[0], horizon - values.shape[1]), np.nan), values]
            )

        scenarios = states.shape[1]
        # Stored as single precision: 8 variables of 20,000 scenarios over 101 years.
        frame = pd.DataFrame(
            np.hstack([values.T for values in variables.values()]).astype("float32"),
            index=pd.RangeIndex(horizon, name="Horizon"),
            columns=pd.MultiIndex.from_product(
                [list(variables), range(1, scenarios + 1)],
                names=["Variable", "Scenario"],
            ),
        )

        return frame

    return collect_cached_data(
        source=policy_model.DE_NEDERLANDSCHE_BANK,
        dataset="scenario_set",
        entity=f"{measure}/{quarter}",
        fetch=fetch,
        description=description,
    )
