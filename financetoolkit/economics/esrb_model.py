"""European Systemic Risk Board (ESRB) Model"""

__docformat__ = "google"

import io
import re

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

# The ESRB designs the adverse scenarios of the EU-wide stress tests of the EBA (banks),
# EIOPA (insurers) and ESMA, and lists their workbooks on this page. The macro-financial
# scenario of the bank stress test is the one with the country panel.
BASE_URL = "https://www.esrb.europa.eu"
LISTING_URL = f"{BASE_URL}/mppa/stress/html/index.en.html"
MACRO_SCENARIO_PATTERN = re.compile(
    r'href="(/mppa/stress/shared/pdf/esrb\.stress_test(\d{6})\.macrofinancialscenario[^"]*\.xlsx)"',
    flags=re.IGNORECASE,
)

# The worksheets with a row per country, region or index and columns per scenario and
# year, the variable each holds, and the scale to a decimal: percentages are divided by
# 100 and basis points (iTraxx) by 10,000; exchange rates are kept.
SHEETS = {
    "GDP": ("Real GDP growth", 100),
    "HICP": ("Inflation", 100),
    "Unemployment": ("Unemployment rate", 100),
    "RRE prices": ("Residential real estate price growth", 100),
    "CRE prices": ("Commercial real estate price growth", 100),
    "Long-term rates": ("Long-term interest rate", 100),
    "Stock prices": ("Stock price deviation from starting point", 100),
    "ForeignDemandall": ("Level deviation from starting point", 100),
    "Itraxx": ("Credit spread", 10_000),
    "Exchange rates": ("Exchange rate", 1),
}

# The names that differ from the ones used elsewhere in the Finance Toolkit.
NAMES = {"Euro area": "Euro Area", "Türkiye": "Turkey"}

# How the column headers name the scenarios. A deviation from the starting point is only
# published for the adverse scenario, since the baseline keeps those variables unchanged.
SCENARIO_LABELS = {
    "historic": ("historical", "starting point rates (%) – average"),
    "baseline": ("baseline",),
    "adverse": (
        "adverse",
        "deviation from the starting point",
        "level deviation from starting point",
    ),
}


def get_macro_financial_scenario_link() -> tuple[str, str]:
    """
    Finds the latest macro-financial scenario workbook of the EU-wide bank stress test.

    Returns:
        tuple[str, str]: The link and the publication date (YYMMDD).

    Raises:
        ValueError: When the page links no such workbook.
    """
    page = get_request(LISTING_URL, timeout=60).text
    links = MACRO_SCENARIO_PATTERN.findall(page)

    if not links:
        raise ValueError(
            "The ESRB stress test page links no macro-financial scenario, which means it changed."
        )

    link, date = max(links, key=lambda found: found[1])

    return f"{BASE_URL}{link}", date


# The years a scenario column can be, which tells the year row from other numbers.
FIRST_YEAR, LAST_YEAR = 1990, 2100

# Country codes ("BE") and rating buckets ("H") sit next to the names, which are longer.
MINIMUM_LABEL_LENGTH = 3


def _parse_sheet(data: pd.DataFrame, scenario: str) -> dict[tuple[str, int], float]:
    """
    Reads one worksheet of the macro-financial scenario.

    Args:
        data (pd.DataFrame): The worksheet, read without a header.
        scenario (str): "historic", "baseline" or "adverse".

    Returns:
        dict[tuple[str, int], float]: The values as published, per row label and year.
    """
    labels = SCENARIO_LABELS[scenario]

    # The row naming the scenarios, directly above the row with the years.
    for header_row in range(min(len(data) - 1, 10)):
        names = data.iloc[header_row].ffill().astype(str).str.strip().str.lower()
        years = pd.to_numeric(data.iloc[header_row + 1], errors="coerce")
        columns = [
            column
            for column in data.columns
            if pd.notna(years[column])
            and FIRST_YEAR <= years[column] <= LAST_YEAR
            and names[column].startswith(labels)
        ]
        if columns:
            break
    else:
        return {}

    cells = data.to_numpy()
    first_value_column = min(columns)
    values: dict[tuple[str, int], float] = {}

    for row in cells[header_row + 2 :]:
        label = next(
            (
                cell.strip()
                for cell in row[:first_value_column]
                if isinstance(cell, str) and len(cell.strip()) >= MINIMUM_LABEL_LENGTH
            ),
            None,
        )
        if label is None:
            continue

        for column in columns:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(value):
                values[(label, int(years[column]))] = float(value)

    return values


def get_macro_financial_scenario(scenario: str) -> pd.DataFrame:
    """
    Retrieves the latest macro-financial scenario of the EU-wide bank stress test,
    yearly for three years: GDP, inflation, unemployment, residential and commercial
    property prices and long-term rates for the EU members and the main other economies,
    stock prices by region, commodity prices and foreign demand, iTraxx credit spreads and
    exchange rates.

    Args:
        scenario (str): "historic" (the starting point), "baseline" or "adverse".

    Returns:
        pd.DataFrame: The variables as decimals (exchange rates as published), indexed by
        year with a column per country (or region, index or commodity) and variable.
    """
    link, date = get_macro_financial_scenario_link()
    description = f"ESRB macro-financial scenario of {date}"

    def fetch() -> pd.DataFrame:
        workbook = pd.ExcelFile(io.BytesIO(get_request(link, timeout=120).content))
        series = {}

        for sheet, (variable, scale) in SHEETS.items():
            if sheet not in workbook.sheet_names:
                continue

            values = _parse_sheet(
                pd.read_excel(workbook, sheet_name=sheet, header=None), scenario
            )
            for (label, year), value in values.items():
                # The notes below a table are text without values and so never reach here.
                name = NAMES.get(label, re.sub(r"\s+", " ", label))
                series[(name, variable, year)] = value / scale

        if not series:
            raise ValueError(
                f"The {description} has none of the expected worksheets, which means its "
                "layout changed."
            )

        frame = pd.Series(series).unstack(level=2).T
        frame.index = pd.PeriodIndex([str(year) for year in frame.index], freq="Y")
        frame.columns.names = ["Country", "Variable"]

        return frame.sort_index(axis=1)

    return collect_cached_data(
        source=policy_model.EUROPEAN_SYSTEMIC_RISK_BOARD,
        dataset="scenario",
        entity=f"{date}/{scenario}",
        fetch=fetch,
        description=description,
    )
