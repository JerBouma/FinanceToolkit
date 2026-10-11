"""Network for Greening the Financial System (NGFS) Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import HEADERS, SESSION, get_request

# The NGFS climate scenarios are hosted by IIASA, whose database answers anonymous users
# with a guest token. Phase 5 is the vintage of November 2024; a new phase is published as
# a new database.
TOKEN_URL = "https://api.manager.ece.iiasa.ac.at/legacy/anonym/"
BASE_URL = "https://db1.ene.iiasa.ac.at/ngfs-phase-5-api/rest/v2.1"

SCENARIOS = [
    "Net Zero 2050",
    "Below 2°C",
    "Low demand",
    "Delayed transition",
    "Nationally Determined Contributions (NDCs)",
    "Current Policies",
    "Fragmented World",
]

# The integrated assessment models that project the energy system and the carbon price;
# NiGEM, the macroeconometric model of the National Institute of Economic and Social
# Research, translates each of their scenarios into the economy and financial markets.
MODELS = {
    "REMIND-MAgPIE": "REMIND-MAgPIE",
    "GCAM": "GCAM",
    "MESSAGEix-GLOBIOM": "MESSAGEix-GLOBIOM",
}

# The NiGEM variables by the names used here, with whether their deviation from the
# baseline is published in percent of the level ("relative") or in percentage points of
# a rate ("absolute"), and whether the level is a rate in percent.
NIGEM_VARIABLES = {
    "gdp": ("Gross Domestic Product (GDP)", "relative", False),
    "inflation": ("Inflation rate ; %", "absolute", True),
    "long_term_interest_rate": ("Long term interest rate ; %", "absolute", True),
    "long_term_real_interest_rate": (
        "Long term real interest rate ; %",
        "absolute",
        True,
    ),
    "policy_rate": (
        "Central bank Intervention rate (policy interest rate) ; %",
        "absolute",
        True,
    ),
    "unemployment_rate": ("Unemployment rate ; %", "absolute", True),
    "equity_prices": ("Equity prices", "relative", False),
    "effective_exchange_rate": ("Effective exchange rate", "relative", False),
    "domestic_demand": ("Domestic demand", "relative", False),
    "exports": ("Exports (goods and services)", "relative", False),
    "imports": ("Imports (goods and services)", "relative", False),
    "oil_price": ("Oil price ; US$ per barrel", "relative", False),
    "gas_price": ("Gas price ; US$ per barrel (equiv)", "relative", False),
    "coal_price": ("Coal price ; US$ per barrel (equiv)", "relative", False),
}

# The risks NiGEM separates: the transition to a low-carbon economy, the physical damage
# of climate change, and both together.
RISKS = ["combined", "transition", "physical"]

CARBON_PRICE = "Price|Carbon"


def _token() -> str:
    """
    Requests a guest token for the IIASA database.

    Returns:
        str: The token.
    """
    # Without asking for JSON the endpoint answers with its HTML page.
    response = get_request(
        TOKEN_URL, timeout=30, extra_headers={"Accept": "application/json"}
    )

    return str(response.json())


def _post(path: str, body: dict) -> list:
    """
    Sends a query to the NGFS database with a fresh guest token.

    Args:
        path (str): The endpoint below the base URL, e.g. "runs/bulk/ts".
        body (dict): The query.

    Returns:
        list: The records of the response.
    """
    response = SESSION.post(
        f"{BASE_URL}/{path}",
        json=body,
        headers={
            **HEADERS,
            "Authorization": f"Bearer {_token()}",
            "Accept": "application/json",
        },
        timeout=180,
    )
    response.raise_for_status()

    return response.json()


def get_runs() -> pd.DataFrame:
    """
    Lists the default runs of the NGFS database: one per model and scenario.

    Returns:
        pd.DataFrame: The run identifier, model and scenario of every run.
    """
    description = "NGFS scenario runs"

    def fetch() -> pd.DataFrame:
        response = get_request(
            f"{BASE_URL}/runs?getOnlyDefaultRuns=true",
            timeout=60,
            extra_headers={
                "Authorization": f"Bearer {_token()}",
                "Accept": "application/json",
            },
        )
        runs = pd.DataFrame(response.json())

        if not {"run_id", "model", "scenario"} <= set(runs.columns):
            raise ValueError(
                f"The {description} miss the run, model or scenario, which means the API changed."
            )

        return runs[["run_id", "model", "scenario"]]

    return collect_cached_data(
        source=policy_model.NGFS,
        dataset="runs",
        entity="phase_5",
        fetch=fetch,
        description=description,
    )


def _find_run(runs: pd.DataFrame, model: str, scenario: str, nigem: bool) -> int | None:
    """
    Finds the run of a model and scenario: the NiGEM run that translates an integrated
    assessment model's scenario, or that model's own run. Downscaled runs and the variant
    with integrated physical damages are not used.

    Args:
        runs (pd.DataFrame): The runs of get_runs.
        model (str): A key of MODELS.
        scenario (str): The scenario, e.g. "Net Zero 2050".
        nigem (bool): Whether to find the NiGEM run.

    Returns:
        int | None: The run identifier, or None when there is no such run.
    """
    names = runs["model"].astype(str)
    selected = (
        names.str.startswith("NiGEM")
        & names.str.contains(f"[{MODELS[model]}", regex=False)
        if nigem
        else names.str.startswith(MODELS[model])
        & ~names.str.contains("IntegratedPhysicalDamages", regex=False)
    )
    matches = runs[selected & (runs["scenario"] == scenario)]

    return int(matches["run_id"].iloc[0]) if not matches.empty else None


def _get_timeseries(run: int, variable: str) -> pd.DataFrame:
    """
    Retrieves one variable of one run for every region and year.

    Args:
        run (int): The run identifier.
        variable (str): The variable as the database names it.

    Returns:
        pd.DataFrame: The value per year (rows) and region (columns), the regions without
        the model prefix ("NiGEM NGFS v1.24.2|Germany" becomes "Germany").
    """
    description = f"NGFS {variable} of run {run}"

    def fetch() -> pd.DataFrame:
        records = _post(
            "runs/bulk/ts",
            {
                "filters": {
                    "runs": [run],
                    "variables": [variable],
                    "regions": [],
                    "units": [],
                    "years": [],
                    "timeslices": [],
                }
            },
        )
        data = pd.DataFrame(records)

        if data.empty:
            return data

        if not {"region", "year", "value"} <= set(data.columns):
            raise ValueError(
                f"The {description} misses the region, year or value, which means the API changed."
            )

        data["region"] = data["region"].astype(str).str.split("|").str[-1]
        values = data.pivot_table(
            index="year", columns="region", values="value", aggfunc="last"
        )
        values.index = pd.PeriodIndex([str(year) for year in values.index], freq="Y")
        values.index.name = None
        values.columns.name = None

        return values.sort_index()

    return collect_cached_data(
        source=policy_model.NGFS,
        dataset="timeseries",
        entity=f"{run}/{variable}",
        fetch=fetch,
        description=description,
    )


def get_nigem_scenario(
    variable: str, scenario: str, model: str, risk: str, deviation: bool
) -> pd.DataFrame:
    """
    Retrieves the path of a macroeconomic or financial variable in an NGFS scenario from
    NiGEM, per country and region, yearly from 2022 to 2050.

    Args:
        variable (str): A key of NIGEM_VARIABLES, e.g. "long_term_interest_rate".
        scenario (str): The scenario, e.g. "Net Zero 2050", or "Baseline".
        model (str): A key of MODELS, the integrated assessment model behind the scenario.
        risk (str): "combined", "transition" or "physical".
        deviation (bool): Whether to return the deviation from the baseline instead of the
            level.

    Returns:
        pd.DataFrame: The level (rates as decimals) or the deviation (relative deviations
        and rate changes as decimals), indexed by year with a column per country.
    """
    name, kind, is_rate = NIGEM_VARIABLES[variable]
    runs = get_runs()

    if runs.empty:
        return pd.DataFrame()

    baseline_run = _find_run(runs, model, "Baseline", nigem=True)
    baseline = (
        _get_timeseries(baseline_run, name)
        if baseline_run is not None
        else pd.DataFrame()
    )

    if scenario == "Baseline":
        if deviation:
            return baseline * 0 if not baseline.empty else baseline
        return baseline / 100 if is_rate and not baseline.empty else baseline

    scenario_run = _find_run(runs, model, scenario, nigem=True)

    if scenario_run is None:
        return pd.DataFrame()

    difference = _get_timeseries(scenario_run, f"{name}({risk})")

    if difference.empty:
        return difference

    # Both kinds of deviation are published in percent.
    difference = difference / 100

    if deviation:
        return difference

    if baseline.empty:
        return pd.DataFrame()

    if kind == "relative":
        level = baseline * (1 + difference)
    else:
        level = baseline / 100 + difference if is_rate else baseline + difference * 100

    return level.dropna(how="all", axis=1)


def get_carbon_price(scenario: str, model: str) -> pd.DataFrame:
    """
    Retrieves the carbon price an integrated assessment model projects in an NGFS
    scenario, for the world and the model's regions, every five years to 2100.

    Args:
        scenario (str): The scenario, e.g. "Net Zero 2050".
        model (str): A key of MODELS.

    Returns:
        pd.DataFrame: The price in US dollars of 2010 per tonne of CO2, indexed by year with
        a column per region.
    """
    runs = get_runs()

    if runs.empty:
        return pd.DataFrame()

    run = _find_run(runs, model, scenario, nigem=False)

    return _get_timeseries(run, CARBON_PRICE) if run is not None else pd.DataFrame()
