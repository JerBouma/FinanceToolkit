"""Federal Reserve Board Model"""

__docformat__ = "google"

import io
import re

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.requests_model import get_request

# The seasonally adjusted industrial production indices of the G.17 release, the source
# of FRED's INDPRO series. Every index is one row per year with the twelve months.
URL = "https://www.federalreserve.gov/releases/g17/Current/ipdisk/ip_sa.txt"

# The total index.
TOTAL_INDEX = "B50001"

# A data row holds the series code, the year and the months: "B50001" 2026 101.926 ...
ROW_PATTERN = re.compile(rf'"{TOTAL_INDEX}"\s+(\d{{4}})\s+(.*)$')

COUNTRY = "United States"


def get_industrial_production_index() -> pd.DataFrame:
    """
    Retrieves the monthly Industrial Production Index (2017 = 100), seasonally adjusted,
    from the Federal Reserve's G.17 release, from 1919.

    Returns:
        pd.DataFrame: A single "United States" column, indexed by the first day of each
        month ("Date") as FRED dates it.

    Raises:
        ValueError: When the file has no rows for the total index, which means its layout
            changed.
    """
    description = "industrial production index"

    def fetch() -> pd.DataFrame:
        response = get_request(URL, timeout=120)
        observations = {}

        # Months not published yet are absent or zero.
        for line in response.text.splitlines():
            # The first row of a series also carries its label in front of the code.
            match = ROW_PATTERN.search(line)
            if match is None:
                continue

            year, values = match.group(1), match.group(2).split()

            for month, value in enumerate(values[:12], start=1):
                number = pd.to_numeric(value, errors="coerce")
                if pd.notna(number) and number > 0:
                    observations[pd.Period(f"{year}-{month:02d}-01", freq="D")] = number

        if not observations:
            raise ValueError(
                f"The G.17 file has no rows for the total {description} ({TOTAL_INDEX}), which "
                "means the Federal Reserve changed its layout."
            )

        index = pd.PeriodIndex(list(observations), freq="D", name="Date")

        return pd.DataFrame(
            {COUNTRY: list(observations.values())}, index=index
        ).sort_index()

    return collect_cached_data(
        source=policy_model.FEDERAL_RESERVE_BOARD,
        dataset="series",
        entity=TOTAL_INDEX,
        fetch=fetch,
        description=description,
    )


# The credit spread of Gilchrist and Zakrajšek (2012) and its excess bond premium, which
# the Federal Reserve Board updates monthly from 1973.
EXCESS_BOND_PREMIUM_URL = (
    "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
)

EXCESS_BOND_PREMIUM_COLUMNS = {
    "gz_spread": "GZ Credit Spread",
    "ebp": "Excess Bond Premium",
    "est_prob": "Recession Probability",
}


def get_excess_bond_premium() -> pd.DataFrame:
    """
    Retrieves the Gilchrist-Zakrajšek credit spread, the average spread of US corporate
    bonds over Treasuries with the same cash flows, its excess bond premium, the part of
    that spread not explained by expected defaults, and the probability of a recession over
    the next twelve months the premium implies, monthly from 1973.

    Returns:
        pd.DataFrame: The spread and premium as decimals and the probability as a fraction,
        indexed by month.

    Raises:
        ValueError: When the file misses one of the expected columns.
    """
    description = "excess bond premium"

    def fetch() -> pd.DataFrame:
        response = get_request(EXCESS_BOND_PREMIUM_URL, timeout=60)
        data = pd.read_csv(io.StringIO(response.text))
        require_columns(data, {"date", *EXCESS_BOND_PREMIUM_COLUMNS}, description)

        # Dates have been written both as 1/1/1973 and as 1973-01-01.
        index = pd.PeriodIndex(pd.to_datetime(data["date"], format="mixed"), freq="M")
        premium = data[list(EXCESS_BOND_PREMIUM_COLUMNS)].rename(
            columns=EXCESS_BOND_PREMIUM_COLUMNS
        )
        premium.index = index
        premium = premium.apply(pd.to_numeric, errors="coerce")

        # The spread and premium are published in percentage points; the probability is
        # already a fraction.
        premium[["GZ Credit Spread", "Excess Bond Premium"]] /= 100

        return premium.sort_index()

    return collect_cached_data(
        source=policy_model.FEDERAL_RESERVE_BOARD,
        dataset="series",
        entity="excess_bond_premium",
        fetch=fetch,
        description=description,
    )


# The scenarios of the Federal Reserve's annual supervisory stress test, published in
# February of the year of the test, each as a CSV file for US ("Domestic") and foreign
# ("International") variables, with the history of the same variables from 1976.
SUPERVISORY_SCENARIO_URL = "https://www.federalreserve.gov/supervisionreg/files/{year}_Final_{scenario}_{region}.csv"
# The status of a scenario file that is not published (yet).
NOT_PUBLISHED = 404

SUPERVISORY_SCENARIOS = {
    "baseline": "Supervisory_Baseline",
    "adverse": "Supervisory_Severely_Adverse",
    "historic": "Historic",
}
SUPERVISORY_REGIONS = {
    "Euro area": "Euro Area",
    "Developing Asia": "Developing Asia",
    "Japan": "Japan",
    "U.K.": "United Kingdom",
}


def _parse_supervisory_scenario(
    text: str, domestic: bool, description: str
) -> pd.DataFrame:
    """
    Reads one scenario file into a column per region and variable, indexed by quarter.

    Args:
        text (str): The CSV file.
        domestic (bool): Whether the file holds the US variables.
        description (str): What is read, used in the error message.

    Returns:
        pd.DataFrame: The variables, with rates and growth as decimals and levels, index
        values and exchange rates as published.
    """
    data = pd.read_csv(io.StringIO(text))
    require_columns(data, {"Date"}, description)

    data.index = pd.PeriodIndex(data["Date"].str.replace(" ", ""), freq="Q")
    data = data.drop(
        columns=[column for column in ("Scenario Name", "Date") if column in data]
    )
    data = data.apply(pd.to_numeric, errors="coerce")
    columns = []

    for column in data.columns:
        if domestic:
            region, variable = "United States", column
        else:
            prefix = next(
                (name for name in SUPERVISORY_REGIONS if column.startswith(name)), None
            )
            if prefix is None:
                region, variable = "Other", column
            else:
                region = SUPERVISORY_REGIONS[prefix]
                variable = column[len(prefix) :].strip()
                variable = variable[:1].upper() + variable[1:]

        # The VIX is a volatility in percent, returned as a decimal like elsewhere.
        if variable.startswith("Market Volatility Index"):
            variable = "Market Volatility Index (VIX)"

        # Growth, inflation, rates and the VIX are published in percent; levels and
        # exchange rates are not.
        if "(Level)" not in variable and "exchange rate" not in variable:
            data[column] = data[column] / 100

        columns.append((region, variable))

    data.columns = pd.MultiIndex.from_tuples(columns, names=["Country", "Variable"])
    data.index.name = None

    return data.sort_index()


def get_supervisory_scenario(scenario: str, year: int | None = None) -> pd.DataFrame:
    """
    Retrieves a scenario of the Federal Reserve's supervisory stress test: the baseline,
    the severely adverse scenario, or the history of the same variables from 1976, for the
    United States (GDP, income, unemployment, inflation, Treasury, BBB corporate and
    mortgage rates, equity, house and commercial property prices and the VIX) and for the
    euro area, developing Asia, Japan and the United Kingdom (GDP, inflation and exchange
    rates).

    Args:
        scenario (str): "baseline", "adverse" (severely adverse) or "historic".
        year (int | None): The year of the stress test. Defaults to None, which takes the
            latest published.

    Returns:
        pd.DataFrame: The variables, indexed by quarter with a column per region and
        variable.
    """
    years = [year] if year else [pd.Timestamp.now().year, pd.Timestamp.now().year - 1]
    description = f"Federal Reserve supervisory {scenario} scenario"

    for candidate in years:

        def fetch(candidate: int = candidate) -> pd.DataFrame:
            frames = []
            for region in ("Domestic", "International"):
                try:
                    response = get_request(
                        SUPERVISORY_SCENARIO_URL.format(
                            year=candidate,
                            scenario=SUPERVISORY_SCENARIOS[scenario],
                            region=region,
                        ),
                        timeout=60,
                    )
                except requests.exceptions.HTTPError as error:
                    # The scenarios of a year are published in February.
                    if (
                        error.response is not None
                        and error.response.status_code == NOT_PUBLISHED
                    ):
                        return pd.DataFrame()
                    raise
                frames.append(
                    _parse_supervisory_scenario(
                        response.text, region == "Domestic", description
                    )
                )
            return pd.concat(frames, axis=1)

        scenario_data = collect_cached_data(
            source=policy_model.FEDERAL_RESERVE_BOARD,
            dataset="scenario",
            entity=f"{candidate}/{scenario}",
            fetch=fetch,
            description=f"{description} of {candidate}",
        )

        if not scenario_data.empty:
            return scenario_data

    return pd.DataFrame()
