"""Deutsche Bundesbank Model"""

__docformat__ = "google"

import io
import re

import numpy as np
import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import (
    collect_cached_data,
    collect_ranged_data,
    require_columns,
)
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


# The yields of listed federal securities by ISIN (the key ends with the rating, left
# open): inflation-linked federal bonds and
# notes are recognised by their title, and every one of them matures on 15 April.
SECURITIES_URL = "https://api.statistiken.bundesbank.de/rest/data/BBSSY/"
LINKER_KEY = "D.REN.EUR.A630+A640.{isins}."
LINKER_TITLE = "Inflationsindex"
LINKER_YEAR_PATTERN = re.compile(r"\((\d{2})\)\s*$")
LINKER_MATURITY_MONTH_DAY = "04-15"


def _read_bundesbank_csv(content: bytes) -> pd.DataFrame:
    """
    Reads an SDMX-CSV response of the Bundesbank, whose separator follows the language.

    Args:
        content (bytes): The response.

    Returns:
        pd.DataFrame: The rows of the response.
    """
    text = content.decode("utf-8-sig")

    if not text.strip():
        return pd.DataFrame()

    header = text.splitlines()[0]
    return pd.read_csv(
        io.StringIO(text), sep=";" if header.count(";") > header.count(",") else ","
    )


def get_inflation_linked_bonds() -> pd.DataFrame:
    """
    Lists the inflation-linked German federal securities the Bundesbank publishes yields
    for, with their maturity.

    Returns:
        pd.DataFrame: The maturity date per ISIN.

    Raises:
        ValueError: When the response misses the columns needed to recognise the bonds.
    """
    description = "German inflation-linked federal securities"

    def fetch() -> pd.DataFrame:
        response = get_request(
            f"{SECURITIES_URL}{LINKER_KEY.format(isins='')}?format=sdmx_csv&lang=en&lastNObservations=1",
            timeout=120,
            extra_headers={"Accept": "text/csv"},
        )
        data = _read_bundesbank_csv(response.content)
        require_columns(data, {"BBK_SEIS_ISIN", "BBK_TITLE_ENG"}, description)

        linkers = data[data["BBK_TITLE_ENG"].str.contains(LINKER_TITLE, na=False)]
        maturities = {}
        for isin, title in zip(
            linkers["BBK_SEIS_ISIN"], linkers["BBK_TITLE_ENG"], strict=True
        ):
            year = LINKER_YEAR_PATTERN.search(str(title))
            if year:
                maturities[isin] = pd.Timestamp(
                    f"20{year.group(1)}-{LINKER_MATURITY_MONTH_DAY}"
                )

        return pd.DataFrame(
            {"Maturity": list(maturities.values())}, index=list(maturities)
        )

    return collect_cached_data(
        source=policy_model.BUNDESBANK,
        dataset="linker_listing",
        entity="BBSSY",
        fetch=fetch,
        description=description,
    )


def get_inflation_linked_yields(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily real yields of the inflation-linked German federal securities,
    which are indexed to euro area inflation (the HICP excluding tobacco), from 2012. Only
    the inflation-linked bonds and only the days that are not cached yet are requested.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The real yields as decimals, indexed by business day with a column per
        ISIN.
    """
    bonds = get_inflation_linked_bonds()
    description = "German inflation-linked federal securities yields"

    if bonds.empty:
        return pd.DataFrame()

    key = LINKER_KEY.format(isins="+".join(bonds.index))

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{SECURITIES_URL}{key}?format=sdmx_csv&lang=en&startPeriod={fetch_start}&endPeriod={fetch_end}",
            timeout=120,
            extra_headers={"Accept": "text/csv"},
        )
        data = _read_bundesbank_csv(response.content)

        if data.empty:
            return data

        require_columns(
            data, {"BBK_SEIS_ISIN", "TIME_PERIOD", "OBS_VALUE"}, description
        )
        data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
        yields = data.pivot(
            index="TIME_PERIOD", columns="BBK_SEIS_ISIN", values="OBS_VALUE"
        )
        yields.index = pd.PeriodIndex(yields.index, freq="D")
        yields.index.name = None
        yields.columns.name = None

        # The Bundesbank publishes the yields in percent; weekends are listed without one.
        return yields.dropna(how="all").sort_index() / 100

    return collect_ranged_data(
        source=policy_model.BUNDESBANK,
        dataset="series",
        entity=f"BBSSY/{key}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


def get_breakeven_inflation(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Computes the breakeven inflation of every inflation-linked German federal security:
    the nominal yield of the Bundesbank's term structure at the bond's remaining maturity,
    interpolated linearly between whole years, minus the bond's real yield. Since the bonds
    are indexed to euro area inflation, this is a market-based measure of the inflation the
    euro area bond market expects.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The breakeven inflation as decimals, indexed by business day with a
        column per bond, named by its maturity year.
    """
    bonds = get_inflation_linked_bonds()
    real_yields = get_inflation_linked_yields(start_date, end_date)
    nominal_curve = get_yield_curve(start_date, end_date)

    if bonds.empty or real_yields.empty or nominal_curve.empty:
        return pd.DataFrame()

    dates = real_yields.index.intersection(nominal_curve.index)
    tenors = np.array([float(label[:-1]) for label in nominal_curve.columns])
    breakeven = pd.DataFrame(index=dates)

    for isin in real_yields.columns:
        if isin not in bonds.index:
            continue

        maturity = bonds.loc[isin, "Maturity"]
        remaining = np.array(
            [(maturity - date.to_timestamp()).days / 365.25 for date in dates]
        )
        nominal = np.array(
            [
                np.interp(years, tenors, row) if years > 0 else np.nan
                for years, row in zip(
                    remaining, nominal_curve.loc[dates].to_numpy(), strict=True
                )
            ]
        )
        breakeven[str(maturity.year)] = (
            nominal - real_yields.loc[dates, isin].to_numpy()
        )

    breakeven = breakeven[sorted(breakeven.columns)]
    breakeven.index.name = None

    return breakeven.dropna(how="all")
