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


# The constant maturities the bonds are interpolated to. Germany has had four to six
# inflation-linked bonds outstanding at a time, so only the maturities between the
# shortest and longest remaining maturity of a day are filled. In its last two years a
# bond's real yield is dominated by the indexation lag and seasonality of the inflation it
# has still to accrue, which distorts the short end, so such bonds are left out.
CONSTANT_MATURITIES = [3, 5, 7, 10, 15, 20, 30]
MINIMUM_REMAINING_YEARS = 2

# Two bonds are needed to interpolate between.
MINIMUM_BONDS = 2


def _to_constant_maturities(
    values: pd.DataFrame, remaining: pd.DataFrame
) -> pd.DataFrame:
    """
    Interpolates per-bond values linearly to constant maturities, day by day, without
    extrapolating beyond the bonds' remaining maturities.

    Args:
        values (pd.DataFrame): The values per day (rows) and bond (columns).
        remaining (pd.DataFrame): The remaining maturity in years, in the same shape.

    Returns:
        pd.DataFrame: The values per day and constant maturity ("3Y" to "30Y").
    """
    rows = []

    for date in values.index:
        years = remaining.loc[date].to_numpy(dtype=float)
        observed = values.loc[date].to_numpy(dtype=float)
        mask = ~np.isnan(observed) & (years >= MINIMUM_REMAINING_YEARS)
        order = np.argsort(years[mask])
        years, observed = years[mask][order], observed[mask][order]

        rows.append(
            [
                (
                    np.interp(maturity, years, observed)
                    if len(years) >= MINIMUM_BONDS and years[0] <= maturity <= years[-1]
                    else np.nan
                )
                for maturity in CONSTANT_MATURITIES
            ]
        )

    curve = pd.DataFrame(
        rows,
        index=values.index,
        columns=[f"{maturity}Y" for maturity in CONSTANT_MATURITIES],
    )

    return curve.dropna(how="all", axis=0).dropna(how="all", axis=1)


def _remaining_maturities(index: pd.Index, bonds: pd.DataFrame, isins) -> pd.DataFrame:
    """
    Computes the remaining maturity in years of every bond on every day.

    Args:
        index (pd.Index): The days, as a daily PeriodIndex.
        bonds (pd.DataFrame): The maturity date per ISIN.
        isins: The ISINs, in the order of the columns.

    Returns:
        pd.DataFrame: The remaining maturity in years per day and ISIN.
    """
    days = index.to_timestamp()

    return pd.DataFrame(
        {isin: (bonds.loc[isin, "Maturity"] - days).days / 365.25 for isin in isins},
        index=index,
    )


def get_real_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Interpolates the real yields of the inflation-linked German federal securities to
    constant maturities, daily from 2012.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The real yields as decimals, indexed by business day with a column per
        maturity ("3Y" to "30Y") between the shortest and longest bond outstanding.
    """
    bonds = get_inflation_linked_bonds()
    real_yields = get_inflation_linked_yields(start_date, end_date)

    if bonds.empty or real_yields.empty:
        return pd.DataFrame()

    real_yields = real_yields[[isin for isin in real_yields if isin in bonds.index]]
    remaining = _remaining_maturities(real_yields.index, bonds, real_yields.columns)

    return _to_constant_maturities(real_yields, remaining)


def get_breakeven_inflation(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Computes the breakeven inflation of the inflation-linked German federal securities at
    constant maturities. For every bond the breakeven is the nominal yield of the
    Bundesbank's term structure at the bond's remaining maturity, interpolated linearly
    between whole years, minus the bond's real yield; those are then interpolated to
    constant maturities. Since the bonds are indexed to euro area inflation, this is a
    market-based measure of the inflation the euro area bond market expects.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The breakeven inflation as decimals, indexed by business day with a
        column per maturity ("3Y" to "30Y") between the shortest and longest bond outstanding.
    """
    bonds = get_inflation_linked_bonds()
    real_yields = get_inflation_linked_yields(start_date, end_date)
    nominal_curve = get_yield_curve(start_date, end_date)

    if bonds.empty or real_yields.empty or nominal_curve.empty:
        return pd.DataFrame()

    dates = real_yields.index.intersection(nominal_curve.index)
    isins = [isin for isin in real_yields.columns if isin in bonds.index]
    remaining = _remaining_maturities(dates, bonds, isins)
    tenors = np.array([float(label[:-1]) for label in nominal_curve.columns])
    nominal_rows = nominal_curve.loc[dates].to_numpy()

    breakeven = pd.DataFrame(
        {
            isin: [
                np.interp(years, tenors, row) if years > 0 else np.nan
                for years, row in zip(remaining[isin], nominal_rows, strict=True)
            ]
            - real_yields.loc[dates, isin].to_numpy()
            for isin in isins
        },
        index=dates,
    )

    return _to_constant_maturities(breakeven, remaining)


# The expected real interest rates the Bundesbank derives from the yields on debt
# securities outstanding issued by residents with a residual maturity of 5 to 6 and 9 to 10
# years (monthly averages), minus the weighted inflation rates Consensus Economics expects
# over those horizons. Adding back the yields gives the survey's expected inflation.
EXPECTATIONS_URL = "https://api.statistiken.bundesbank.de/rest/data/BBSEI/"
EXPECTED_REAL_RATE_KEY = "M.ERZ.IHS.DE._Z.R05XX+R10XX"
OUTSTANDING_YIELD_KEY = "M.I.UMR.RD.EUR.A.B.A.R0506+R0910.R.A.A._Z._Z.A"
EXPECTATION_HORIZONS = {
    "R05XX": "5Y",
    "R10XX": "10Y",
    "R0506": "5Y",
    "R0910": "10Y",
}


def _get_monthly_series(
    url: str, key: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Retrieves monthly Bundesbank series with one column per horizon, requesting only the
    months that are not cached yet.

    Args:
        url (str): The dataflow URL.
        key (str): The series key.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The values as decimals, indexed by month with a column per horizon.
    """
    description = "German survey-based inflation expectations"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{url}{key}?format=sdmx_csv&lang=en&startPeriod={fetch_start[:7]}&endPeriod={fetch_end[:7]}",
            timeout=120,
            extra_headers={"Accept": "text/csv"},
        )
        data = _read_bundesbank_csv(response.content)

        if data.empty:
            return data

        require_columns(
            data, {"BBK_SEIS_MATURITY", "TIME_PERIOD", "OBS_VALUE"}, description
        )
        data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
        values = data.pivot(
            index="TIME_PERIOD", columns="BBK_SEIS_MATURITY", values="OBS_VALUE"
        ).rename(columns=EXPECTATION_HORIZONS)
        values.index = pd.PeriodIndex(values.index, freq="M")
        values.index.name = None
        values.columns.name = None

        # The Bundesbank publishes the rates in percent.
        return values[["5Y", "10Y"]].dropna(how="all").sort_index() / 100

    return collect_ranged_data(
        source=policy_model.BUNDESBANK,
        dataset="series",
        entity=f"{url.rstrip('/').rsplit('/', 1)[-1]}/{key}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


def get_expected_real_rates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the Bundesbank's expected real interest rates for Germany over 5 and 10 years,
    monthly from 1989: the yields on debt securities outstanding issued by residents minus
    the weighted inflation rates Consensus Economics expects over the same horizon.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The expected real rates as decimals, indexed by month with the columns
        "5Y" and "10Y".
    """
    return _get_monthly_series(
        EXPECTATIONS_URL,
        EXPECTED_REAL_RATE_KEY,
        start_date,
        end_date,
    )


def get_survey_inflation_expectations(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Derives the inflation Consensus Economics' survey expects for Germany over 5 and 10
    years, monthly from 1990, by adding the yields the Bundesbank's expected real rates are
    computed from back: the yield on debt securities outstanding with a residual maturity of
    5 to 6 (or 9 to 10) years minus the expected real rate is the expected inflation.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The expected inflation as decimals, indexed by month with the columns
        "5Y" and "10Y".
    """
    real_rates = get_expected_real_rates(start_date, end_date)
    yields = _get_monthly_series(BASE_URL, OUTSTANDING_YIELD_KEY, start_date, end_date)

    if real_rates.empty or yields.empty:
        return pd.DataFrame()

    return (yields - real_rates).dropna(how="all")
