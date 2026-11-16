"""U.S. Treasury HQM Corporate Bond Yield Curve Model"""

__docformat__ = "google"

import io
from datetime import datetime

import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.economics.sources import treasury_model
from financetoolkit.helpers import round_half_up
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request

# The Treasury publishes the High Quality Market (HQM) corporate bond yield curve as Excel
# files, the source of FRED's HQMCB series: the monthly average spot rates in one file per
# five years (1984-1988, 1989-1993, ...) and the par yields in a single file.
BASE_URL = "https://home.treasury.gov/system/files/226/"
SPOT_FILE = "hqm_{first:02d}_{last:02d}.xls"
PAR_FILE = "hqm_qh_pars.xls"

FIRST_YEAR = 1984
YEARS_PER_FILE = 5

# The spot rates hold a row per maturity in years, every half year from 0.5 to 100. These
# are the maturities the curve is returned for, named as the columns of the FRED series.
SPOT_MATURITIES: dict[float, str] = {
    0.5: "6M",
    1: "1Y",
    2: "2Y",
    3: "3Y",
    4: "4Y",
    5: "5Y",
    7: "7Y",
    10: "10Y",
    15: "15Y",
    20: "20Y",
    25: "25Y",
    30: "30Y",
    40: "40Y",
    50: "50Y",
    60: "60Y",
    75: "75Y",
    100: "100Y",
}

# The par yields hold a column per maturity, after the date and an empty column.
PAR_COLUMNS: dict[int, str] = {2: "2Y", 3: "5Y", 4: "10Y", 5: "30Y"}

# The rows of the spot file: the year above each month, the month names, and from the
# seventh row the maturities.
SPOT_YEAR_ROW, SPOT_MONTH_ROW, FIRST_DATA_ROW = 3, 4, 6


def _get_spot_file(first_year: int) -> pd.DataFrame:
    """
    Retrieves the monthly average HQM spot rates of one five-year file. A file of past
    years no longer changes, so it is cached for longer than the current one.

    Args:
        first_year (int): The first year of the file (1984, 1989, ...).

    Returns:
        pd.DataFrame: The spot rates in percent, indexed by month with a column per
        maturity ("6M" to "100Y").
    """
    last_year = first_year + YEARS_PER_FILE - 1
    file_name = SPOT_FILE.format(first=first_year % 100, last=last_year % 100)
    description = f"HQM spot rates of {first_year}-{last_year}"

    def fetch() -> pd.DataFrame:
        content = get_request(
            BASE_URL + file_name, timeout=60, extra_headers=TOOLKIT_HEADERS
        ).content
        sheet = pd.read_excel(io.BytesIO(content), header=None)

        # Each year is named above its first month only.
        years = sheet.iloc[SPOT_YEAR_ROW].ffill()
        months = sheet.iloc[SPOT_MONTH_ROW]
        columns = {
            column: pd.Period(f"{int(years[column])}-{months[column]}", freq="M")
            for column in sheet.columns[1:]
            if isinstance(months[column], str) and pd.notna(years[column])
        }

        if not columns:
            raise ValueError(
                f"The Treasury file {file_name} has no months, which means its layout "
                "changed."
            )

        rates = sheet.iloc[FIRST_DATA_ROW:].set_index(0)[list(columns)]
        rates.index = pd.to_numeric(rates.index, errors="coerce")
        rates = rates.rename(columns=columns).T
        rates = rates[[maturity for maturity in SPOT_MATURITIES if maturity in rates]]

        if rates.empty:
            raise ValueError(
                f"The Treasury file {file_name} has none of the maturities, which means "
                "its layout changed."
            )

        # Months not published yet are empty.
        return (
            rates.rename(columns=SPOT_MATURITIES)
            .apply(pd.to_numeric, errors="coerce")
            .dropna(how="all")
            .rename_axis(index=None, columns=None)
        )

    return collect_cached_data(
        source=policy_model.US_TREASURY,
        dataset="hqm" if last_year >= datetime.now().year else "hqm_years",
        entity=file_name,
        fetch=fetch,
        description=description,
    )


def get_hqm_spot_rates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the monthly average spot (zero-coupon) rates of the HQM corporate bond yield
    curve from the U.S. Treasury, from 1984, for 17 maturities from 6 months to 100 years.
    Only the files that cover the date range are requested.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: The rates as decimals (semi-annual bond-equivalent), indexed by month
        with a column per maturity ("6M" to "100Y").
    """
    first_year = max(FIRST_YEAR, int(start_date[:4]))
    last_year = min(datetime.now().year, int(end_date[:4]))

    files = sorted(
        {
            FIRST_YEAR + (year - FIRST_YEAR) // YEARS_PER_FILE * YEARS_PER_FILE
            for year in range(first_year, last_year + 1)
        }
    )
    frames = [
        frame
        for frame in helpers.run_in_parallel(
            _get_spot_file, [(year,) for year in files]
        )
        if not frame.empty
    ]

    if not frames:
        return pd.DataFrame()

    rates = pd.concat(frames).sort_index()

    # The Treasury publishes the rates in percent.
    return rates.dropna(how="all") / 100


def get_hqm_par_yields() -> pd.DataFrame:
    """
    Retrieves the monthly average par yields of the HQM corporate bond yield curve from the
    U.S. Treasury, from 1984, at 2, 5, 10 and 30 years.

    Returns:
        pd.DataFrame: The yields as decimals (semi-annual bond-equivalent), indexed by
        month with a column per maturity.
    """
    description = "HQM par yields"

    def fetch() -> pd.DataFrame:
        content = get_request(
            BASE_URL + PAR_FILE, timeout=60, extra_headers=TOOLKIT_HEADERS
        ).content
        sheet = pd.read_excel(io.BytesIO(content), header=None)

        dates = pd.to_datetime(sheet[0], format="%b %Y", errors="coerce")
        rows = dates.notna()

        if not rows.any():
            raise ValueError(
                f"The Treasury file {PAR_FILE} has no months, which means its layout "
                "changed."
            )

        yields = sheet.loc[rows, list(PAR_COLUMNS)].rename(columns=PAR_COLUMNS)
        yields.index = pd.PeriodIndex(dates[rows], freq="M")

        return (
            yields.apply(pd.to_numeric, errors="coerce")
            .dropna(how="all")
            .rename_axis(index=None, columns=None)
        )

    # The Treasury publishes the yields in percent.
    return (
        collect_cached_data(
            source=policy_model.US_TREASURY,
            dataset="hqm",
            entity=PAR_FILE,
            fetch=fetch,
            description=description,
        )
        / 100
    )


# The Treasury par yields at the maturities the HQM par yields are published for, as the
# daily par yield curve names them.
TREASURY_COLUMNS: dict[str, str] = {
    "2 Yr": "2Y",
    "5 Yr": "5Y",
    "10 Yr": "10Y",
    "30 Yr": "30Y",
}


def get_hqm_corporate_bond_spread(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Computes the spread of the HQM corporate par yields over the Treasury par yields at 2,
    5, 10 and 30 years, both monthly averages, from the U.S. Treasury's own data.

    The Treasury yields are the monthly averages of its daily par yield curve, rounded to
    two decimals in percent as the Federal Reserve's H.15 release publishes them. The
    daily curve starts in 1990 and has no 30-year yield from February 2002 to February
    2006, when no 30-year bond was issued. Only completed months are averaged.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: The spreads as decimals (0.012 for 1.2 percentage points), indexed
        by month with a column per maturity.
    """
    corporate = get_hqm_par_yields()
    daily = treasury_model.get_yield_curve("nominal", start_date, end_date)

    if corporate.empty or daily.empty:
        return pd.DataFrame()

    daily = daily[[column for column in TREASURY_COLUMNS if column in daily]]
    daily = daily.rename(columns=TREASURY_COLUMNS)

    treasury = (daily * 100).groupby(daily.index.asfreq("M")).mean()
    treasury = treasury[treasury.index < pd.Period(datetime.now(), freq="M")]

    # Rounded half up to two decimals in percent, as H.15 publishes the monthly averages.
    treasury = treasury.map(round_half_up, decimals=2)

    return (corporate - treasury / 100).dropna(how="all")
