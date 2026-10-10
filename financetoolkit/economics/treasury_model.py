"""U.S. Department of the Treasury Model"""

__docformat__ = "google"

import io
from datetime import datetime

import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.requests_model import get_request

logger = get_logger()

BASE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type={curve_type}"
    "&field_tdr_date_value={year}&page&_format=csv"
)

# The daily par yield curve is published from 1990 and the real (TIPS) par yield curve
# from 2003, one file per year each.
CURVES = {
    "nominal": {"type": "daily_treasury_yield_curve", "first_year": 1990},
    "real": {"type": "daily_treasury_real_yield_curve", "first_year": 2003},
}

# The maturities of the real yield curve, named as FRED's real yield series are in the
# Economics module, with the column each curve uses for them.
MATURITIES = {
    "5 Year": {"nominal": "5 Yr", "real": "5 YR"},
    "7 Year": {"nominal": "7 Yr", "real": "7 YR"},
    "10 Year": {"nominal": "10 Yr", "real": "10 YR"},
    "20 Year": {"nominal": "20 Yr", "real": "20 YR"},
    "30 Year": {"nominal": "30 Yr", "real": "30 YR"},
}

COUNTRY = "United States"


def _get_year(year: int, curve: str) -> pd.DataFrame:
    """
    Retrieves the daily nominal or real par yield curve of one year. A past year no
    longer changes, so it is cached under its own entry and only the current year is
    requested again.

    Args:
        year (int): The year to retrieve.
        curve (str): "nominal" or "real".

    Returns:
        pd.DataFrame: The yields as decimals, indexed by day with a column per maturity
        as the Treasury names them (e.g. "10 Yr", or "10 YR" for the real curve).
    """
    description = f"{year} Treasury {curve} par yield curve"

    def fetch() -> pd.DataFrame:
        response = get_request(
            BASE_URL.format(year=year, curve_type=CURVES[curve]["type"]), timeout=120
        )
        data = pd.read_csv(io.StringIO(response.text))
        require_columns(
            data, {"Date", "10 Yr" if curve == "nominal" else "10 YR"}, description
        )

        data = data.set_index("Date")
        data.index = pd.PeriodIndex(
            pd.to_datetime(data.index, format="%m/%d/%Y"), freq="D"
        )
        data.index.name = None

        # The Treasury publishes the yields in percent.
        return data.apply(pd.to_numeric, errors="coerce").sort_index() / 100

    current_year = datetime.now().year
    suffix = "" if curve == "nominal" else "_real"

    return collect_cached_data(
        source=policy_model.US_TREASURY,
        dataset=(
            f"par_yield_curve{suffix}"
            if year == current_year
            else f"par_yield_curve{suffix}_year"
        ),
        entity=str(year),
        fetch=fetch,
        description=description,
    )


def get_yield_curve(
    curve: str = "nominal", start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """
    Retrieves the daily nominal or real par yield curve of US Treasuries as published by
    the U.S. Department of the Treasury. Only the years between the start and end date
    are requested, one file per year.

    Args:
        curve (str): "nominal" (from 1990) or "real" (TIPS, from 2003). Defaults to
            "nominal".
        start_date (str | None): The start date (YYYY-MM-DD). Defaults to None, which
            starts at the first year of the curve.
        end_date (str | None): The end date (YYYY-MM-DD). Defaults to None, which ends in
            the current year.

    Returns:
        pd.DataFrame: The yields as decimals, indexed by business day with a column per
        maturity as the Treasury names them.
    """
    first_year = CURVES[curve]["first_year"]
    current_year = datetime.now().year

    first_year = max(first_year, int(start_date[:4]) if start_date else first_year)
    last_year = min(current_year, int(end_date[:4]) if end_date else current_year)
    years = list(range(first_year, last_year + 1))

    if not years:
        logger.warning(
            "The Treasury %s par yield curve is published from %s to today, so there is no "
            "data between %s and %s.",
            curve,
            CURVES[curve]["first_year"],
            start_date,
            end_date,
        )
        return pd.DataFrame()

    frames = helpers.run_in_parallel(_get_year, [(year, curve) for year in years])
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames).sort_index()


def get_long_term_interest_rate(
    start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """
    Retrieves the daily 10-year par yield of US Treasuries as published by the U.S.
    Department of the Treasury, from 1990 onwards.

    Args:
        start_date (str | None): The start date (YYYY-MM-DD). Defaults to None, which
            starts in 1990.
        end_date (str | None): The end date (YYYY-MM-DD). Defaults to None, which ends in
            the current year.

    Returns:
        pd.DataFrame: A single "United States" column with the yield as a decimal,
        indexed by business day.
    """
    yield_curve = get_yield_curve("nominal", start_date, end_date)

    if yield_curve.empty:
        return yield_curve

    return yield_curve[["10 Yr"]].rename(columns={"10 Yr": COUNTRY}).dropna()


def get_real_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily real par yield curve of Treasury Inflation-Protected Securities
    (TIPS) for the 5, 7, 10, 20 and 30-Year maturities, the same figures FRED republishes
    as the DFII series.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The real yields as decimals, indexed by business day ("Date") with a
        column per maturity ("5 Year" to "30 Year").
    """
    real_curve = get_yield_curve("real", start_date, end_date)

    if real_curve.empty:
        return real_curve

    real_yield_curve = pd.DataFrame(
        {
            maturity: real_curve.get(columns["real"])
            for maturity, columns in MATURITIES.items()
        },
        index=real_curve.index,
    )
    real_yield_curve.index.name = "Date"

    return real_yield_curve.dropna(how="all")


def get_breakeven_inflation_expectations(
    start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Computes the daily breakeven inflation rates, the nominal par yield minus the real
    par yield, for the 5, 7, 10, 20 and 30-Year maturities, plus the 5-Year, 5-Year
    Forward Inflation Expectation Rate. These are the formulas FRED uses for its T5YIE,
    T10YIE and T5YIFR series, applied to the same Treasury curves.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The breakeven rates as decimals, indexed by business day ("Date")
        with a column per maturity plus "5 Year, 5 Year Forward".
    """
    nominal_curve = get_yield_curve("nominal", start_date, end_date)
    real_curve = get_yield_curve("real", start_date, end_date)

    if nominal_curve.empty or real_curve.empty:
        return pd.DataFrame()

    index = nominal_curve.index.intersection(real_curve.index)
    breakeven_inflation = pd.DataFrame(
        {
            maturity: nominal_curve.loc[index, columns["nominal"]]
            - real_curve.loc[index, columns["real"]]
            for maturity, columns in MATURITIES.items()
            if columns["nominal"] in nominal_curve and columns["real"] in real_curve
        },
        index=index,
    )

    # The inflation rate the market expects on average over the five years that start
    # five years from now, from the 5 and 10-year breakevens compounded annually.
    breakeven_inflation["5 Year, 5 Year Forward"] = (
        (1 + breakeven_inflation["10 Year"]) ** 10
        / (1 + breakeven_inflation["5 Year"]) ** 5
    ) ** (1 / 5) - 1

    breakeven_inflation.index.name = "Date"

    return breakeven_inflation.dropna(how="all")
