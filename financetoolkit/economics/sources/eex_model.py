"""European Energy Exchange (EEX) Model"""

__docformat__ = "google"


import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import (
    collect_cached_data,
    is_unreachable,
    require_columns,
    stop_at_unreachable_sources,
)
from financetoolkit.utilities.excel_model import read_excel
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.requests_model import get_request

logger = get_logger()

# The EEX holds the primary auctions of EU emission allowances (EUAs) for the European
# Union, Germany and Poland, almost every trading day, and publishes their results in one
# workbook per year, from 2020.
AUCTION_REPORT_URL = (
    "https://public.eex-group.com/eex/eua-auction-report/"
    "emission-spot-primary-market-auction-report-{year}-data.xlsx"
)
FIRST_YEAR = 2020

# The rows above the column names, the contract of the general allowances (aviation
# allowances, EUAA, are a separate contract) and the column with the clearing price.
HEADER_ROW = 5
GENERAL_ALLOWANCES = "T3PA"
PRICE_COLUMN = "Auction Price €/tCO2"

# The status of a year's report that is not published (yet).
NOT_PUBLISHED = 404

# The reasons a year has no prices although the EEX answered, reported as a warning rather than an error.  # noqa: E501
NOT_PUBLISHED_REASON = "the EEX has not published that year's report"
NO_AUCTIONS_REASON = "the EEX report holds no successful auctions"


def _get_year(year: int, failures: dict[str, str]) -> pd.DataFrame:
    """
    Retrieves the clearing price of the auctions of general allowances of one year,
    averaged over the auctions held on the same day.

    Args:
        year (int): The year.
        failures (dict[str, str]): Where the reason is recorded, per year, when the year's
            report could not be retrieved or is not published.

    Returns:
        pd.DataFrame: The price in euro per tonne of CO2, indexed by day with a "European
        Union" column.
    """
    current_year = year == pd.Timestamp.now().year
    description = f"EEX emission allowance auctions of {year}"

    def fetch() -> pd.DataFrame:
        try:
            response = get_request(AUCTION_REPORT_URL.format(year=year), timeout=120)
        except requests.exceptions.HTTPError as error:
            if (
                error.response is not None
                and error.response.status_code == NOT_PUBLISHED
            ):
                failures[str(year)] = NOT_PUBLISHED_REASON
                return pd.DataFrame()
            raise

        data = read_excel(response.content, description, header=HEADER_ROW)

        require_columns(data, {"Date", "Contract", "Status", PRICE_COLUMN}, description)
        data = data[
            (data["Contract"] == GENERAL_ALLOWANCES)
            & (data["Status"].astype(str).str.lower() == "successful")
        ]
        prices = pd.to_numeric(data[PRICE_COLUMN], errors="coerce")
        prices.index = pd.PeriodIndex(pd.to_datetime(data["Date"]), freq="D")

        return prices.groupby(level=0).mean().sort_index().to_frame("European Union")

    return collect_cached_data(
        source=policy_model.EEX,
        dataset="auction_current_year" if current_year else "auction_year",
        entity=str(year),
        fetch=fetch,
        description=description,
        expect_empty=True,
        failures=failures,
    )


def get_carbon_price(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the clearing price of the primary auctions of EU emission allowances, the price of emitting one tonne of
    CO2 under the EU Emissions Trading System, daily from 2020.

    Only the years between the start and end date are retrieved, one workbook per year. Once
    the EEX cannot be reached, the later years are not requested (they would fail alike, each
    after its own retries), and the years without prices are reported in one line with their
    reasons: an error when a request failed, a warning when the EEX published no results.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The price in euro per tonne of CO2, indexed by day with a "European
        Union" column.
    """
    first = max(pd.Timestamp(start_date).year, FIRST_YEAR)
    last = min(pd.Timestamp(end_date).year, pd.Timestamp.now().year)

    failures: dict[str, str] = {}

    with stop_at_unreachable_sources():
        years = {year: _get_year(year, failures) for year in range(first, last + 1)}
        unreachable = is_unreachable(policy_model.EEX)

    # The years after the one the EEX could not be reached for were not requested, so they share its reason.  # noqa: E501
    last_failure = max(map(int, failures), default=None)
    years_per_reason: dict[str, list[str]] = {}
    for year, prices in years.items():
        if not prices.empty:
            continue
        if str(year) in failures:
            reason = failures[str(year)]
        elif unreachable and last_failure is not None and year > last_failure:
            reason = failures[str(last_failure)]
        else:
            reason = NO_AUCTIONS_REASON
        years_per_reason.setdefault(reason, []).append(str(year))

    if years_per_reason:
        published_nothing = set(years_per_reason) <= {
            NOT_PUBLISHED_REASON,
            NO_AUCTIONS_REASON,
        }
        (logger.warning if published_nothing else logger.error)(
            "No carbon price is returned for the EEX emission allowance auctions of %s.",
            "; ".join(
                f"{', '.join(missing)} ({reason})"
                for reason, missing in years_per_reason.items()
            ),
        )

    frames = [prices for prices in years.values() if not prices.empty]

    return pd.concat(frames).sort_index() if frames else pd.DataFrame()
