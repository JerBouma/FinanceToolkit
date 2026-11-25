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


def _get_year(year: int) -> pd.DataFrame:
    """
    Retrieves the clearing price of the auctions of general allowances of one year,
    averaged over the auctions held on the same day.

    Args:
        year (int): The year.

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
        # get_carbon_price reports the years the EEX could not be reached for in one line.
        quiet=True,
    )


def get_carbon_price(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the clearing price of the primary auctions of EU emission allowances, the price of emitting one tonne of
    CO2 under the EU Emissions Trading System, daily from 2020.

    Only the years between the start and end date are retrieved, one workbook per year. Once
    the EEX cannot be reached, the later years are not requested (they would fail alike, each
    after its own retries), and the years without prices are reported in one line.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The price in euro per tonne of CO2, indexed by day with a "European
        Union" column.
    """
    first = max(pd.Timestamp(start_date).year, FIRST_YEAR)
    last = min(pd.Timestamp(end_date).year, pd.Timestamp.now().year)

    with stop_at_unreachable_sources():
        years = {year: _get_year(year) for year in range(first, last + 1)}
        unreachable = is_unreachable(policy_model.EEX)

    if unreachable and (
        missing := [str(year) for year, prices in years.items() if prices.empty]
    ):
        logger.error(
            "Could not reach the EEX for the emission allowance auctions of %s, so no carbon "
            "price is returned for %s.",
            ", ".join(missing),
            "that year" if len(missing) == 1 else "those years",
        )

    frames = [prices for prices in years.values() if not prices.empty]

    return pd.concat(frames).sort_index() if frames else pd.DataFrame()
