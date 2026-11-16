"""U.S. Census Bureau Model"""

__docformat__ = "google"

import io
import re

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, to_us_series
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request

# The monthly retail trade and food services sales of the Advance Monthly Retail Trade
# Survey (category 44X72), the source of FRED's RSAFS series. The file holds the
# seasonally adjusted sales by year and month, followed by the seasonal factors.
RETAIL_SALES_URL = "https://www.census.gov/retail/marts/www/adv44X72.txt"

# A data row holds the year and the twelve months: "2025   687058    683772 ..."
RETAIL_ROW_PATTERN = re.compile(r"^(\d{4})\s+(.*)$")

# The monthly housing starts of the Survey of Construction, the source of FRED's HOUST
# series: one sheet per adjustment, with the date in the first column and the United
# States total in the second.
HOUSING_STARTS_URL = "https://www.census.gov/construction/nrc/xls/starts_cust.xlsx"
HOUSING_STARTS_SHEET = "Seasonally Adjusted"


def get_retail_sales() -> pd.DataFrame:
    """
    Retrieves the monthly advance retail trade and food services sales, in millions of
    dollars and seasonally adjusted, from the U.S. Census Bureau, from 1992.

    Returns:
        pd.DataFrame: A single "United States" column indexed by the first day of each
        month ("Date").
    """
    description = "retail sales"

    def fetch() -> pd.DataFrame:
        text = get_request(
            RETAIL_SALES_URL, timeout=60, extra_headers=TOOLKIT_HEADERS
        ).text
        # The seasonal factors follow the sales under their own heading.
        sales = text.split("SEASONAL FACTORS")[0]
        observations = {}

        for line in sales.splitlines():
            match = RETAIL_ROW_PATTERN.match(line.strip())
            if match is None:
                continue

            # Months not published yet are left out of the row.
            for month, value in enumerate(match.group(2).split()[:12], start=1):
                number = pd.to_numeric(value, errors="coerce")
                if pd.notna(number):
                    observations[pd.Period(f"{match.group(1)}-{month:02d}-01", "D")] = (
                        number
                    )

        return to_us_series(observations, "Census Bureau file", description)

    return collect_cached_data(
        source=policy_model.CENSUS_BUREAU,
        dataset="series",
        entity="retail_sales",
        fetch=fetch,
        description=description,
    )


def get_housing_starts() -> pd.DataFrame:
    """
    Retrieves the monthly new privately-owned housing units started, in thousands of
    units at a seasonally adjusted annual rate, from the U.S. Census Bureau, from 1959.

    Returns:
        pd.DataFrame: A single "United States" column indexed by the first day of each
        month ("Date").
    """
    description = "housing starts"

    def fetch() -> pd.DataFrame:
        content = get_request(
            HOUSING_STARTS_URL, timeout=60, extra_headers=TOOLKIT_HEADERS
        ).content
        sheet = pd.read_excel(
            io.BytesIO(content), sheet_name=HOUSING_STARTS_SHEET, header=None
        )

        # The heading and the notes around the table have no date in the first column.
        dates = pd.to_datetime(sheet[0], errors="coerce", format="mixed")
        values = pd.to_numeric(sheet[1], errors="coerce")
        rows = dates.notna() & values.notna()

        observations = {
            pd.Period(date, freq="D"): value
            for date, value in zip(dates[rows], values[rows], strict=True)
        }

        return to_us_series(observations, "Census Bureau file", description)

    return collect_cached_data(
        source=policy_model.CENSUS_BUREAU,
        dataset="series",
        entity="housing_starts",
        fetch=fetch,
        description=description,
    )
