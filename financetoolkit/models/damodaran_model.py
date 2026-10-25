"""Aswath Damodaran Synthetic Rating Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.excel_model import read_excel
from financetoolkit.utilities.requests_model import get_request

# Aswath Damodaran's synthetic rating workbook (NYU Stern), which maps a company's interest
# coverage ratio to a bond rating and the default spread of that rating. He updates the
# spreads every January.
RATINGS_URL = "https://pages.stern.nyu.edu/~adamodar/pc/ratings.xls"

# The title above each table, the column the table starts in, and the firm type it covers.
TABLES = {
    "For large non-financial service firms": (0, "Large"),
    "For smaller and riskier firms": (0, "Small"),
    "For financial service firms": (5, "Financial"),
}

# Each table starts three rows below its title and has one row per rating.
ROWS_BELOW_TITLE = 3
RATINGS_PER_TABLE = 15


def get_synthetic_rating_table() -> pd.DataFrame:
    """
    Retrieves Aswath Damodaran's synthetic rating tables: per firm type (large
    non-financial, smaller and riskier, financial service firms), the interest coverage
    range of every bond rating and the default spread of that rating.

    Returns:
        pd.DataFrame: One row per firm type and rating with the Lower and Upper bound of the
        interest coverage ratio, the Rating (Moody's/S&P notation) and the Default Spread as
        a decimal.

    Raises:
        ValueError: When the workbook's layout changed.
    """
    description = "Damodaran synthetic rating table"

    def fetch() -> pd.DataFrame:
        sheet = read_excel(
            get_request(RATINGS_URL, timeout=60).content, sheet_name=0, header=None
        )
        tables = []

        for title, (column, firm_type) in TABLES.items():
            matches = sheet.index[
                sheet[column].astype(str).str.strip().str.startswith(title)
            ]
            if matches.empty:
                raise ValueError(
                    f"The {description} has no '{title}' table, which means its layout changed."
                )
            start = matches[0] + ROWS_BELOW_TITLE
            rows = sheet.iloc[
                start : start + RATINGS_PER_TABLE, column : column + 4
            ].copy()
            rows.columns = ["Lower", "Upper", "Rating", "Default Spread"]
            rows[["Lower", "Upper", "Default Spread"]] = rows[
                ["Lower", "Upper", "Default Spread"]
            ].apply(pd.to_numeric, errors="coerce")
            rows = rows.dropna(subset=["Lower", "Upper", "Default Spread"])
            if len(rows) != RATINGS_PER_TABLE:
                raise ValueError(
                    f"The '{title}' table of the {description} does not have "
                    f"{RATINGS_PER_TABLE} ratings, which means its layout changed."
                )
            rows.insert(0, "Firm Type", firm_type)
            tables.append(rows)

        return pd.concat(tables).reset_index(drop=True)

    return collect_cached_data(
        source=policy_model.DAMODARAN,
        dataset="synthetic_ratings",
        entity="ratings",
        fetch=fetch,
        description=description,
    )
