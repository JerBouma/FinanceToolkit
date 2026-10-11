"""Jordà-Schularick-Taylor Macrohistory Database Model"""

__docformat__ = "google"

import io

import pandas as pd
from pandas.io.stata import StataReader

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.requests_model import get_request

# Release 6 of the Jordà-Schularick-Taylor Macrohistory Database: 18 advanced economies,
# yearly from 1870 to 2020. The link names a workbook but serves a Stata file.
DATASET_URL = "https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.xlsx"

# The total (nominal) returns of the asset classes, as decimals.
ASSET_CLASSES = {
    "equity": "eq_tr",
    "housing": "housing_tr",
    "bonds": "bond_tr",
    "bills": "bill_rate",
    "risky": "risky_tr",
    "safe": "safe_tr",
    "wealth": "capital_tr",
}

NAMES = {"UK": "United Kingdom", "USA": "United States"}


def get_dataset() -> pd.DataFrame:
    """
    Retrieves the Macrohistory Database with a column per variable and country.

    Returns:
        pd.DataFrame: The variables as published, indexed by year with a (variable,
        country) column.

    Raises:
        ValueError: When the file misses the country or year.
    """
    description = "Jordà-Schularick-Taylor Macrohistory Database"

    def fetch() -> pd.DataFrame:
        response = get_request(DATASET_URL, timeout=120)

        with StataReader(io.BytesIO(response.content)) as reader:
            data = reader.read()

        require_columns(data, {"country", "year"}, description)
        data["country"] = data["country"].replace(NAMES)
        data["year"] = pd.PeriodIndex(data["year"].astype(int).astype(str), freq="Y")
        numeric = data.drop(columns=["iso", "ifs"], errors="ignore").set_index(
            ["year", "country"]
        )
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        dataset = numeric.unstack(level=1).sort_index(axis=1)
        dataset.index.name = None

        return dataset

    return collect_cached_data(
        source=policy_model.MACROHISTORY,
        dataset="dataset",
        entity="JST_R6",
        fetch=fetch,
        description=description,
    )


def get_asset_returns(asset_class: str, real: bool) -> pd.DataFrame:
    """
    Retrieves the yearly total return of an asset class per country.

    Args:
        asset_class (str): A key of ASSET_CLASSES, e.g. "equity".
        real (bool): Whether to deflate the return by consumer price inflation.

    Returns:
        pd.DataFrame: The returns as decimals, indexed by year with a column per country.
    """
    dataset = get_dataset()

    if dataset.empty:
        return dataset

    returns = dataset[ASSET_CLASSES[asset_class]]

    if real:
        inflation = dataset["cpi"].pct_change(fill_method=None)
        returns = (1 + returns) / (1 + inflation) - 1

    returns = returns.dropna(how="all").dropna(how="all", axis=1)
    returns.columns.name = None

    return returns
