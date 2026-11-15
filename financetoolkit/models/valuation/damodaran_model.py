"""Aswath Damodaran Synthetic Rating Model and Industry Datasets"""

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


# Aswath Damodaran's industry datasets (NYU Stern), updated every January from the
# previous year's figures. The US file and the regional files of a dataset share one layout.
INDUSTRY_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/{file}.xls"

# Each dataset's file name for US companies and the prefix of its regional files.
INDUSTRY_DATASETS = {
    "betas": ("betas", "beta"),
    "total_betas": ("totalbeta", "totalbeta"),
    "cost_of_capital": ("wacc", "wacc"),
    "margins": ("margin", "margin"),
    "price_to_earnings": ("pedata", "pe"),
    "price_to_book": ("pbvdata", "pbv"),
    "price_to_sales": ("psdata", "ps"),
    "enterprise_value_multiples": ("vebitda", "vebitda"),
}

# Each region and the suffix of its file; the US files have none.
INDUSTRY_REGIONS = {
    "United States": "",
    "Europe": "Europe",
    "Japan": "Japan",
    "Australia, New Zealand and Canada": "Rest",
    "Emerging Markets": "emerg",
    "China": "China",
    "India": "India",
    "Global": "Global",
}


def get_industry_file_name(dataset: str, region: str) -> str:
    """
    Returns the file name of one of Aswath Damodaran's industry datasets for a region.

    Args:
        dataset (str): The dataset, one of the keys of INDUSTRY_DATASETS.
        region (str): The region, one of the keys of INDUSTRY_REGIONS.

    Returns:
        str: The file name without its extension, e.g. "waccEurope".

    Raises:
        ValueError: When the dataset or region is not available.
    """
    if dataset not in INDUSTRY_DATASETS:
        raise ValueError(
            f"The dataset must be one of {', '.join(INDUSTRY_DATASETS)}, not '{dataset}'."
        )
    if region not in INDUSTRY_REGIONS:
        raise ValueError(
            f"The region must be one of {', '.join(INDUSTRY_REGIONS)}, not '{region}'."
        )

    us_file, regional_prefix = INDUSTRY_DATASETS[dataset]
    suffix = INDUSTRY_REGIONS[region]

    return regional_prefix + suffix if suffix else us_file


def parse_industry_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Reads the "Industry Averages" worksheet of one of Aswath Damodaran's industry datasets.

    The worksheet opens with a block of notes, followed by a header row that starts with
    "Industry Name" and one row per industry, ending with the market totals. Columns
    that share a name (such as EV/EBITDA for firms with a positive EBITDA and for all
    firms) are told apart by the group label above them. The yearly history columns at
    the end of the beta files are left out: their label differs per region (unlevered
    beta corrected for cash in the US file, levered beta elsewhere), so they cannot be
    named reliably.

    Args:
        sheet (pd.DataFrame): The worksheet, read without a header.

    Returns:
        pd.DataFrame: One row per industry (and the market totals), indexed by Industry,
        with Damodaran's figures as he reports them: rates, margins and returns as
        decimals, betas and multiples as numbers.

    Raises:
        ValueError: When the worksheet has no "Industry Name" header row, which means its
            layout changed.
    """
    first_column = sheet[0].astype(str).str.strip()
    header_rows = sheet.index[first_column == "Industry Name"]
    if header_rows.empty:
        raise ValueError(
            "The Damodaran industry dataset has no 'Industry Name' header row, "
            "which means its layout changed."
        )
    header_row = header_rows[0]
    position = sheet.index.get_loc(header_row)

    names = [" ".join(str(name).split()) for name in sheet.iloc[position]]
    groups = (
        sheet.iloc[position - 1].ffill().astype(str).tolist()
        if position > 0
        else [""] * len(names)
    )

    columns = {}
    for index, name in enumerate(names[1:], start=1):
        if name == "nan":
            continue
        # A year such as 2023 or 2023.0 heads a column of the yearly beta history, which
        # ends with its average, e.g. "Average (2022-2026)".
        if name.replace(".0", "").isdigit() or (
            name.startswith("Average")
            and any(character.isdigit() for character in name)
        ):
            continue
        # The files write "Number of firms" and "Number of Firms".
        columns[index] = (
            "Number of Firms" if name.lower() == "number of firms" else name
        )

    duplicated = {
        name for name in columns.values() if list(columns.values()).count(name) > 1
    }
    for index, name in columns.items():
        if name in duplicated:
            columns[index] = f"{name} ({' '.join(groups[index].split())})"

    data = sheet.iloc[position + 1 :, [0, *columns]].copy()
    data.columns = ["Industry", *columns.values()]
    data["Industry"] = data["Industry"].astype(str).str.strip()
    data = data[~data["Industry"].isin(["", "nan"])]
    # The European files call the market total "Grand Total", the others "Total Market".
    data["Industry"] = data["Industry"].replace("Grand Total", "Total Market")

    data = data.set_index("Industry")
    data = data.apply(pd.to_numeric, errors="coerce")

    return data.dropna(how="all")


def get_industry_statistics(dataset: str, region: str) -> pd.DataFrame:
    """
    Retrieves one of Aswath Damodaran's industry datasets for a region.

    Args:
        dataset (str): The dataset, one of the keys of INDUSTRY_DATASETS.
        region (str): The region, one of the keys of INDUSTRY_REGIONS.

    Returns:
        pd.DataFrame: One row per industry (and the market totals), indexed by Industry.

    Raises:
        ValueError: When the dataset or region is not available, or the workbook's layout
            changed.
    """
    file_name = get_industry_file_name(dataset, region)
    description = f"Damodaran {dataset.replace('_', ' ')} dataset for {region}"

    def fetch() -> pd.DataFrame:
        sheet = read_excel(
            get_request(INDUSTRY_URL.format(file=file_name), timeout=60).content,
            sheet_name="Industry Averages",
            header=None,
        )

        return parse_industry_sheet(sheet)

    return collect_cached_data(
        source=policy_model.DAMODARAN,
        dataset="industry_statistics",
        entity=file_name,
        fetch=fetch,
        description=description,
    )
