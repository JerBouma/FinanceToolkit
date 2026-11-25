"""FHFA Model"""

__docformat__ = "google"


import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import (
    collect_cached_data,
    open_zip_archive,
    require_columns,
)
from financetoolkit.utilities.requests_model import get_request

# The outstanding residential mortgage statistics of the FHFA's National Mortgage Database
# (NMDB) for the United States, its census regions and divisions and its rural and
# non-rural areas, quarterly from 2013, in one zipped CSV with the full history.
OUTSTANDING_MORTGAGES_URL = (
    "https://www.fhfa.gov/document/d/nmdb/"
    "nmdb-outstanding-mortgage-statistics-national-census-areas-quarterly.zip"
)

# The contract interest rate series, as named in Appendix C of the NMDB Aggregate
# Statistics Data Dictionary: the average rate and the share of mortgages per rate bucket.
INTEREST_RATE_SERIES = {
    "AVE_INTRATE": "Average Rate",
    "PCT_INTRATE_LT_3": "Below 3%",
    "PCT_INTRATE_3_4": "3% to 4%",
    "PCT_INTRATE_4_5": "4% to 5%",
    "PCT_INTRATE_5_6": "5% to 6%",
    "PCT_INTRATE_GE_6": "6% and Above",
}

# VALUE1 is weighted by the number of mortgages and VALUE2 by their unpaid principal
# balance at the end of the quarter (Data Dictionary, fields 13 and 14).
WEIGHTINGS = {"VALUE1": "Loans", "VALUE2": "Balance"}


def get_outstanding_mortgage_interest_rates() -> pd.DataFrame:
    """
    Retrieve the contract interest rates of the residential mortgages outstanding at the end of each quarter from the
    FHFA's National Mortgage Database, from 2013 Q1.

    It gives the average rate and the share of mortgages below 3%, between 3% and 4%, 4% and 5%, 5% and 6%, and at or
    above 6%. The NMDB is a 1-in-20 sample of the closed-end first-lien mortgages reported to a national credit
    bureau.

    Returns:
        pd.DataFrame: The rates and shares as decimals (0.192 for 19.2%), indexed by
        quarter ("Date"), with the columns Region (the United States, a census region or
        division, or "United States Rural" and "United States Non-Rural"), Market ("All
        Mortgages", "Enterprise Acquisitions", "Government / Non-Conventional" or "Other
        Conventional Market"), Weighting ("Loans", weighted by the number of mortgages, or
        "Balance", weighted by their unpaid principal balance) and the series.
    """
    description = "outstanding mortgage interest rates"

    def fetch() -> pd.DataFrame:
        response = get_request(OUTSTANDING_MORTGAGES_URL, timeout=120)

        with open_zip_archive(
            response, f"FHFA archive for the {description}"
        ) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]

            if len(names) != 1:
                raise ValueError(
                    f"The FHFA archive for the {description} holds {len(names)} CSV "
                    "files instead of one, so its layout changed."
                )

            with archive.open(names[0]) as file:
                data = pd.read_csv(file, dtype=str, keep_default_na=False)

        require_columns(
            data,
            {"GEONAME", "MARKET", "PERIOD", "SUPPRESSED", "SERIESID", *WEIGHTINGS},
            description,
        )
        data = data[data["SERIESID"].isin(INTEREST_RATE_SERIES)]

        if data.empty:
            raise ValueError(
                f"The FHFA archive has none of the series of the {description}, so its "
                "layout changed."
            )

        # A statistic based on fewer than 3 loans is suppressed (SUPPRESSED = 1), so it
        # has no value.
        suppressed = data["SUPPRESSED"].str.strip() == "1"
        values = data.melt(
            id_vars=["GEONAME", "MARKET", "PERIOD", "SERIESID"],
            value_vars=list(WEIGHTINGS),
            var_name="Weighting",
            value_name="Value",
        )
        values["Value"] = pd.to_numeric(values["Value"], errors="coerce").where(
            ~pd.concat([suppressed] * len(WEIGHTINGS), ignore_index=True)
        )

        values["Date"] = pd.PeriodIndex(values["PERIOD"], freq="Q")
        values["Weighting"] = values["Weighting"].map(WEIGHTINGS)
        values["Series"] = values["SERIESID"].map(INTEREST_RATE_SERIES)

        table = values.pivot_table(
            index="Date",
            columns=["GEONAME", "MARKET", "Weighting", "Series"],
            values="Value",
            aggfunc="first",
            dropna=False,
        )
        table.columns.names = ["Region", "Market", "Weighting", "Series"]

        # The NMDB publishes the rates and shares in percent.
        return (table / 100).sort_index()

    return collect_cached_data(
        source=policy_model.FEDERAL_HOUSING_FINANCE_AGENCY,
        dataset="outstanding_mortgages",
        entity="interest_rates",
        fetch=fetch,
        description=description,
    )
