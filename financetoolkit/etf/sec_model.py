"""SEC Fund Prospectus Fee Model"""

__docformat__ = "google"

import csv
import io
import os
import zipfile
from datetime import datetime

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

# The SEC publishes the fee tables of every fund prospectus (Form N-1A) each quarter, and a
# list of the tickers of every fund share class.
RISK_RETURN_URL = (
    "https://www.sec.gov/files/dera/data/mutual-fund-prospectus-risk/"
    "return-summary-data-sets/{year}q{quarter}_rr1.zip"
)
FUND_TICKERS_URL = "https://www.sec.gov/files/company_tickers_mf.json"
NOT_PUBLISHED = 404

# The SEC asks every automated request to identify itself with a name and an email address.
USER_AGENT_ENVIRONMENT_VARIABLE = "FINANCETOOLKIT_SEC_USER_AGENT"

# The fee tags of the prospectus fee table, with the names the Finance Toolkit gives them,
# kept per share class; the portfolio turnover is given per fund (series).
FEE_TAGS = {
    "ManagementFeesOverAssets": "Management Fee",
    "DistributionAndService12b1FeesOverAssets": "Distribution and Service (12b-1) Fee",
    "OtherExpensesOverAssets": "Other Expenses",
    "AcquiredFundFeesAndExpensesOverAssets": "Acquired Fund Fees",
    "ExpensesOverAssets": "Gross Expense Ratio",
    "FeeWaiverOrReimbursementOverAssets": "Fee Waiver",
    "NetExpensesOverAssets": "Net Expense Ratio",
}
TURNOVER_TAG = "PortfolioTurnoverRate"
# A fund updates its prospectus once a year, so a year and a quarter of quarterly files covers
# every fund.
QUARTERS_BACK = 5


def resolve_user_agent(user_agent: str | None = None) -> str:
    """
    The User-Agent the SEC requests: the one given, or the FINANCETOOLKIT_SEC_USER_AGENT
    environment variable.

    Args:
        user_agent (str | None, optional): A name and email address, e.g.
            "Jane Doe jane.doe@example.com". Defaults to None.

    Returns:
        str: The User-Agent.

    Raises:
        ValueError: When neither is set.
    """
    user_agent = user_agent or os.environ.get(USER_AGENT_ENVIRONMENT_VARIABLE, "")
    if "@" not in user_agent:
        raise ValueError(
            "The SEC asks every automated request to identify itself with a name and an email "
            "address. Pass sec_user_agent='Your Name your.email@example.com' or set the "
            f"{USER_AGENT_ENVIRONMENT_VARIABLE} environment variable."
        )
    return user_agent


def get_fund_classes(user_agent: str) -> pd.DataFrame:
    """
    Retrieves the SEC's list of fund share classes with their ticker.

    Args:
        user_agent (str): The User-Agent, see resolve_user_agent.

    Returns:
        pd.DataFrame: The CIK, Series and Class of every ticker.
    """

    def fetch() -> pd.DataFrame:
        response = get_request(
            FUND_TICKERS_URL, timeout=60, extra_headers={"User-Agent": user_agent}
        ).json()
        classes = pd.DataFrame(response["data"], columns=response["fields"])
        classes = classes.rename(
            columns={
                "cik": "CIK",
                "seriesId": "Series",
                "classId": "Class",
                "symbol": "Symbol",
            }
        )
        return classes.drop_duplicates("Symbol").set_index("Symbol")

    return collect_cached_data(
        source=policy_model.SEC,
        dataset="fund_tickers",
        entity="company_tickers_mf",
        fetch=fetch,
        description="SEC fund tickers",
    )


def _filings(archive: zipfile.ZipFile) -> dict[str, tuple[str, str, str]]:
    """
    The acceptance time, filing date and prospectus date (both YYYYMMDD) of every filing
    (adsh) in a file. The prospectus date is the date on the cover of the prospectus, the one
    its fee table speaks for; the period end date of the facts (ddate) is often no fiscal
    year end at all, so it is not reported.
    """
    if "sub.tsv" not in archive.namelist():
        return {}
    with archive.open("sub.tsv") as file:
        reader = csv.DictReader(
            io.TextIOWrapper(file, encoding="utf-8", errors="replace"), delimiter="\t"
        )
        return {
            row["adsh"]: (
                row.get("accepted") or "",
                row.get("filed") or "",
                row.get("pdate") or "",
            )
            for row in reader
        }


def read_risk_return_fees(zip_data: bytes) -> pd.DataFrame:
    """
    Reads the fee table of every share class from a quarterly risk/return summary file,
    reading the numbers line by line so the (large) file is never held as a table. A class
    can be in several filings of the quarter (the annual prospectus and supplements that
    change the fees), so only the latest accepted filing of each class is kept, and the
    portfolio turnover of each fund (series) likewise.

    Args:
        zip_data (bytes): The quarterly ZIP file.

    Returns:
        pd.DataFrame: The fees per share class, with the Series, the Prospectus Date (the
        date on the cover of the prospectus, empty when the filer gave none), the Filing Date
        and the Portfolio Turnover of its fund.
    """
    fees: dict[str, dict] = {}
    turnover: dict[str, tuple[tuple[str, str], float]] = {}
    with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
        filings = _filings(archive)
        with archive.open("num.tsv") as file:
            reader = csv.DictReader(
                io.TextIOWrapper(file, encoding="utf-8", errors="replace"),
                delimiter="\t",
            )
            for row in reader:
                tag = row["tag"]
                if tag not in FEE_TAGS and tag != TURNOVER_TAG:
                    continue
                try:
                    value = float(row["value"])
                except ValueError:
                    continue
                accepted, filed, prospectus = filings.get(row["adsh"], ("", "", ""))
                # Filings of one fiscal year share the period end date of their facts
                # (ddate), so the acceptance time decides which is the latest.
                key = (accepted, row["ddate"])
                if tag == TURNOVER_TAG:
                    series = row["series"]
                    if series and (
                        series not in turnover or key >= turnover[series][0]
                    ):
                        turnover[series] = (key, value)
                    continue
                share_class = row["class"] or next(
                    (
                        part.split("=", 1)[1]
                        for part in row["otherdims"].split(";")
                        if part.startswith("Class=")
                    ),
                    "",
                )
                if not share_class:
                    continue
                record = fees.get(share_class)
                if record is None or accepted > record["Accepted"]:
                    record = fees[share_class] = {
                        "Series": row["series"],
                        "Accepted": accepted,
                        "Period": row["ddate"],
                        "Prospectus Date": prospectus,
                        "Filing Date": filed,
                    }
                elif accepted < record["Accepted"]:
                    continue
                name = FEE_TAGS[tag]
                # Within a filing a fact of a later period replaces an earlier one.
                if name not in record or row["ddate"] >= record["Period"]:
                    record[name] = value
                    record["Period"] = max(record["Period"], row["ddate"])

    if not fees:
        return pd.DataFrame()

    table = pd.DataFrame.from_dict(fees, orient="index").drop(
        columns=["Accepted", "Period"]
    )
    for column in ("Prospectus Date", "Filing Date"):
        table[column] = pd.to_datetime(
            table[column], format="%Y%m%d", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    table["Portfolio Turnover"] = table["Series"].map(
        {series: value for series, (_, value) in turnover.items()}
    )
    return table


def _quarter_fees(year: int, quarter: int, user_agent: str) -> pd.DataFrame:
    """The fees of every share class in the risk/return summary file of a quarter."""
    description = f"SEC risk/return summary of {year}Q{quarter}"

    def fetch() -> pd.DataFrame:
        try:
            response = get_request(
                RISK_RETURN_URL.format(year=year, quarter=quarter),
                timeout=300,
                extra_headers={"User-Agent": user_agent},
            )
        except requests.exceptions.HTTPError as error:
            # A quarter is published a few weeks after it ends.
            if (
                error.response is not None
                and error.response.status_code == NOT_PUBLISHED
            ):
                return pd.DataFrame()
            raise
        return read_risk_return_fees(response.content)

    return collect_cached_data(
        source=policy_model.SEC,
        dataset="risk_return_fees",
        entity=f"{year}q{quarter}",
        fetch=fetch,
        description=description,
        expect_empty=True,
    )


def get_fee_breakdown(
    symbols: list[str], user_agent: str, today: datetime | None = None
) -> pd.DataFrame:
    """
    Retrieves the fee table of the prospectus of every fund share class: the management fee,
    the distribution and service (12b-1) fee, the other expenses, the fees of the funds it
    invests in, the gross expense ratio, the fee waiver (negative, as the fee table reports
    it), the net expense ratio and the portfolio turnover of the fund, as decimals. The
    quarterly files are read from the latest back, up to five quarters, until every fund is
    found.

    Args:
        symbols (list[str]): The tickers.
        user_agent (str): The User-Agent, see resolve_user_agent.
        today (datetime | None, optional): The date to count the quarters back from. Defaults
            to None, which is today.

    Returns:
        pd.DataFrame: The fees per ticker found.
    """
    classes = get_fund_classes(user_agent)
    wanted = {
        symbol: classes.loc[symbol, "Class"]
        for symbol in symbols
        if symbol in classes.index
    }
    found: dict[str, pd.Series] = {}

    today = today or datetime.now()
    year, quarter = today.year, (today.month - 1) // 3 + 1
    for _ in range(QUARTERS_BACK):
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
        remaining = {s: c for s, c in wanted.items() if s not in found}
        if not remaining:
            break
        fees = _quarter_fees(year, quarter, user_agent)
        for symbol, share_class in remaining.items():
            if not fees.empty and share_class in fees.index:
                found[symbol] = fees.loc[share_class]

    if not found:
        return pd.DataFrame()

    breakdown = pd.DataFrame(found).T
    numbers = breakdown.reindex(
        columns=["Gross Expense Ratio", "Net Expense Ratio", "Fee Waiver"]
    ).apply(pd.to_numeric, errors="coerce")
    gross, net, waiver = (numbers[column] for column in numbers)
    # The fee table defines the net expense ratio as the gross one plus the (negative)
    # waiver, but some filers enter the waiver as a positive reduction: with both ratios and
    # a waiver reported the waiver is their difference, otherwise the reported waiver is made
    # negative (a missing gross ratio must not lose it).
    both = net.notna() & gross.notna() & waiver.notna()
    waiver = (net - gross).round(6).where(both, -waiver.abs())
    breakdown["Fee Waiver"] = waiver
    # A missing ratio follows from the other and the waiver, if any.
    breakdown["Gross Expense Ratio"] = gross.fillna(net - waiver.fillna(0))
    breakdown["Net Expense Ratio"] = net.fillna(gross + waiver.fillna(0))
    breakdown["Class"] = [wanted[symbol] for symbol in breakdown.index]
    columns = [
        *FEE_TAGS.values(),
        "Portfolio Turnover",
        "Prospectus Date",
        "Filing Date",
        "Series",
        "Class",
    ]
    return breakdown.reindex(columns=columns)
