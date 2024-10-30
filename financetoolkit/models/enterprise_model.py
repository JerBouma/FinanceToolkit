"""Enterprise Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.ratios import valuation_model


def get_enterprise_value_breakdown(
    share_price: float | pd.Series,
    shares_outstanding: float | pd.Series,
    total_debt: float | pd.Series,
    minority_interest: float | pd.Series,
    preferred_equity: float | pd.Series,
    cash_and_cash_equivalents: float | pd.Series,
) -> pd.DataFrame:
    """
    The Enterprise Value breakdown corresponds to the following components:
        - Share Price: given for each quarter or year.
        - Market cap: The total value of a company's outstanding common and preferred shares
        - Debt: The sum of long-term and short-term debt
        - Preferred equity: The value of preferred shares
        - Minority interest: The equity value of a subsidiary with less than 50% ownership.
        - Cash and cash equivalents: The total amount of cash, certificates of
        deposit, drafts, money orders, commercial paper, marketable securities, money market
        funds, short-term government bonds, or Treasury bills a company possesses.

    Args:
        share_price (float | pd.Series): The share price of the company.
        shares_outstanding (float | pd.Series): The total shares outstanding of the company.
        total_debt (float | pd.Series): The total debt of the company.
        minority_interest (float | pd.Series): The minority interest of the company.
        preferred_equity (float | pd.Series): The preferred equity of the company.
        cash_and_cash_equivalents (float | pd.Series): The cash and cash equivalents of the company.

    Returns:
        pd.DataFrame: the Enterprise Value breakdown.
    """
    # Calculate the net profit margin
    market_cap = valuation_model.get_market_cap(
        share_price=share_price, total_shares_outstanding=shares_outstanding
    )

    # Calculate the market cap
    enterprise_value = valuation_model.get_enterprise_value(
        market_cap=market_cap,
        total_debt=total_debt,
        minority_interest=minority_interest,
        preferred_equity=preferred_equity,
        cash_and_cash_equivalents=cash_and_cash_equivalents,
    )

    # Create a dictionary with the Dupont analysis components
    components = {
        "Share Price": share_price,
        "Market Capitalization": market_cap,
        "Total Debt": total_debt,
        "Minority Interest": minority_interest,
        "Preferred Equity": preferred_equity,
        "Cash and Cash Equivalents": cash_and_cash_equivalents,
        "Enterprise Value": enterprise_value,
    }

    if isinstance(enterprise_value, pd.DataFrame):
        return (
            pd.concat(components)
            .swaplevel(1, 0)
            .sort_index(level=0, sort_remaining=False)
        )

    return pd.DataFrame.from_dict(components, orient="index")
