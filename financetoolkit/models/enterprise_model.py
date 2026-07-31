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


def get_tobins_q_ratio(
    market_value_of_equity: float | pd.Series | pd.DataFrame,
    total_liabilities: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Tobin's Q Ratio is a financial metric, developed by economist James Tobin, that compares
    the market value of a company to the cost of replacing its assets. It is used as a
    valuation gauge for the company as a whole as well as an indicator of management's
    incentive to invest: a high Q suggests the market values the firm well above the cost of
    replacing its assets (making new investment attractive), while a low Q suggests the
    opposite.

    The formula is as follows:

        - Tobin's Q Ratio = (Market Value of Equity + Total Liabilities) / Total Assets

    Also known as: Tobin's Q, Q ratio.

    Args:
        market_value_of_equity (float | pd.Series | pd.DataFrame): The market value of a
        company's equity, i.e. the share price multiplied by the total shares outstanding.
        total_liabilities (float | pd.Series | pd.DataFrame): The sum of a company's current
        and non-current liabilities.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and
        non-current assets.

    Returns:
        float | pd.Series | pd.DataFrame: Tobin's Q Ratio.

    Notes:
    - A Q ratio greater than 1 indicates that the market values the company above the
    (book) cost of replacing its assets, which can reflect growth expectations, intangible
    assets not fully captured on the balance sheet (e.g. brand value, intellectual property),
    or overvaluation.
    - A Q ratio less than 1 indicates that the market values the company below the (book)
    cost of replacing its assets, which can reflect undervaluation, declining growth
    prospects, or assets that are worth less than their book value.
    - This implementation uses two simplifications relative to Tobin's original formulation:
    the market value of debt is approximated with the book value of Total Liabilities (rather
    than the market value of debt), and the replacement cost of assets is approximated with
    the book value of Total Assets (rather than their true replacement cost). Both
    simplifications are standard practice given that market values for debt and asset
    replacement costs are rarely observable, and mirror the same simplifications used
    elsewhere in this toolkit's Weighted Average Cost of Capital calculation.

    References:
    - Tobin, James. "A General Equilibrium Approach to Monetary Theory." Journal of Money,
    Credit and Banking, Vol. 1, No. 1, 1969, pp. 15-29.
    """
    if not isinstance(market_value_of_equity, int | float | pd.Series | pd.DataFrame):
        raise TypeError(
            "market_value_of_equity must be a float, pd.Series or pd.DataFrame, "
            f"not {type(market_value_of_equity)}."
        )

    return (market_value_of_equity + total_liabilities) / total_assets
