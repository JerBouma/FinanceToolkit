"""Enterprise Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.ratios import valuation_model
from financetoolkit.utilities.dataframe_model import concat_frames


def get_enterprise_value_breakdown(
    share_price: pd.Series | pd.DataFrame,
    shares_outstanding: pd.Series | pd.DataFrame,
    total_debt: pd.Series | pd.DataFrame,
    minority_interest: pd.Series | pd.DataFrame,
    preferred_equity: pd.Series | pd.DataFrame,
    cash_and_cash_equivalents: pd.Series | pd.DataFrame,
    short_term_investments: pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    The Enterprise Value breakdown corresponds to the following components:
        - Share Price: given for each quarter or year.
        - Market cap: The total value of a company's outstanding *common* shares, i.e. the share
        price multiplied by the shares outstanding. Preferred shares are deliberately excluded here
        and enter the Enterprise Value as their own component below, so that they are counted once
        rather than twice.
        - Debt: The sum of long-term and short-term debt
        - Preferred equity: The value of preferred shares
        - Minority interest: The equity value of a subsidiary with less than 50% ownership.
        - Cash and cash equivalents: The cash and the investments that are readily convertible
        to cash within three months, such as money market funds and Treasury bills.
        - Short term investments: The marketable securities that are not cash equivalents, such
        as certificates of deposit and commercial paper with a longer maturity.

    The formula is as follows:

        - Enterprise Value = Market Capitalization + Total Debt + Minority Interest
        + Preferred Equity - (Cash and Cash Equivalents + Short Term Investments)

    Cash and marketable securities are deducted together (Damodaran), the same Enterprise
    Value as the Ratios module calculates. A missing Short Term Investments value counts as 0.

    Args:
        share_price (pd.Series | pd.DataFrame): The share price of the company.
        shares_outstanding (pd.Series | pd.DataFrame): The total shares outstanding of the company.
        total_debt (pd.Series | pd.DataFrame): The total debt of the company.
        minority_interest (pd.Series | pd.DataFrame): The minority interest of the company.
        preferred_equity (pd.Series | pd.DataFrame): The preferred equity of the company.
        cash_and_cash_equivalents (pd.Series | pd.DataFrame): The cash and cash equivalents of the
            company.
        short_term_investments (pd.Series | pd.DataFrame | None, optional): The short term
            investments of the company. Defaults to None, which deducts only the cash and cash
            equivalents and leaves the Short Term Investments row out of the breakdown.

    Returns:
        pd.DataFrame: the Enterprise Value breakdown.

    References:
    - Damodaran, Aswath. "Investment Valuation: Tools and Techniques for Determining the
    Value of Any Asset." 3rd ed., Wiley, 2012, chapter 16.
    """
    # The market value of the common shares.
    market_cap = valuation_model.get_market_cap(
        share_price=share_price, total_shares_outstanding=shares_outstanding
    )

    # The two separate rows are added so that a missing short term investment value never
    # leaves the cash out of the deduction.
    cash_and_short_term_investments = (
        cash_and_cash_equivalents.add(short_term_investments, fill_value=0)
        if short_term_investments is not None
        else cash_and_cash_equivalents
    )

    enterprise_value = valuation_model.get_enterprise_value(
        market_cap=market_cap,
        total_debt=total_debt,
        minority_interest=minority_interest,
        preferred_equity=preferred_equity,
        cash_and_cash_equivalents=cash_and_short_term_investments,
    )

    components = {
        "Share Price": share_price,
        "Market Capitalization": market_cap,
        "Total Debt": total_debt,
        "Minority Interest": minority_interest,
        "Preferred Equity": preferred_equity,
        "Cash and Cash Equivalents": cash_and_cash_equivalents,
    }

    if short_term_investments is not None:
        components["Short Term Investments"] = short_term_investments

    components["Enterprise Value"] = enterprise_value

    if isinstance(enterprise_value, pd.DataFrame):
        return (
            concat_frames(components)
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
