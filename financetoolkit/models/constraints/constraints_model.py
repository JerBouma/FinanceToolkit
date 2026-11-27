"""Financial Constraints Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import drop_non_finite

# Hadlock and Pierce (2010) replace the size with that of $4.5 billion (in 2004 dollars) and
# the age with 37 years when a firm exceeds them: beyond these the constraints hardly change
# with size and age in their sample.
SIZE_CAP_IN_MILLIONS = 4_500
AGE_CAP_IN_YEARS = 37


def get_cash_flow_to_capital(
    income_before_extraordinary_items: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    lagged_capital: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The cash flow to capital of the Kaplan-Zingales index.

    It is the cash flow (income before extraordinary items plus depreciation and
    amortization) relative to the net property, plant and equipment at the start of the
    period.

    The formula is as follows:

        - Cash Flow to Capital = (Income Before Extraordinary Items + Depreciation and
          Amortization) / Lagged Net PP&E

    Args:
        income_before_extraordinary_items (pd.Series | pd.DataFrame): The income before
            extraordinary items (Compustat item 18).
        depreciation_and_amortization (pd.Series | pd.DataFrame): The depreciation and
            amortization (Compustat item 14).
        lagged_capital (pd.Series | pd.DataFrame): The net property, plant and equipment at
            the end of the previous period (Compustat item 8, lagged).

    Returns:
        pd.Series | pd.DataFrame: The cash flow to capital, NaN where the capital is zero.
    """
    return drop_non_finite(
        (income_before_extraordinary_items + depreciation_and_amortization)
        / lagged_capital
    )


def get_kaplan_zingales_q(
    total_assets: pd.Series | pd.DataFrame,
    market_value_of_equity: pd.Series | pd.DataFrame,
    common_equity: pd.Series | pd.DataFrame,
    deferred_taxes: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Q (market to book ratio) of the Kaplan-Zingales index.

    It is the book value of the assets with the book value of the common equity replaced by
    its market value, relative to the book value of the assets.

    The formula is as follows:

        - Q = (Total Assets + Market Value of Equity - Common Equity - Deferred Taxes) /
          Total Assets

    Args:
        total_assets (pd.Series | pd.DataFrame): The total assets (Compustat item 6).
        market_value_of_equity (pd.Series | pd.DataFrame): The market value of the common
            equity (CRSP December market equity in Lamont, Polk and Saa-Requejo).
        common_equity (pd.Series | pd.DataFrame): The book value of the common equity
            (Compustat item 60).
        deferred_taxes (pd.Series | pd.DataFrame): The deferred taxes of the balance sheet
            (Compustat item 74).

    Returns:
        pd.Series | pd.DataFrame: The Q, NaN where the total assets are zero.
    """
    return drop_non_finite(
        (total_assets + market_value_of_equity - common_equity - deferred_taxes)
        / total_assets
    )


def get_debt_to_total_capital(
    short_term_debt: pd.Series | pd.DataFrame,
    long_term_debt: pd.Series | pd.DataFrame,
    stockholders_equity: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The debt to total capital of the Kaplan-Zingales index: the debt relative to the debt
    plus the stockholders' equity, all at book value.

    The formula is as follows:

        - Debt to Total Capital = (Long-Term Debt + Short-Term Debt) /
          (Long-Term Debt + Short-Term Debt + Stockholders' Equity)

    Args:
        short_term_debt (pd.Series | pd.DataFrame): The debt in current liabilities
            (Compustat item 34).
        long_term_debt (pd.Series | pd.DataFrame): The long-term debt (Compustat item 9).
        stockholders_equity (pd.Series | pd.DataFrame): The total stockholders' equity,
            preferred stock included (Compustat item 216).

    Returns:
        pd.Series | pd.DataFrame: The debt to total capital, NaN where the total capital is
        zero. With negative equity it lies above 1, or below 0 when the equity is more
        negative than the debt is large.
    """
    debt = long_term_debt + short_term_debt

    return drop_non_finite(debt / (debt + stockholders_equity))


def get_dividends_to_capital(
    dividends: pd.Series | pd.DataFrame,
    lagged_capital: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The dividends to capital of the Kaplan-Zingales index: the common and preferred
    dividends relative to the net property, plant and equipment at the start of the period.

    The formula is as follows:

        - Dividends to Capital = (Common Dividends + Preferred Dividends) / Lagged Net PP&E

    Args:
        dividends (pd.Series | pd.DataFrame): The common and preferred dividends (Compustat
            items 21 and 19). The size is used, as cash flow statements report the
            dividends paid as a negative amount.
        lagged_capital (pd.Series | pd.DataFrame): The net property, plant and equipment at
            the end of the previous period (Compustat item 8, lagged).

    Returns:
        pd.Series | pd.DataFrame: The dividends to capital, NaN where the capital is zero.
    """
    return drop_non_finite(dividends.abs() / lagged_capital)


def get_cash_to_capital(
    cash_and_short_term_investments: pd.Series | pd.DataFrame,
    lagged_capital: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The cash to capital of the Kaplan-Zingales index: the cash and short-term investments
    relative to the net property, plant and equipment at the start of the period.

    The formula is as follows:

        - Cash to Capital = Cash and Short-Term Investments / Lagged Net PP&E

    Args:
        cash_and_short_term_investments (pd.Series | pd.DataFrame): The cash and short-term
            investments (Compustat item 1).
        lagged_capital (pd.Series | pd.DataFrame): The net property, plant and equipment at
            the end of the previous period (Compustat item 8, lagged).

    Returns:
        pd.Series | pd.DataFrame: The cash to capital, NaN where the capital is zero.
    """
    return drop_non_finite(cash_and_short_term_investments / lagged_capital)


def get_kaplan_zingales_index(
    cash_flow_to_capital: pd.Series | pd.DataFrame,
    q: pd.Series | pd.DataFrame,
    debt_to_total_capital: pd.Series | pd.DataFrame,
    dividends_to_capital: pd.Series | pd.DataFrame,
    cash_to_capital: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Kaplan-Zingales (KZ) index of financial constraints, with the coefficients of
    Lamont, Polk and Saa-Requejo (2001).

    Steven Kaplan's re-estimation of the ordered logit of Kaplan and Zingales (1997) on the
    variables available in Compustat (Table 9 of Lamont, Polk and Saa-Requejo: 719
    observations, the firm-years of the low-dividend manufacturing firms of Kaplan and
    Zingales). A higher index means a more constrained firm: high debt, low cash flow, low
    cash and low dividends, and a high Q (a constrained firm needs good investment
    opportunities to be constrained).

    The formula is as follows:

        - KZ = -1.001909 * Cash Flow to Capital + 0.2826389 * Q +
          3.139193 * Debt to Total Capital - 39.3678 * Dividends to Capital -
          1.314759 * Cash to Capital

    Args:
        cash_flow_to_capital (pd.Series | pd.DataFrame): The cash flow to lagged capital.
        q (pd.Series | pd.DataFrame): The market to book ratio of the assets.
        debt_to_total_capital (pd.Series | pd.DataFrame): The debt to total capital.
        dividends_to_capital (pd.Series | pd.DataFrame): The dividends to lagged capital.
        cash_to_capital (pd.Series | pd.DataFrame): The cash to lagged capital.

    Returns:
        pd.Series | pd.DataFrame: The Kaplan-Zingales index.

    References:
    - Lamont, Owen, Christopher Polk and Jesus Saa-Requejo. "Financial Constraints and Stock
      Returns." The Review of Financial Studies, Vol. 14, No. 2, 2001, pp. 529-554 (Appendix).
    - Kaplan, Steven N. and Luigi Zingales. "Do Investment-Cash Flow Sensitivities Provide
      Useful Measures of Financing Constraints?" The Quarterly Journal of Economics,
      Vol. 112, No. 1, 1997, pp. 169-215.
    """
    return (
        -1.001909 * cash_flow_to_capital
        + 0.2826389 * q
        + 3.139193 * debt_to_total_capital
        - 39.3678 * dividends_to_capital
        - 1.314759 * cash_to_capital
    )


def get_cash_flow_to_total_assets(
    income_before_extraordinary_items: pd.Series | pd.DataFrame,
    depreciation_and_amortization: pd.Series | pd.DataFrame,
    total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The cash flow to total assets (CF) of the Whited-Wu index.

    The formula is as follows:

        - CF = (Income Before Extraordinary Items + Depreciation and Amortization) /
          Total Assets

    Args:
        income_before_extraordinary_items (pd.Series | pd.DataFrame): The income before
            extraordinary items.
        depreciation_and_amortization (pd.Series | pd.DataFrame): The depreciation and
            amortization.
        total_assets (pd.Series | pd.DataFrame): The total assets.

    Returns:
        pd.Series | pd.DataFrame: The cash flow to total assets, NaN where the total assets
        are zero.
    """
    return drop_non_finite(
        (income_before_extraordinary_items + depreciation_and_amortization)
        / total_assets
    )


def get_dividend_payer_indicator(
    dividends: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The positive dividend indicator (DIVPOS) of the Whited-Wu index: 1 when the firm paid
    cash dividends (common or preferred) in the period, 0 when it paid none.

    Args:
        dividends (pd.Series | pd.DataFrame): The common and preferred dividends paid, as a
            positive or (as cash flow statements report them) negative amount.

    Returns:
        pd.Series | pd.DataFrame: 1.0 or 0.0, NaN where the dividends are not reported.
    """
    return (dividends.abs() > 0).astype(float).where(dividends.notna())


def get_long_term_debt_to_total_assets(
    long_term_debt: pd.Series | pd.DataFrame,
    total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The long-term debt to total assets (TLTD) of the Whited-Wu index.

    The formula is as follows:

        - TLTD = Long-Term Debt / Total Assets

    Args:
        long_term_debt (pd.Series | pd.DataFrame): The long-term debt.
        total_assets (pd.Series | pd.DataFrame): The total assets.

    Returns:
        pd.Series | pd.DataFrame: The long-term debt to total assets, NaN where the total
        assets are zero.
    """
    return drop_non_finite(long_term_debt / total_assets)


def get_log_of_total_assets_in_millions(
    total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The natural log of the total assets in millions (LNTA) of the Whited-Wu index, the unit
    of the Compustat data the index was estimated on.

    The formula is as follows:

        - LNTA = ln(Total Assets / 1,000,000)

    Args:
        total_assets (pd.Series | pd.DataFrame): The total assets in currency units.

    Returns:
        pd.Series | pd.DataFrame: The log of the total assets in millions, NaN where the
        total assets are not positive.
    """
    millions = total_assets / 1_000_000

    return drop_non_finite(millions.where(millions > 0).apply(np.log))


def get_industry_sales_growth(
    sales_growth: pd.DataFrame,
    industries: pd.Series,
) -> pd.DataFrame:
    """
    The industry sales growth (ISG) of the Whited-Wu index.

    Per period, the average sales growth of the firms in each firm's industry, the firm
    itself included. A firm without an industry is its own industry, so it gets its own
    sales growth.

    The formula is as follows:

        - ISG (industry j, period t) = mean of Sales Growth (i, t) over the firms i in j

    Args:
        sales_growth (pd.DataFrame): The sales growth per ticker (rows) and period
            (columns).
        industries (pd.Series): The industry per ticker; NaN or missing for a ticker
            without one.

    Returns:
        pd.DataFrame: The industry sales growth per ticker (rows) and period (columns).
        A period is averaged over the firms of the industry with a sales growth in it.
    """
    groups = industries.reindex(sales_growth.index)
    groups = groups.where(groups.notna(), pd.Series(groups.index, index=groups.index))

    return sales_growth.groupby(groups).transform("mean")


def get_whited_wu_index(
    cash_flow_to_total_assets: pd.Series | pd.DataFrame,
    dividend_payer_indicator: pd.Series | pd.DataFrame,
    long_term_debt_to_total_assets: pd.Series | pd.DataFrame,
    log_of_total_assets: pd.Series | pd.DataFrame,
    industry_sales_growth: pd.Series | pd.DataFrame,
    sales_growth: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Whited-Wu (WW) index of financial constraints.

    It is the shadow value of external funds estimated from the Euler equation of investment
    on quarterly Compustat data for 1975-2001 (Whited and Wu 2006, equation 13 from column 4
    of Table 1). A higher index means a more constrained firm: a small firm with low cash
    flow, no dividends and much long-term debt that grows slowly in a fast-growing industry.

    The formula is as follows:

        - WW = -0.091 * CF - 0.062 * DIVPOS + 0.021 * TLTD - 0.044 * LNTA +
          0.102 * ISG - 0.035 * SG

    Args:
        cash_flow_to_total_assets (pd.Series | pd.DataFrame): The cash flow to total assets
            (CF).
        dividend_payer_indicator (pd.Series | pd.DataFrame): 1 for a firm paying cash
            dividends, else 0 (DIVPOS).
        long_term_debt_to_total_assets (pd.Series | pd.DataFrame): The long-term debt to
            total assets (TLTD).
        log_of_total_assets (pd.Series | pd.DataFrame): The natural log of the total
            assets in millions (LNTA).
        industry_sales_growth (pd.Series | pd.DataFrame): The sales growth of the firm's
            industry (ISG).
        sales_growth (pd.Series | pd.DataFrame): The firm's sales growth (SG).

    Returns:
        pd.Series | pd.DataFrame: The Whited-Wu index.

    References:
    - Whited, Toni M. and Guojun Wu. "Financial Constraints Risk." The Review of Financial
      Studies, Vol. 19, No. 2, 2006, pp. 531-559.
    """
    return (
        -0.091 * cash_flow_to_total_assets
        - 0.062 * dividend_payer_indicator
        + 0.021 * long_term_debt_to_total_assets
        - 0.044 * log_of_total_assets
        + 0.102 * industry_sales_growth
        - 0.035 * sales_growth
    )


def get_size_age_size(
    total_assets: pd.Series | pd.DataFrame,
    price_level: pd.Series | pd.DataFrame | float = 1.0,
) -> pd.Series | pd.DataFrame:
    """
    The Size of the size-age (SA) index of Hadlock and Pierce (2010): the natural log of
    the book assets in millions of 2004 dollars, capped at the log of $4.5 billion.

    The formula is as follows:

        - Size = ln(min(Total Assets / 1,000,000 / Price Level, 4,500))

    Args:
        total_assets (pd.Series | pd.DataFrame): The total assets in currency units.
        price_level (pd.Series | pd.DataFrame | float, optional): The price level of the
            period relative to that of 2004 (the consumer price index of the period divided
            by its 2004 average), which turns the assets into 2004 dollars. Defaults to 1.0,
            which takes the assets as they are.

    Returns:
        pd.Series | pd.DataFrame: The Size, NaN where the total assets are not positive.
    """
    millions = total_assets / 1_000_000 / price_level
    millions = millions.where(millions > 0).clip(upper=SIZE_CAP_IN_MILLIONS)

    return drop_non_finite(millions.apply(np.log))


def get_size_age_age(
    years_listed: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Age of the size-age (SA) index of Hadlock and Pierce (2010): the number of years
    the firm is listed, capped at 37.

    The formula is as follows:

        - Age = min(Years Listed, 37)

    Args:
        years_listed (pd.Series | pd.DataFrame): The number of years the firm has been
            listed with a stock price.

    Returns:
        pd.Series | pd.DataFrame: The Age, NaN where the years listed are negative (a
        period before the listing) or not known.
    """
    return years_listed.where(years_listed >= 0).clip(upper=AGE_CAP_IN_YEARS)


def get_size_age_index(
    size: pd.Series | pd.DataFrame,
    age: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The size-age (SA) index of financial constraints of Hadlock and Pierce (2010).

    It is the ordered logit of the financial constraints they coded from the 10-K filings of
    a random sample of firms (1995-2004) on size and age only. A higher (less negative)
    index means a more constrained firm: constraints fall as firms grow and age, and less so
    as they get large (the Size squared term).

    The formula is as follows:

        - SA = -0.737 * Size + 0.043 * Size^2 - 0.040 * Age

    Args:
        size (pd.Series | pd.DataFrame): The log of the book assets in millions of 2004
            dollars, capped at the log of $4.5 billion.
        age (pd.Series | pd.DataFrame): The years listed, capped at 37.

    Returns:
        pd.Series | pd.DataFrame: The size-age index.

    Notes:
    - Within the caps the index falls with size (its slope -0.737 + 0.086 * Size is negative
      up to a Size of 8.57, beyond the cap of ln(4,500) = 8.41), so its lowest value is -4.64
      (a firm at both caps). A new firm with $1 million of assets has 0, and a smaller one
      above 0 (its Size is negative).

    References:
    - Hadlock, Charles J. and Joshua R. Pierce. "New Evidence on Measuring Financial
      Constraints: Moving Beyond the KZ Index." The Review of Financial Studies, Vol. 23,
      No. 5, 2010, pp. 1909-1940.
    """
    return -0.737 * size + 0.043 * size**2 - 0.040 * age
