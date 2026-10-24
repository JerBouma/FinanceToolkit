"""Quality and Earnings Manipulation Score Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

# pylint: disable=too-many-arguments,too-many-locals

# The coefficients of Model 1 of Dechow, Ge, Larson and Sloan (2011), and the share of firms
# with a misstatement in their sample, which scales the probability into the F-Score.
DECHOW_INTERCEPT = -7.893
DECHOW_COEFFICIENTS = {
    "RSST Accruals": 0.790,
    "Change in Receivables": 2.518,
    "Change in Inventory": 1.191,
    "Soft Assets": 1.979,
    "Change in Cash Sales": 0.171,
    "Change in Return on Assets": -0.932,
    "Actual Issuance": 1.029,
}
DECHOW_UNCONDITIONAL_PROBABILITY = 0.0037

# Montier counts total asset growth above ten percent as a warning sign.
MONTIER_ASSET_GROWTH = 0.10

# Mohanram measures the variability of the return on assets and of sales growth over the past
# five years and requires at least three of them.
MOHANRAM_VARIANCE_PERIODS = 5
MOHANRAM_MINIMUM_VARIANCE_PERIODS = 3

# Every comparison below is made `lag` periods back. The models were estimated on yearly
# statements, so with quarterly flows summed over the trailing twelve months the lag is 4: a
# change is then year over year, between sums that do not overlap, instead of quarter over
# quarter, where three of the four quarters cancel out and the change is a quarter of its size.


def _previous(frame: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """The value `lag` periods back (frames are tickers by periods)."""
    return frame.shift(lag, axis=1)


def _change(frame: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """The change from the value `lag` periods back."""
    return frame - _previous(frame, lag)


def _average(frame: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """The average of the current value and the value `lag` periods back."""
    return (frame + _previous(frame, lag)) / 2


def _increased(frame: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """1 where the value rose from `lag` periods back, 0 where not, NaN where unknown."""
    previous = _previous(frame, lag)
    return (frame > previous).astype(float).where(frame.notna() & previous.notna())


def _decreased(frame: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """1 where the value fell from `lag` periods back, 0 where not, NaN where unknown."""
    previous = _previous(frame, lag)
    return (frame < previous).astype(float).where(frame.notna() & previous.notna())


def _variance(
    frame: pd.DataFrame, periods: int, lag: int, min_periods: int
) -> pd.DataFrame:
    """
    The variance of the latest `periods` values `lag` periods apart, NaN when fewer than
    `min_periods` of them are available. With a lag of 1 this is a rolling variance; with
    quarterly trailing-twelve-month values and a lag of 4 it takes one value per year.
    """
    observations = pd.concat(
        [_previous(frame, period * lag) for period in range(periods)],
        keys=range(periods),
    )
    grouped = observations.groupby(level=1, sort=False)
    return grouped.var().where(grouped.count() >= min_periods).reindex(frame.index)


def get_magic_formula(
    operating_income: pd.DataFrame,
    enterprise_value: pd.DataFrame,
    net_working_capital: pd.DataFrame,
    net_fixed_assets: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Greenblatt's Magic Formula: companies are ranked on their earnings yield (EBIT divided by
    enterprise value: how cheap they are) and their return on capital (EBIT divided by net
    working capital plus net fixed assets: how good they are), and the two ranks are added.
    The best combined rank buys good businesses at low prices.

    Args:
        operating_income (pd.DataFrame): EBIT per ticker (rows) and period (columns).
        enterprise_value (pd.DataFrame): The enterprise value.
        net_working_capital (pd.DataFrame): The net working capital (current assets
            without cash minus current liabilities without short-term debt).
        net_fixed_assets (pd.DataFrame): The net property, plant and equipment.

    Returns:
        dict[str, pd.DataFrame]: The Earnings Yield, Return on Capital, their ranks and the
        Magic Formula Rank (1 is best) per ticker and period.
    """
    earnings_yield = operating_income / enterprise_value
    invested = (net_working_capital + net_fixed_assets).where(
        lambda capital: capital > 0
    )
    return_on_capital = operating_income / invested

    earnings_yield_rank = earnings_yield.rank(axis=0, ascending=False)
    return_on_capital_rank = return_on_capital.rank(axis=0, ascending=False)
    combined = earnings_yield_rank + return_on_capital_rank

    return {
        "Earnings Yield": earnings_yield,
        "Return on Capital": return_on_capital,
        "Earnings Yield Rank": earnings_yield_rank,
        "Return on Capital Rank": return_on_capital_rank,
        "Magic Formula Rank": combined.rank(axis=0, method="min"),
    }


def get_mohanram_g_score(
    net_income: pd.DataFrame,
    operating_cash_flow: pd.DataFrame,
    total_assets: pd.DataFrame,
    revenue: pd.DataFrame,
    research_and_development: pd.DataFrame,
    capital_expenditure: pd.DataFrame,
    selling_and_marketing: pd.DataFrame,
    variance_periods: int = MOHANRAM_VARIANCE_PERIODS,
    lag: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Mohanram's G-Score for growth (low book-to-market) companies: eight signals, each 1
    when the company beats the median of its peers (the tickers given) and 0 otherwise.

    - G1: return on assets above the median.
    - G2: cash flow from operations to assets above the median.
    - G3: cash flow from operations above net income (earnings quality).
    - G4: variance of the return on assets below the median (stable earnings).
    - G5: variance of revenue growth below the median (stable growth).
    - G6: research and development to assets above the median.
    - G7: capital expenditure to assets above the median.
    - G8: advertising (selling and marketing) to assets above the median.

    As in the paper, net income, cash flow from operations, research and development,
    capital expenditure and advertising are scaled by the total assets at the beginning of
    the period, the variances use the past five yearly values of which at least three must
    be available, and a signal that cannot be calculated for a company (a missing item, or
    too short a history for the variances) scores 0 rather than leaving the G-Score empty.

    Args:
        net_income (pd.DataFrame): Net income per ticker (rows) and period (columns).
        operating_cash_flow (pd.DataFrame): Cash flow from operations.
        total_assets (pd.DataFrame): Total assets.
        revenue (pd.DataFrame): Revenue.
        research_and_development (pd.DataFrame): Research and development expenses.
        capital_expenditure (pd.DataFrame): Capital expenditure (the sign is ignored).
        selling_and_marketing (pd.DataFrame): Selling and marketing expenses.
        variance_periods (int, optional): The number of values, `lag` periods apart, the
            variances of G4 and G5 are calculated over. At least three of them must be
            available (all of them when fewer than three are asked for). Defaults to 5.
        lag (int, optional): The number of periods back that a comparison is made, one
            year: 1 for yearly statements and 4 for quarterly flows summed over the trailing
            twelve months. The beginning assets, the revenue growth and the values in the
            variances are this many periods apart. Defaults to 1.

    Returns:
        dict[str, pd.DataFrame]: The eight signals and the G-Score (0 to 8).

    Raises:
        ValueError: When variance_periods is below 2, as a variance needs two values.
    """
    if variance_periods <= 1:
        raise ValueError("The variances need at least two periods.")

    beginning_assets = _previous(total_assets, lag)
    return_on_assets = net_income / beginning_assets
    revenue_growth = revenue / _previous(revenue, lag) - 1
    minimum_periods = min(MOHANRAM_MINIMUM_VARIANCE_PERIODS, variance_periods)

    # A missing value compares as False, which gives the 0 the paper assigns to a signal that
    # cannot be calculated; the median is that of the tickers with a value.
    def above_median(frame: pd.DataFrame) -> pd.DataFrame:
        return (frame > frame.median(axis=0)).astype(float)

    def below_median(frame: pd.DataFrame) -> pd.DataFrame:
        return (frame < frame.median(axis=0)).astype(float)

    signals = {
        "G1 Return on Assets": above_median(return_on_assets),
        "G2 Cash Flow Return on Assets": above_median(
            operating_cash_flow / beginning_assets
        ),
        "G3 Cash Flow above Net Income": (operating_cash_flow > net_income).astype(
            float
        ),
        "G4 Earnings Variability": below_median(
            _variance(return_on_assets, variance_periods, lag, minimum_periods)
        ),
        "G5 Growth Variability": below_median(
            _variance(revenue_growth, variance_periods, lag, minimum_periods)
        ),
        "G6 Research and Development Intensity": above_median(
            research_and_development / beginning_assets
        ),
        "G7 Capital Expenditure Intensity": above_median(
            capital_expenditure.abs() / beginning_assets
        ),
        "G8 Advertising Intensity": above_median(
            selling_and_marketing / beginning_assets
        ),
    }
    # Only a period without a balance sheet has no score at all.
    reported = total_assets.notna()
    signals = {name: signal.where(reported) for name, signal in signals.items()}
    signals["G-Score"] = sum(signals.values())

    return signals


def get_dechow_f_score(
    total_assets: pd.DataFrame,
    current_assets: pd.DataFrame,
    current_liabilities: pd.DataFrame,
    total_liabilities: pd.DataFrame,
    cash_and_short_term_investments: pd.DataFrame,
    short_term_investments: pd.DataFrame,
    long_term_investments: pd.DataFrame,
    short_term_debt: pd.DataFrame,
    long_term_debt: pd.DataFrame,
    preferred_stock: pd.DataFrame,
    accounts_receivable: pd.DataFrame,
    inventory: pd.DataFrame,
    property_plant_and_equipment: pd.DataFrame,
    revenue: pd.DataFrame,
    net_income: pd.DataFrame,
    debt_issued: pd.DataFrame,
    stock_issued: pd.DataFrame,
    lag: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    The Dechow F-Score (Model 1 of Dechow, Ge, Larson and Sloan, 2011), the probability of
    an accounting misstatement relative to the average firm. It combines accrual quality
    (RSST accruals, changes in receivables and inventory, the share of soft assets),
    performance (the change in cash sales and in return on assets) and the market incentive
    to look good (issuing debt or equity):

        Predicted Value = -7.893 + 0.790 RSST Accruals + 2.518 ΔReceivables
                          + 1.191 ΔInventory + 1.979 Soft Assets + 0.171 ΔCash Sales
                          - 0.932 ΔROA + 1.029 Issuance
        Probability = e^PV / (1 + e^PV),  F-Score = Probability / 0.0037

    The accruals and changes are scaled by average total assets, soft assets are total
    assets minus net property, plant and equipment and cash and short-term investments
    (Compustat AT - PPENT - CHE) as a share of total assets, and Issuance is 1 when the
    company issued long-term debt or stock during the period (DLTIS > 0 or SSTK > 0).

    An F-Score above 1 means a higher than average risk of misstatement; above 2.45 the
    risk is high.

    Args:
        total_assets (pd.DataFrame): Total assets per ticker (rows) and period (columns).
        current_assets (pd.DataFrame): Total current assets.
        current_liabilities (pd.DataFrame): Total current liabilities.
        total_liabilities (pd.DataFrame): Total liabilities.
        cash_and_short_term_investments (pd.DataFrame): Cash and short-term investments.
        short_term_investments (pd.DataFrame): Short-term investments.
        long_term_investments (pd.DataFrame): Long-term investments.
        short_term_debt (pd.DataFrame): Short-term debt.
        long_term_debt (pd.DataFrame): Long-term debt.
        preferred_stock (pd.DataFrame): Preferred stock.
        accounts_receivable (pd.DataFrame): Accounts receivable.
        inventory (pd.DataFrame): Inventory.
        property_plant_and_equipment (pd.DataFrame): Net property, plant and equipment.
        revenue (pd.DataFrame): Revenue.
        net_income (pd.DataFrame): Net income.
        debt_issued (pd.DataFrame): Long-term debt issued, gross of repayments. When only
            the net issuance (issuance minus repayments) is available, a company that
            issued debt but repaid more counts as not issuing.
        stock_issued (pd.DataFrame): Stock issued, gross of repurchases.
        lag (int, optional): The number of periods back that a change and the average
            total assets are measured, one year: 1 for yearly statements and 4 for
            quarterly flows summed over the trailing twelve months. Defaults to 1.

    Returns:
        dict[str, pd.DataFrame]: The seven inputs, the Predicted Value, the Probability and
        the F-Score.
    """
    average_assets = _average(total_assets, lag)
    fill = lambda frame: frame.fillna(0)  # noqa: E731

    working_capital = (current_assets - cash_and_short_term_investments) - (
        current_liabilities - fill(short_term_debt)
    )
    non_current_operating = (
        total_assets - current_assets - fill(long_term_investments)
    ) - (total_liabilities - current_liabilities - fill(long_term_debt))
    financial = (fill(short_term_investments) + fill(long_term_investments)) - (
        fill(long_term_debt) + fill(short_term_debt) + fill(preferred_stock)
    )

    cash_sales = revenue - _change(accounts_receivable, lag)
    return_on_assets = net_income / average_assets

    inputs = {
        "RSST Accruals": (
            _change(working_capital, lag)
            + _change(non_current_operating, lag)
            + _change(financial, lag)
        )
        / average_assets,
        "Change in Receivables": _change(accounts_receivable, lag) / average_assets,
        "Change in Inventory": _change(fill(inventory), lag) / average_assets,
        # The paper's "cash and cash equivalent" is Compustat item 1, which is cash and
        # short-term investments (CHE), as in its working capital definition.
        "Soft Assets": (
            total_assets
            - property_plant_and_equipment
            - cash_and_short_term_investments
        )
        / total_assets,
        "Change in Cash Sales": cash_sales / _previous(cash_sales, lag) - 1,
        "Change in Return on Assets": _change(return_on_assets, lag),
        "Actual Issuance": ((fill(debt_issued) > 0) | (fill(stock_issued) > 0))
        .astype(float)
        .where(total_assets.notna()),
    }

    predicted = DECHOW_INTERCEPT + sum(
        coefficient * inputs[name] for name, coefficient in DECHOW_COEFFICIENTS.items()
    )
    probability = 1 / (1 + np.exp(-predicted))

    return {
        **inputs,
        "Predicted Value": predicted,
        "Probability of Misstatement": probability,
        "F-Score": probability / DECHOW_UNCONDITIONAL_PROBABILITY,
    }


def get_montier_c_score(
    net_income: pd.DataFrame,
    operating_cash_flow: pd.DataFrame,
    accounts_receivable: pd.DataFrame,
    revenue: pd.DataFrame,
    inventory: pd.DataFrame,
    cost_of_goods_sold: pd.DataFrame,
    other_current_assets: pd.DataFrame,
    depreciation_and_amortization: pd.DataFrame,
    property_plant_and_equipment: pd.DataFrame,
    total_assets: pd.DataFrame,
    lag: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Montier's C-Score ("cooking the books"): six warning signs of earnings manipulation,
    each 1 when present, summed into a score from 0 to 6.

    - A growing gap between net income and cash flow from operations.
    - Rising days sales outstanding (receivables growing faster than sales).
    - Rising days sales of inventory (inventory growing faster than cost of sales).
    - Rising other current assets relative to revenue.
    - Falling depreciation relative to gross property, plant and equipment.
    - Total asset growth above 10% (acquisitive growth).

    Args:
        net_income (pd.DataFrame): Net income per ticker (rows) and period (columns).
        operating_cash_flow (pd.DataFrame): Cash flow from operations.
        accounts_receivable (pd.DataFrame): Accounts receivable.
        revenue (pd.DataFrame): Revenue.
        inventory (pd.DataFrame): Inventory.
        cost_of_goods_sold (pd.DataFrame): Cost of goods sold.
        other_current_assets (pd.DataFrame): Other current assets.
        depreciation_and_amortization (pd.DataFrame): Depreciation and amortization.
        property_plant_and_equipment (pd.DataFrame): Gross property, plant and equipment,
            Montier's measure. Net property, plant and equipment can stand in when the gross
            amount is not reported, although the ratio to the net amount also rises as the
            assets age, which can hide a decline in the depreciation rate.
        total_assets (pd.DataFrame): Total assets.
        lag (int, optional): The number of periods back that every sign compares with,
            one year: 1 for yearly statements and 4 for quarterly flows summed over the
            trailing twelve months. Defaults to 1.

    Returns:
        dict[str, pd.DataFrame]: The six signals and the C-Score.
    """
    previous_assets = _previous(total_assets, lag)
    signals = {
        "Net Income to Cash Flow Gap": _increased(
            net_income - operating_cash_flow, lag
        ),
        "Days Sales Outstanding": _increased(accounts_receivable / revenue, lag),
        "Days Sales of Inventory": _increased(
            inventory.where(inventory > 0) / cost_of_goods_sold, lag
        ),
        "Other Current Assets to Revenue": _increased(
            other_current_assets / revenue, lag
        ),
        "Depreciation to Fixed Assets": _decreased(
            depreciation_and_amortization / property_plant_and_equipment, lag
        ),
        "Total Asset Growth": (
            total_assets / previous_assets - 1 > MONTIER_ASSET_GROWTH
        )
        .astype(float)
        .where(total_assets.notna() & previous_assets.notna()),
    }
    # Inventory-free companies cannot raise their inventory days.
    signals["Days Sales of Inventory"] = (
        signals["Days Sales of Inventory"]
        .where(inventory.fillna(0) > 0, 0.0)
        .where(revenue.notna())
    )
    signals["C-Score"] = sum(signals.values())

    return signals


def get_accruals_ratios(
    current_assets: pd.DataFrame,
    cash_and_cash_equivalents: pd.DataFrame,
    current_liabilities: pd.DataFrame,
    short_term_debt: pd.DataFrame,
    tax_payables: pd.DataFrame,
    depreciation_and_amortization: pd.DataFrame,
    net_income: pd.DataFrame,
    operating_cash_flow: pd.DataFrame,
    investing_cash_flow: pd.DataFrame,
    total_assets: pd.DataFrame,
    lag: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    The accruals ratios: the part of earnings that is not backed by cash, scaled by the
    average of the total assets at the start and the end of the period. Sloan (1996)
    showed that companies with high accruals have less persistent earnings and lower
    future returns.

    - Balance Sheet Accruals Ratio (Sloan, 1996): [(ΔCurrent Assets - ΔCash) - (ΔCurrent
      Liabilities - ΔShort-term Debt - ΔTaxes Payable) - Depreciation] / Average Total
      Assets.
    - Cash Flow Accruals Ratio: (Net Income - Cash Flow from Operations - Cash Flow from
      Investing) / Average Total Assets, earnings minus free cash flow, taken from the cash
      flow statement. Hribar and Collins (2002) showed that accruals derived from balance
      sheet changes contain errors when a company acquires, divests or discontinues
      operations; they measure accruals as earnings before extraordinary items minus cash
      flow from operations. Subtracting the investing cash flow as well also counts the
      investment in long-term operating assets as an accrual, as the broad accruals of
      Richardson, Sloan, Soliman and Tuna (2005) do.

    Args:
        current_assets (pd.DataFrame): Total current assets per ticker (rows) and period.
        cash_and_cash_equivalents (pd.DataFrame): Cash and cash equivalents.
        current_liabilities (pd.DataFrame): Total current liabilities.
        short_term_debt (pd.DataFrame): Short-term debt.
        tax_payables (pd.DataFrame): Taxes payable.
        depreciation_and_amortization (pd.DataFrame): Depreciation and amortization.
        net_income (pd.DataFrame): Net income.
        operating_cash_flow (pd.DataFrame): Cash flow from operations.
        investing_cash_flow (pd.DataFrame): Cash flow from investing.
        total_assets (pd.DataFrame): Total assets.
        lag (int, optional): The number of periods back that the balance sheet changes and
            the average total assets are measured, one year: 1 for yearly statements and 4
            for quarterly flows summed over the trailing twelve months. Defaults to 1.

    Returns:
        dict[str, pd.DataFrame]: The Balance Sheet Accruals Ratio and the Cash Flow Accruals
        Ratio.
    """
    average_assets = _average(total_assets, lag)

    def change(frame: pd.DataFrame) -> pd.DataFrame:
        return _change(frame.fillna(0), lag)

    balance_sheet_accruals = (
        (change(current_assets) - change(cash_and_cash_equivalents))
        - (change(current_liabilities) - change(short_term_debt) - change(tax_payables))
        - depreciation_and_amortization
    )
    cash_flow_accruals = net_income - operating_cash_flow - investing_cash_flow

    return {
        "Balance Sheet Accruals Ratio": balance_sheet_accruals / average_assets,
        "Cash Flow Accruals Ratio": cash_flow_accruals / average_assets,
    }
