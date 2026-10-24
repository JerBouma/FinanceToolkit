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


def _previous(frame: pd.DataFrame) -> pd.DataFrame:
    """The value of the previous period (frames are tickers by periods)."""
    return frame.shift(1, axis=1)


def _average(frame: pd.DataFrame) -> pd.DataFrame:
    """The average of the current and previous period."""
    return (frame + _previous(frame)) / 2


def _increased(frame: pd.DataFrame) -> pd.DataFrame:
    """1 where the value rose from the previous period, 0 where not, NaN where unknown."""
    previous = _previous(frame)
    return (frame > previous).astype(float).where(frame.notna() & previous.notna())


def _decreased(frame: pd.DataFrame) -> pd.DataFrame:
    """1 where the value fell from the previous period, 0 where not, NaN where unknown."""
    previous = _previous(frame)
    return (frame < previous).astype(float).where(frame.notna() & previous.notna())


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
    variance_periods: int = 4,
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

    Args:
        net_income (pd.DataFrame): Net income per ticker (rows) and period (columns).
        operating_cash_flow (pd.DataFrame): Cash flow from operations.
        total_assets (pd.DataFrame): Total assets.
        revenue (pd.DataFrame): Revenue.
        research_and_development (pd.DataFrame): Research and development expenses.
        capital_expenditure (pd.DataFrame): Capital expenditure (as a positive amount).
        selling_and_marketing (pd.DataFrame): Selling and marketing expenses.
        variance_periods (int, optional): The number of periods the variances are
            calculated over. Defaults to 4.

    Returns:
        dict[str, pd.DataFrame]: The eight signals and the G-Score (0 to 8).
    """
    assets = _average(total_assets)
    return_on_assets = net_income / assets
    cash_flow_return = operating_cash_flow / assets
    revenue_growth = revenue / _previous(revenue) - 1

    def above_median(frame: pd.DataFrame) -> pd.DataFrame:
        median = frame.median(axis=0)
        return (frame > median).astype(float).where(frame.notna())

    def below_median(frame: pd.DataFrame) -> pd.DataFrame:
        median = frame.median(axis=0)
        return (frame < median).astype(float).where(frame.notna())

    signals = {
        "G1 Return on Assets": above_median(return_on_assets),
        "G2 Cash Flow Return on Assets": above_median(cash_flow_return),
        "G3 Cash Flow above Net Income": (operating_cash_flow > net_income)
        .astype(float)
        .where(operating_cash_flow.notna() & net_income.notna()),
        "G4 Earnings Variability": below_median(
            return_on_assets.T.rolling(variance_periods).var().T
        ),
        "G5 Growth Variability": below_median(
            revenue_growth.T.rolling(variance_periods).var().T
        ),
        "G6 Research and Development Intensity": above_median(
            research_and_development / assets
        ),
        "G7 Capital Expenditure Intensity": above_median(
            capital_expenditure.abs() / assets
        ),
        "G8 Advertising Intensity": above_median(selling_and_marketing / assets),
    }
    signals["G-Score"] = sum(signals.values())

    return signals


def get_dechow_f_score(
    total_assets: pd.DataFrame,
    current_assets: pd.DataFrame,
    current_liabilities: pd.DataFrame,
    total_liabilities: pd.DataFrame,
    cash_and_short_term_investments: pd.DataFrame,
    cash_and_cash_equivalents: pd.DataFrame,
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

    An F-Score above 1 means a higher than average risk of misstatement; above 2.45 the
    risk is high.

    Args:
        total_assets (pd.DataFrame): Total assets per ticker (rows) and period (columns).
        current_assets (pd.DataFrame): Total current assets.
        current_liabilities (pd.DataFrame): Total current liabilities.
        total_liabilities (pd.DataFrame): Total liabilities.
        cash_and_short_term_investments (pd.DataFrame): Cash and short-term investments.
        cash_and_cash_equivalents (pd.DataFrame): Cash and cash equivalents.
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
        debt_issued (pd.DataFrame): Long-term debt issued.
        stock_issued (pd.DataFrame): Common stock issued.

    Returns:
        dict[str, pd.DataFrame]: The seven inputs, the Predicted Value, the Probability and
        the F-Score.
    """
    average_assets = _average(total_assets)
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

    def change(frame: pd.DataFrame) -> pd.DataFrame:
        return frame - _previous(frame)

    cash_sales = revenue - change(accounts_receivable)
    return_on_assets = net_income / average_assets

    inputs = {
        "RSST Accruals": (
            change(working_capital) + change(non_current_operating) + change(financial)
        )
        / average_assets,
        "Change in Receivables": change(accounts_receivable) / average_assets,
        "Change in Inventory": change(fill(inventory)) / average_assets,
        "Soft Assets": (
            total_assets - property_plant_and_equipment - cash_and_cash_equivalents
        )
        / total_assets,
        "Change in Cash Sales": cash_sales / _previous(cash_sales) - 1,
        "Change in Return on Assets": change(return_on_assets),
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
) -> dict[str, pd.DataFrame]:
    """
    Montier's C-Score ("cooking the books"): six warning signs of earnings manipulation,
    each 1 when present, summed into a score from 0 to 6.

    - A growing gap between net income and cash flow from operations.
    - Rising days sales outstanding (receivables growing faster than sales).
    - Rising days sales of inventory (inventory growing faster than cost of sales).
    - Rising other current assets relative to revenue.
    - Falling depreciation relative to property, plant and equipment.
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
        property_plant_and_equipment (pd.DataFrame): Net property, plant and equipment.
        total_assets (pd.DataFrame): Total assets.

    Returns:
        dict[str, pd.DataFrame]: The six signals and the C-Score.
    """
    previous_assets = _previous(total_assets)
    signals = {
        "Net Income to Cash Flow Gap": _increased(net_income - operating_cash_flow),
        "Days Sales Outstanding": _increased(accounts_receivable / revenue),
        "Days Sales of Inventory": _increased(
            inventory.where(inventory > 0) / cost_of_goods_sold
        ),
        "Other Current Assets to Revenue": _increased(other_current_assets / revenue),
        "Depreciation to Fixed Assets": _decreased(
            depreciation_and_amortization / property_plant_and_equipment
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
) -> dict[str, pd.DataFrame]:
    """
    The accruals ratios: the part of earnings that is not backed by cash, scaled by average
    total assets. Sloan (1996) showed that companies with high accruals have less
    persistent earnings and lower future returns.

    - Balance Sheet Accruals Ratio (Sloan, 1996): [(ΔCurrent Assets - ΔCash) - (ΔCurrent
      Liabilities - ΔShort-term Debt - ΔTaxes Payable) - Depreciation] / Average Assets.
    - Cash Flow Accruals Ratio (Hribar and Collins, 2002): (Net Income - Cash Flow from
      Operations - Cash Flow from Investing) / Average Assets, which avoids the distortions
      of acquisitions and divestments in balance sheet changes.

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

    Returns:
        dict[str, pd.DataFrame]: The Balance Sheet Accruals Ratio and the Cash Flow Accruals
        Ratio.
    """
    average_assets = _average(total_assets)

    def change(frame: pd.DataFrame) -> pd.DataFrame:
        filled = frame.fillna(0)
        return filled - _previous(filled)

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
