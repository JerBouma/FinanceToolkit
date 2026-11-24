"""Quality and Earnings Manipulation Score Model"""

__docformat__ = "google"

import operator
from functools import reduce

import numpy as np
import pandas as pd

from financetoolkit.models import helpers
from financetoolkit.utilities.error_model import check_option

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

# The modified Jones model regresses the total accruals on these three terms, without an
# intercept as in Jones (1991) unless one is asked for (Kothari, Leone and Wasley, 2005), and
# Dechow, Sloan and Sweeney (1995) estimated it on at least ten years of each company's history.
JONES_REGRESSORS = [
    "Inverse of Beginning Assets",
    "Change in Revenue",
    "Property, Plant and Equipment",
]
JONES_MINIMUM_OBSERVATIONS = 10

# Z-scores of ranks need a spread of ranks, which three tickers are the fewest to give in a useful way.
QUALITY_MINIMUM_TICKERS = 3

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
    Greenblatt's Magic Formula.

    Companies are ranked on their earnings yield (EBIT divided by enterprise value: how
    cheap they are) and their return on capital (EBIT divided by net working capital plus
    net fixed assets: how good they are), and the two ranks are added. The best combined
    rank buys good businesses at low prices.

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
    an accounting misstatement relative to the average firm.

    It combines accrual quality (RSST accruals, changes in receivables and inventory, the
    share of soft assets), performance (the change in cash sales and in return on assets)
    and the market incentive to look good (issuing debt or equity):

        Predicted Value = -7.893 + 0.790 RSST Accruals + 2.518 ΔReceivables +
                          1.191 ΔInventory + 1.979 Soft Assets + 0.171 ΔCash Sales -
                          0.932 ΔROA + 1.029 Issuance
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
    average of the total assets at the start and the end of the period.

    Sloan (1996) showed that companies with high accruals have less persistent earnings and
    lower future returns.

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


def get_modified_jones_accruals(
    net_income: pd.DataFrame,
    operating_cash_flow: pd.DataFrame,
    total_assets: pd.DataFrame,
    revenue: pd.DataFrame,
    accounts_receivable: pd.DataFrame,
    property_plant_and_equipment: pd.DataFrame,
    lag: int = 1,
    method: str = "time_series",
    min_observations: int = JONES_MINIMUM_OBSERVATIONS,
    intercept: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    The discretionary accruals of the modified Jones model (Dechow, Sloan and Sweeney,
    1995).

    It is the part of the total accruals that the change in revenues and the fixed assets do
    not explain.

    Every term is scaled by the total assets at the start of the period:

    - Total Accruals = (Net Income - Cash Flow from Operations) / Total Assets(t-1)
    - Estimation (Jones, 1991): Total Accruals = a1 × 1 / A(t-1) + a2 × ΔREV / A(t-1) +
      a3 × PPE / A(t-1) + e, without an intercept
    - Non-Discretionary Accruals = a1 × 1 / A(t-1) + a2 × (ΔREV - ΔREC) / A(t-1) +
      a3 × PPE / A(t-1)
    - Discretionary Accruals = Total Accruals - Non-Discretionary Accruals

    With an intercept a constant a0 is added to the estimation and to the non-discretionary
    accruals (Kothari, Leone and Wasley, 2005).

    In the time series method the coefficients of every period are estimated on the
    ticker's own earlier periods (`lag` periods apart, so that quarterly trailing twelve
    months do not overlap), with the change in revenues as in Jones (1991), and the change in
    receivables is subtracted only for the period itself, as in Dechow, Sloan and Sweeney
    (1995). In the cross-section method they are estimated per period across the tickers,
    with the change in revenues minus the change in receivables in the estimation as well,
    as in the cross-sectional modified Jones model of the later literature.

    Referenced against:
    - statsmodels, OLS (time series): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
    - statsmodels, OLS (with an intercept): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html

    Args:
        net_income (pd.DataFrame): Net income per ticker (rows) and period (columns).
        operating_cash_flow (pd.DataFrame): Cash flow from operations.
        total_assets (pd.DataFrame): Total assets.
        revenue (pd.DataFrame): Revenue.
        accounts_receivable (pd.DataFrame): Accounts receivable.
        property_plant_and_equipment (pd.DataFrame): Property, plant and equipment (gross in
            Jones, 1991).
        lag (int, optional): The number of periods back that the changes and the beginning
            assets are taken, one year: 1 for yearly statements and 4 for quarterly flows
            summed over the trailing twelve months. Defaults to 1.
        method (str, optional): Options are: "time_series" (each ticker's earlier periods),
            "cross_section" (the tickers of each period). Defaults to "time_series".
        min_observations (int, optional): The fewest observations a regression is estimated
            on, earlier periods or tickers. Defaults to 10.
        intercept (bool, optional): Whether to add a constant to the regression (Kothari,
            Leone and Wasley, 2005), which then needs one observation more. Defaults to False,
            the formulation of Jones (1991) and Dechow, Sloan and Sweeney (1995).

    Raises:
        ValueError: If the method is not "time_series" or "cross_section", or fewer
            observations than the coefficients (three, or four with the intercept) are asked for.

    Returns:
        dict[str, pd.DataFrame]: The Total Accruals, Non-Discretionary Accruals and
        Discretionary Accruals, scaled by the beginning total assets; the last two NaN where
        fewer than `min_observations` observations are available.
    """
    check_option("method", method, ("cross_section", "time_series"))
    coefficient_count = len(JONES_REGRESSORS) + intercept
    if min_observations < coefficient_count:
        raise ValueError(
            f"The model estimates {coefficient_count} coefficients and needs at least "
            f"as many observations, not {min_observations}."
        )

    # A company with no or negative assets at the start of the period has no scale.
    beginning_assets = _previous(total_assets, lag)
    beginning_assets = beginning_assets.where(beginning_assets > 0)

    total_accruals = (net_income - operating_cash_flow) / beginning_assets
    inverse_assets = 1 / beginning_assets
    revenue_change = _change(revenue, lag) / beginning_assets
    cash_revenue_change = (
        revenue_change - _change(accounts_receivable, lag) / beginning_assets
    )
    fixed_assets = property_plant_and_equipment / beginning_assets

    # Observations (tickers by periods) by regressor, in the order of JONES_REGRESSORS and
    # followed by a column of ones for the intercept, which enters the estimation and the
    # non-discretionary accruals alike.
    constant = [np.ones(inverse_assets.shape)] if intercept else []
    estimation = np.stack(
        [
            inverse_assets.to_numpy(float),
            (
                revenue_change if method == "time_series" else cash_revenue_change
            ).to_numpy(float),
            fixed_assets.to_numpy(float),
            *constant,
        ],
        axis=-1,
    )
    event = np.stack(
        [
            inverse_assets.to_numpy(float),
            cash_revenue_change.to_numpy(float),
            fixed_assets.to_numpy(float),
            *constant,
        ],
        axis=-1,
    )
    dependent = total_accruals.to_numpy(float)

    non_discretionary = np.full(dependent.shape, np.nan)
    tickers, periods = dependent.shape
    for period in range(periods):
        if method == "cross_section":
            coefficients = helpers.least_squares_coefficients(
                dependent[:, period], estimation[:, period], min_observations
            )
            non_discretionary[:, period] = event[:, period] @ coefficients
            continue
        # Only the periods before, one year apart, so the estimate uses what was known
        # at the start of the period and quarterly trailing sums do not overlap.
        earlier = np.arange(period - lag, -1, -lag)
        for ticker in range(tickers):
            coefficients = helpers.least_squares_coefficients(
                dependent[ticker, earlier],
                estimation[ticker, earlier],
                min_observations,
            )
            non_discretionary[ticker, period] = event[ticker, period] @ coefficients

    non_discretionary_accruals = pd.DataFrame(
        non_discretionary, index=total_accruals.index, columns=total_accruals.columns
    )

    return {
        "Total Accruals": total_accruals,
        "Non-Discretionary Accruals": non_discretionary_accruals,
        "Discretionary Accruals": total_accruals - non_discretionary_accruals,
    }


def get_rank_z_score(frame: pd.DataFrame) -> pd.DataFrame:
    """
    The cross-sectional z-score of the ranks of a ticker by period frame.

    In every period the tickers are ranked (the average rank for ties) and the ranks are
    standardized to a mean of 0 and a standard deviation of 1 (ddof=1) across the tickers
    with a value. Ranks make the score robust to outliers, as in Asness, Frazzini and
    Pedersen (2019).

    Args:
        frame (pd.DataFrame): the values with the tickers as the index and the periods as the columns.

    Returns:
        pd.DataFrame: the rank z-scores in the same shape as the input, NaN where a ticker has no value.
    """
    ranks = frame.rank(axis=0)
    return (ranks - ranks.mean(axis=0)) / ranks.std(axis=0)


def get_quality_minus_junk(
    components: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """
    The Quality Minus Junk score of Asness, Frazzini and Pedersen (2019) from its measures,
    each a ticker by period frame in which a higher value means a higher quality.

    Every measure is turned into the cross-sectional z-score of its ranks, each component is
    the z-score of the average of its measures, and the Quality Score is the z-score of the
    sum of the components. A measure that is missing for a ticker is left out of that
    ticker's average.

    Args:
        components (dict[str, dict[str, pd.DataFrame]]): The measures per component, e.g.
            {"Profitability": {"Gross Profits to Assets": ..., ...}, "Safety": {...}}.

    Returns:
        dict[str, pd.DataFrame]: The z-score of every component and the Quality Score, each a
        ticker by period frame.
    """
    scores: dict[str, pd.DataFrame] = {}
    for component, measures in components.items():
        z_scores = [get_rank_z_score(measure) for measure in measures.values()]
        average = pd.concat(z_scores).groupby(level=0, sort=False).mean()
        scores[component] = get_rank_z_score(average.reindex(z_scores[0].index))

    # A ticker without every component has no Quality Score, rather than one from fewer components.  # noqa: E501
    total = reduce(operator.add, scores.values())
    scores["Quality Score"] = get_rank_z_score(total)
    return scores


def get_accrual_quality(
    working_capital_accruals: pd.DataFrame,
    operating_cash_flow: pd.DataFrame,
    window: int = 8,
    lag: int = 1,
    change_in_revenue: pd.DataFrame | None = None,
    property_plant_and_equipment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    The accrual quality of Dechow and Dichev (2002).

    It is the standard deviation of the residuals of a regression of each company's working
    capital accruals on its past, present and future operating cash flows. Accruals that do
    not map into cash flows (large residuals) are of low quality, so a higher value means
    lower quality. With the change in revenue and the property, plant and equipment it is
    the model of McNichols (2002).

    The regression, estimated per company over the `window` years before the period:

        - ΔWC (t) = b0 + b1 CFO (t - 1) + b2 CFO (t) + b3 CFO (t + 1) [+ b4 ΔRev (t) + b5 PPE (t)] + e (t)
        - Accrual Quality = σ(e), the standard deviation of the residuals (ddof=1)

    Every variable is scaled by the average total assets of its year before it is passed in.

    Args:
        working_capital_accruals (pd.DataFrame): The increase in working capital (ΔWC) per ticker
            (rows) and period (columns), scaled by the average total assets.
        operating_cash_flow (pd.DataFrame): The cash flow from operations, scaled likewise.
        window (int, optional): The number of years in every regression; Dechow and Dichev
            required at least 8. Defaults to 8.
        lag (int, optional): The number of columns in a year: 1 for yearly statements and 4 for
            quarterly flows summed over the trailing twelve months. Defaults to 1.
        change_in_revenue (pd.DataFrame | None, optional): The change in revenue, scaled, for the
            McNichols model. Defaults to None.
        property_plant_and_equipment (pd.DataFrame | None, optional): The property, plant and
            equipment, scaled, for the McNichols model. Defaults to None.

    Returns:
        pd.DataFrame: The accrual quality per ticker and period: the regression of period T uses
        the accruals of the `window` years up to T - 1 (whose future cash flow, that of T at the
        latest, is known at T), so it never looks ahead. NaN with fewer complete years.

    References:
    - Dechow, P. M. and Dichev, I. D. (2002). "The Quality of Accruals and Earnings: The Role
      of Accrual Estimation Errors." The Accounting Review, 77(Supplement), 35-59.
    - McNichols, M. F. (2002). "Discussion of The Quality of Accruals and Earnings."
      The Accounting Review, 77(Supplement), 61-69.
    """
    regressors = [
        operating_cash_flow.shift(lag, axis=1),
        operating_cash_flow,
        operating_cash_flow.shift(-lag, axis=1),
    ]
    if change_in_revenue is not None and property_plant_and_equipment is not None:
        regressors += [change_in_revenue, property_plant_and_equipment]
    coefficient_count = len(regressors) + 1
    if window <= coefficient_count:
        raise ValueError(
            f"The regression estimates {coefficient_count} coefficients and needs a window of "
            f"more years, not {window}."
        )

    design = np.stack(
        [np.ones(working_capital_accruals.shape)]
        + [frame.to_numpy(float) for frame in regressors],
        axis=-1,
    )
    dependent = working_capital_accruals.to_numpy(float)
    quality = np.full(dependent.shape, np.nan)
    tickers, periods = dependent.shape

    for period in range(periods):
        # The years T - 1, T - 2, ... of the window: their future cash flows are known at T.
        years = np.arange(period - lag, period - lag * (window + 1), -lag)
        if years.min() < 0:
            continue
        for ticker in range(tickers):
            y, x = dependent[ticker, years], design[ticker, years]
            if not (np.isfinite(y).all() and np.isfinite(x).all()):
                continue
            coefficients = helpers.least_squares_coefficients(y, x, window)
            if np.isnan(coefficients).any():
                continue
            quality[ticker, period] = np.std(y - x @ coefficients, ddof=1)

    return pd.DataFrame(
        quality,
        index=working_capital_accruals.index,
        columns=working_capital_accruals.columns,
    )


def get_earnings_attributes(
    earnings_per_share: pd.DataFrame,
    earnings_to_assets: pd.DataFrame,
    cash_flow_to_assets: pd.DataFrame,
    window: int = 10,
    lag: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    The accounting-based earnings attributes of Francis, LaFond, Olsson and Schipper (2004),
    each over the `window` years up to the period:

        - Persistence: φ1 of EPS (t) = φ0 + φ1 EPS (t - 1) + ν (t), estimated per company
        - Predictability: sqrt(σ²(ν)), the square root of the error variance of that
          regression, sqrt(Σ ν² / (N - 2))
        - Smoothness: σ(Earnings / Total Assets (t - 1)) / σ(Cash Flow from Operations / Total
          Assets (t - 1)) (ddof=1)

    Higher persistence means more persistent earnings; higher predictability and smoothness
    values mean less predictable and less smooth earnings (as in the paper, larger values are
    less desirable).

    Args:
        earnings_per_share (pd.DataFrame): The earnings before extraordinary items per share, per
            ticker (rows) and period (columns).
        earnings_to_assets (pd.DataFrame): The earnings over the total assets at the start of the
            year.
        cash_flow_to_assets (pd.DataFrame): The cash flow from operations over the total assets
            at the start of the year.
        window (int, optional): The number of years in every window (Francis et al. use 10).
            Defaults to 10.
        lag (int, optional): The number of columns in a year: 1 for yearly statements and 4 for
            quarterly flows summed over the trailing twelve months. Defaults to 1.

    Returns:
        dict[str, pd.DataFrame]: The Persistence, Predictability and Smoothness per ticker and
        period, NaN unless every year of the window (and the year before it for the
        persistence regression) is available.

    References:
    - Francis, J., LaFond, R., Olsson, P. M. and Schipper, K. (2004). "Costs of Equity and
      Earnings Attributes." The Accounting Review, 79(4), 967-1010.
    """
    if window < 3:  # noqa: PLR2004
        raise ValueError("The window needs at least 3 years for a regression.")

    earnings = earnings_per_share.to_numpy(float)
    previous_earnings = earnings_per_share.shift(lag, axis=1).to_numpy(float)
    scaled_earnings = earnings_to_assets.to_numpy(float)
    scaled_cash_flow = cash_flow_to_assets.to_numpy(float)
    tickers, periods = earnings.shape
    persistence = np.full(earnings.shape, np.nan)
    predictability = np.full(earnings.shape, np.nan)
    smoothness = np.full(earnings.shape, np.nan)

    for period in range(periods):
        years = np.arange(period, period - lag * window, -lag)
        if years.min() < 0:
            continue
        for ticker in range(tickers):
            y, x = earnings[ticker, years], previous_earnings[ticker, years]
            if np.isfinite(y).all() and np.isfinite(x).all():
                design = np.column_stack([np.ones(window), x])
                coefficients = helpers.least_squares_coefficients(y, design, window)
                if np.isfinite(coefficients).all():
                    residuals = y - design @ coefficients
                    persistence[ticker, period] = coefficients[1]
                    predictability[ticker, period] = np.sqrt(
                        (residuals**2).sum() / (window - 2)
                    )
            income, cash = (
                scaled_earnings[ticker, years],
                scaled_cash_flow[ticker, years],
            )
            if np.isfinite(income).all() and np.isfinite(cash).all():
                cash_deviation = np.std(cash, ddof=1)
                if cash_deviation > 0:
                    smoothness[ticker, period] = np.std(income, ddof=1) / cash_deviation

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            values, index=earnings_per_share.index, columns=earnings_per_share.columns
        )

    return {
        "Persistence": frame(persistence),
        "Predictability": frame(predictability),
        "Smoothness": frame(smoothness),
    }


def get_graham_criteria(
    revenue: pd.DataFrame,
    total_current_assets: pd.DataFrame,
    total_current_liabilities: pd.DataFrame,
    long_term_debt: pd.DataFrame,
    net_income: pd.DataFrame,
    dividends_paid: pd.DataFrame,
    earnings_per_share: pd.DataFrame,
    share_price: pd.DataFrame,
    book_value_per_share: pd.DataFrame,
    minimum_revenue: float = 800_000_000,
    lag: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    The seven criteria Benjamin Graham sets for the stocks of the defensive investor,
    whether each is met.

    The criteria come from The Intelligent Investor (1973, chapter 14), each 1 when met, 0
    when not and NaN when the data does not reach back far enough:

        - Adequate Size: Revenue >= minimum_revenue
        - Strong Financial Condition: Current Assets >= 2 * Current Liabilities and Long Term
          Debt <= Current Assets - Current Liabilities
        - Earnings Stability: a positive Net Income in each of the past 10 years
        - Dividend Record: dividends paid in each of the past 20 years
        - Earnings Growth: the average EPS of the last 3 years at least a third above the
          average EPS of the 3 years ending ten years earlier
        - Moderate Price to Earnings: Price <= 15 * the average EPS of the last 3 years
        - Moderate Price to Assets: Price / Earnings (3-year average) * Price / Book Value <= 22.5

    Args:
        revenue (pd.DataFrame): The yearly revenue per ticker (rows) and period (columns).
        total_current_assets (pd.DataFrame): The current assets.
        total_current_liabilities (pd.DataFrame): The current liabilities.
        long_term_debt (pd.DataFrame): The long-term debt.
        net_income (pd.DataFrame): The yearly net income.
        dividends_paid (pd.DataFrame): The yearly dividends paid (any sign).
        earnings_per_share (pd.DataFrame): The yearly earnings per share.
        share_price (pd.DataFrame): The share price at the end of the period.
        book_value_per_share (pd.DataFrame): The book value per share.
        minimum_revenue (float, optional): The revenue of the size criterion. Defaults to
            800 million, Graham's $100 million of 1971 in 2025 prices.
        lag (int, optional): The number of columns in a year: 1 for yearly statements and 4 for
            quarterly flows summed over the trailing twelve months. Defaults to 1.

    Returns:
        dict[str, pd.DataFrame]: Every criterion per ticker and period.

    References:
    - Graham, B. (1973). "The Intelligent Investor." Harper & Row, 4th revised edition,
      chapter 14.
    """

    def known(condition: pd.DataFrame, *frames: pd.DataFrame) -> pd.DataFrame:
        # 1 or 0 where every input is known, NaN otherwise.
        mask = reduce(operator.and_, [frame.notna() for frame in frames])
        return condition.astype(float).where(mask)

    def years(frame: pd.DataFrame, count: int, start: int = 0) -> list[pd.DataFrame]:
        return [frame.shift((start + year) * lag, axis=1) for year in range(count)]

    def every_year(frames: list[pd.DataFrame], test) -> pd.DataFrame:
        passed = reduce(operator.and_, [test(frame) for frame in frames])
        return known(passed, *frames)

    working_capital = total_current_assets - total_current_liabilities
    recent_earnings = sum(years(earnings_per_share, 3)) / 3
    past_earnings = sum(years(earnings_per_share, 3, start=10)) / 3
    price_to_earnings = share_price / recent_earnings.where(recent_earnings > 0)
    price_to_book = share_price / book_value_per_share.where(book_value_per_share > 0)

    return {
        "Adequate Size": known(revenue >= minimum_revenue, revenue),
        "Strong Financial Condition": known(
            (total_current_assets >= 2 * total_current_liabilities)
            & (long_term_debt <= working_capital),
            total_current_assets,
            total_current_liabilities,
            long_term_debt,
        ),
        "Earnings Stability": every_year(
            years(net_income, 10), lambda frame: frame > 0
        ),
        "Dividend Record": every_year(
            years(dividends_paid, 20), lambda frame: frame != 0
        ),
        "Earnings Growth": known(
            recent_earnings >= past_earnings + abs(past_earnings) / 3,
            recent_earnings,
            past_earnings,
        ),
        "Moderate Price to Earnings": known(
            (price_to_earnings <= 15) & (recent_earnings > 0),  # noqa: PLR2004
            share_price,
            recent_earnings,
        ),
        "Moderate Price to Assets": known(
            (price_to_earnings * price_to_book <= 22.5)  # noqa: PLR2004
            & (recent_earnings > 0)
            & (book_value_per_share > 0),
            share_price,
            recent_earnings,
            book_value_per_share,
        ),
    }
