"""Growth Model"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.ratios.helpers import map_period_data_to_daily_data


def get_present_value_of_growth_opportunities(
    cost_of_equity: pd.DataFrame,
    earnings_per_share: pd.Series | pd.DataFrame,
    close_prices: pd.DataFrame,
    calculate_daily: bool = False,
    period_end_dates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    The Present Value of Growth Opportunities (PVGO) is the net present value of all future
    investments a company is expected to make.

    It is calculated as the difference between the Close Price and the Earnings Per Share
    divided by the Cost of Equity.

    It is meant to be used as relative valuation metric and therefore doesn't necessarily have a meaning when used
    for one company.


    The formula is as follows:

        PVGO = Close Price — (Earnings Per Share / Cost of Equity)

    Args:
        cost_of_equity (pd.DataFrame): The cost of equity, the return shareholders require.
        earnings_per_share (pd.Series | pd.DataFrame): The earnings per share.
        close_prices (pd.DataFrame): The close prices, per period (rows) and ticker (columns): the
            close at the end of every statement period, or the daily closes with calculate_daily.
        calculate_daily (bool): Whether to calculate the PVGO on a daily basis. If False, the PVGO is calculated
        based on the provided close_prices DataFrame.
        period_end_dates (pd.DataFrame | None, optional): With calculate_daily and yearly or
            quarterly periods (a PeriodIndex) as columns, the last day of the fiscal period behind
            each column, with the tickers as index and the periods as columns: each period's
            no-growth value applies from that day on (see
            financetoolkit.ratios.helpers.map_period_data_to_daily_data). Defaults to None, which
            applies each value from the last day of its column's period.

    Returns:
        pd.DataFrame: The PVGO.

    Notes:
    - Following the textbook derivation of PVGO (e.g. Bodie, Kane & Marcus, "Investments"), the
    no-growth value EPS / r is discounted at r = the cost of equity (the "market capitalization rate"
    for the stock), since both the share Price and EPS are equity-only, per-share quantities.
    """
    no_growth_value = earnings_per_share / cost_of_equity

    if calculate_daily and period_end_dates is not None:
        # A period's earnings are only known after its fiscal period ends, so each no-growth
        # value applies from the period's last day on (the next trading day when it falls on a
        # weekend), as the daily ratios do.
        no_growth_value = map_period_data_to_daily_data(
            period_data=no_growth_value,
            daily_dates=close_prices.index,
            quarterly=no_growth_value.columns.freqstr.startswith("Q"),
            period_end_dates=period_end_dates,
        )
    elif calculate_daily:
        no_growth_value = no_growth_value.T
        no_growth_value.index = pd.PeriodIndex(no_growth_value.index, freq="D")
        # Fill on the union so observations on non-trading dates are not discarded.
        no_growth_value = (
            pd.DataFrame(no_growth_value)
            .reindex(no_growth_value.index.union(close_prices.index))
            .sort_index()
            .ffill()
            .reindex(close_prices.index)
        )
    else:
        no_growth_value = no_growth_value.T

    return close_prices - no_growth_value


def get_sustainable_growth_rate(
    return_on_equity: float | pd.Series | pd.DataFrame,
    retention_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Sustainable Growth Rate (SGR) is the maximum rate at which a company can grow its
    revenue without new equity or more leverage.

    It uses internally generated funds only.

    The formula is as follows:

        SGR = Return on Equity * Retention Ratio

    Also known as: SGR, self-sustainable growth rate.

    Args:
        return_on_equity (float | pd.Series | pd.DataFrame): The return on equity (ROE) of
        the company, measured on the equity at the beginning of the period (Higgins).
        retention_ratio (float | pd.Series | pd.DataFrame): The retention ratio of the company,
        i.e. the proportion of net income retained rather than paid out as dividends. This is
        equal to (1 - Dividend Payout Ratio).

    Returns:
        float | pd.Series | pd.DataFrame: The Sustainable Growth Rate.

    Notes:
    - Growing faster than the SGR without external financing typically requires either
    improving profitability, reducing the dividend payout, or increasing leverage.
    """
    return return_on_equity * retention_ratio


def get_internal_growth_rate(
    return_on_assets: float | pd.Series | pd.DataFrame,
    retention_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Internal Growth Rate (IGR) is the maximum rate at which a company can grow its
    revenue using only its retained earnings.

    It raises no external financing, neither debt nor equity.

    The formula is as follows:

        IGR = (Return on Assets * Retention Ratio) / (1 - (Return on Assets * Retention Ratio))

    Also known as: IGR.

    Args:
        return_on_assets (float | pd.Series | pd.DataFrame): The return on assets (ROA) of
        the company, measured on the total assets at the end of the period.
        retention_ratio (float | pd.Series | pd.DataFrame): The retention ratio of the company,
        i.e. the proportion of net income retained rather than paid out as dividends. This is
        equal to (1 - Dividend Payout Ratio).

    Returns:
        float | pd.Series | pd.DataFrame: The Internal Growth Rate.

    Notes:
    - The IGR is more conservative than the Sustainable Growth Rate (SGR) since it assumes
    no additional debt is raised to fund growth, whereas the SGR assumes the company
    maintains its current level of financial leverage.
    """
    return_on_assets_retained = return_on_assets * retention_ratio

    return return_on_assets_retained / (1 - return_on_assets_retained)


def get_return_decomposition(
    close_prices: pd.DataFrame,
    earnings_per_share: pd.DataFrame,
    dividends_per_share: pd.DataFrame,
    lag: int = 1,
    logarithmic: bool = False,
) -> pd.DataFrame:
    """
    Splits the share price return over a window into the growth of the Earnings per Share
    and the change of the Price-to-Earnings ratio.

    It adds the dividends paid over the window for the total return.

    Since the share price equals the Earnings per Share times the P/E ratio, the price relative
    over a window is the product of the earnings relative and the multiple relative:

        - Price_t / Price_t-lag = (EPS_t / EPS_t-lag) * (P/E_t / P/E_t-lag)
        - Dividend Return = Dividends per Share paid over the window / Price_t-lag
        - Total Return = Price Return + Dividend Return

    In logarithms the parts add up: ln(1 + Total Return) = ln(1 + EPS Growth) +
    ln(1 + P/E Change) + ln(1 + Dividends / Price_t).

    Args:
        close_prices (pd.DataFrame): The share price at the end of every period, with the
            tickers as index and the periods as columns.
        earnings_per_share (pd.DataFrame): The Earnings per Share of every period, in the same
            layout.
        dividends_per_share (pd.DataFrame): The Dividends per Share paid within every period,
            in the same layout.
        lag (int): The number of periods the window spans. Defaults to 1.
        logarithmic (bool): Whether to return log returns, which add up, instead of simple
            returns, which multiply. Defaults to False.

    Returns:
        pd.DataFrame: The Price Return, EPS Growth, P/E Change, Dividend Return and Total Return
        with the tickers and these parts as index and the periods as columns.

    Notes:
    - The P/E ratio has no meaning when the earnings are zero or negative, so the EPS Growth and
    P/E Change are NaN when the Earnings per Share at either end of the window is not positive.
    The Price Return, Dividend Return and Total Return remain.
    - Dividends are added as paid, without reinvestment.
    """
    price_relative = close_prices / close_prices.shift(lag, axis=1)
    previous_earnings = earnings_per_share.shift(lag, axis=1)
    positive_earnings = (earnings_per_share > 0) & (previous_earnings > 0)
    # Without a price at both ends there is no return to split, even when earnings are reported.
    earnings_relative = (earnings_per_share / previous_earnings).where(
        positive_earnings & price_relative.notna()
    )
    multiple_relative = price_relative / earnings_relative

    window_dividends = dividends_per_share.T.rolling(lag, min_periods=lag).sum().T
    dividend_return = window_dividends / close_prices.shift(lag, axis=1)
    total_relative = price_relative + dividend_return

    if logarithmic:
        price_relative, earnings_relative, multiple_relative = (
            np.log(price_relative),
            np.log(earnings_relative),
            np.log(multiple_relative),
        )
        dividend_return = np.log(total_relative) - price_relative
        total_relative = np.log(total_relative)
        parts = {
            "Price Return": price_relative,
            "EPS Growth": earnings_relative,
            "P/E Change": multiple_relative,
            "Dividend Return": dividend_return,
            "Total Return": total_relative,
        }
    else:
        parts = {
            "Price Return": price_relative - 1,
            "EPS Growth": earnings_relative - 1,
            "P/E Change": multiple_relative - 1,
            "Dividend Return": dividend_return,
            "Total Return": total_relative - 1,
        }

    decomposition = pd.concat(parts, names=["Part", "Ticker"])

    return decomposition.swaplevel().reindex(
        pd.MultiIndex.from_product([close_prices.index, list(parts)])
    )


def get_reinvestment_rate(
    net_capital_expenditure: pd.Series | pd.DataFrame,
    change_in_working_capital: pd.Series | pd.DataFrame,
    net_operating_profit_after_taxes: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Damodaran's reinvestment rate.

    It is the share of the after-tax operating income that a company reinvests in its
    operations, in capital expenditure beyond its depreciation and in working capital.

    The formula is as follows:

        - Reinvestment Rate = (Net Capital Expenditure + Change in Working Capital) / NOPAT

    Args:
        net_capital_expenditure (pd.Series | pd.DataFrame): The capital expenditure minus the
            depreciation and amortization.
        change_in_working_capital (pd.Series | pd.DataFrame): The increase in (non-cash)
            working capital, positive when working capital grows.
        net_operating_profit_after_taxes (pd.Series | pd.DataFrame): EBIT * (1 - tax rate).

    Returns:
        pd.Series | pd.DataFrame: The reinvestment rate as a decimal, NaN where the NOPAT is
        not positive (the rate has no meaning there).

    References:
    - Damodaran, A. (2012). "Investment Valuation." Wiley, 3rd edition, chapter 11.
    """
    return (net_capital_expenditure + change_in_working_capital) / (
        net_operating_profit_after_taxes.where(net_operating_profit_after_taxes > 0)
    )


def get_fundamental_growth_rate(
    reinvestment_rate: pd.Series | pd.DataFrame,
    return_on_invested_capital: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Damodaran's fundamental growth rate of operating income: the growth that the reinvestment
    of a company produces at its return on capital.

    The formula is as follows:

        - Expected Growth in EBIT = Reinvestment Rate * Return on Invested Capital

    Args:
        reinvestment_rate (pd.Series | pd.DataFrame): The reinvestment rate.
        return_on_invested_capital (pd.Series | pd.DataFrame): The return on invested capital,
            NOPAT over the invested capital at the beginning of the period.

    Returns:
        pd.Series | pd.DataFrame: The expected growth rate of the operating income per period
        as a decimal.

    References:
    - Damodaran, A. (2012). "Investment Valuation." Wiley, 3rd edition, chapter 11.
    """
    return reinvestment_rate * return_on_invested_capital


def get_sales_to_capital_ratio(
    revenue: pd.Series | pd.DataFrame,
    invested_capital: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Damodaran's sales-to-capital ratio.

    It is the revenue every unit of invested capital generates, which turns revenue growth
    into the reinvestment it requires (Reinvestment = Change in Revenue / Sales to Capital).

    The formula is as follows:

        - Sales to Capital = Revenue / Invested Capital

    Args:
        revenue (pd.Series | pd.DataFrame): The revenue.
        invested_capital (pd.Series | pd.DataFrame): The invested capital.

    Returns:
        pd.Series | pd.DataFrame: The sales-to-capital ratio, NaN where the invested capital is
        not positive.

    References:
    - Damodaran, A. (2012). "Investment Valuation." Wiley, 3rd edition, chapter 11.
    """
    return revenue / invested_capital.where(invested_capital > 0)


def get_rule_of_40(
    revenue_growth: pd.Series | pd.DataFrame,
    profit_margin: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Rule of 40 score of software and other subscription companies: the revenue growth
    plus a profit margin (the free cash flow margin or the EBITDA margin).

    A score of 0.40 (40%) or more is seen as a healthy balance between growth and
    profitability.

    The formula is as follows:

        - Rule of 40 Score = Revenue Growth + Profit Margin

    Args:
        revenue_growth (pd.Series | pd.DataFrame): The year-over-year revenue growth as a
            decimal.
        profit_margin (pd.Series | pd.DataFrame): The free cash flow or EBITDA margin as a
            decimal.

    Returns:
        pd.Series | pd.DataFrame: The Rule of 40 score as a decimal (0.40 is the threshold).
    """
    return revenue_growth + profit_margin
