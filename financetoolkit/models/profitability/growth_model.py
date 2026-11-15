"""Growth Model"""

__docformat__ = "google"


import numpy as np
import pandas as pd


def get_present_value_of_growth_opportunities(
    cost_of_equity: pd.DataFrame,
    earnings_per_share: pd.Series | pd.DataFrame,
    close_prices: pd.DataFrame,
    calculate_daily: bool = False,
) -> pd.DataFrame:
    """
    The Present Value of Growth Opportunities (PVGO) is the net present value of all future investments a company is
    expected to make. It is calculated as the difference between the Close Price and the Earnings Per Share divided by
    the Cost of Equity.

    It is meant to be used as relative valuation metric and therefore doesn't necessarily have a meaning when used
    for one company.


    The formula is as follows:

        PVGO = Close Price — (Earnings Per Share / Cost of Equity)

    Args:
        cost_of_equity (pd.DataFrame): The cost of equity, the return shareholders require.
        earnings_per_share (pd.Series | pd.DataFrame): The earnings per share.
        close_prices (pd.DataFrame): The close prices.
        calculate_daily (bool): Whether to calculate the PVGO on a daily basis. If False, the PVGO is calculated
        based on the provided close_prices DataFrame.

    Returns:
        pd.DataFrame: The PVGO.

    Notes:
    - Following the textbook derivation of PVGO (e.g. Bodie, Kane & Marcus, "Investments"), the
    no-growth value EPS / r is discounted at r = the cost of equity (the "market capitalization rate"
    for the stock), since both the share Price and EPS are equity-only, per-share quantities.
    """
    no_growth_value = (earnings_per_share / cost_of_equity).T

    if calculate_daily:
        no_growth_value.index = pd.PeriodIndex(no_growth_value.index, freq="D")
        # Fill on the union so observations on non-trading dates are not discarded.
        no_growth_value = (
            pd.DataFrame(no_growth_value)
            .reindex(no_growth_value.index.union(close_prices.index))
            .sort_index()
            .ffill()
            .reindex(close_prices.index)
        )

    return close_prices - no_growth_value


def get_sustainable_growth_rate(
    return_on_equity: float | pd.Series | pd.DataFrame,
    retention_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Sustainable Growth Rate (SGR) is the maximum rate at which a company can grow its
    revenue, using internally generated funds only, without having to raise additional equity
    or increase its financial leverage.

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
    revenue using only its retained earnings, without raising any external financing
    (neither debt nor equity).

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
    Splits the share price return over a window into the growth of the Earnings per Share and
    the change of the Price-to-Earnings ratio, and adds the dividends paid over the window for
    the total return.

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
