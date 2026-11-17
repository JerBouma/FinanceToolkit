"""Market (Trading) Liquidity Model"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    VOLATILITY_WINDOW_TRANSLATION,
    by_own_trading_days,
    drop_non_finite,
    to_period_index,
    validate_period,
)

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# The Roll Spread needs a few consecutive changes for a lag-1 autocovariance.
MINIMUM_ROLL_OBSERVATIONS = 3

# The denominator 3 - 2√2 of alpha in Corwin and Schultz (2012).
CORWIN_SCHULTZ_DENOMINATOR = 3 - 2 * np.sqrt(2)

# Ang, Hodrick, Xing and Zhang (2006) require at least 15 daily returns within a month.
MINIMUM_IDIOSYNCRATIC_OBSERVATIONS = 15


def get_amihud_illiquidity(
    returns: pd.Series | pd.DataFrame,
    dollar_volume: pd.Series | pd.DataFrame,
    scale: float = 1_000_000,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Amihud (2002) Illiquidity ratio.

    The Amihud ratio measures the average price impact of trading, i.e. how much the
    price moves per dollar of trading volume. It is the average, over the sample, of
    the absolute return on a day divided by the dollar volume traded that day:

    - ILLIQ = mean( |Return_t| / DollarVolume_t )

    Where `DollarVolume_t` is typically the Close price times the traded Volume on
    day `t`. A high ILLIQ value means that even a small amount of trading moves the
    price a lot -- i.e. the asset is illiquid -- while a low ILLIQ value means the
    asset can absorb a large amount of trading with little price impact -- i.e. the
    asset is liquid. Since the raw ratio is extremely small for actively traded
    assets, it is conventionally scaled up by `scale` (1,000,000 by default, as in
    the original paper) purely for readability. Note that Amihud's sample predates
    the modern mega-cap: for a stock turning over billions of dollars a day the
    scaled ratio is still on the order of 1e-6 and will round away entirely at the
    default rounding, so raise `scale` (e.g. to 1e12) when comparing such names.

    Unlike the bid-ask-based Roll Spread (see `get_roll_spread`), the Amihud ratio
    does not require intraday quote data and can be computed directly from daily
    Close prices and Volume, which makes it the most widely used illiquidity proxy
    in the empirical asset pricing literature.

    Also known as: Amihud illiquidity ratio, ILLIQ, price impact ratio.

    For more information about the method, see the following paper:

    - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and
    Time-Series Effects." Journal of Financial Markets, 5(1), 31-56.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        dollar_volume (pd.Series | pd.DataFrame): A Series or Dataframe of daily dollar
        trading volume (e.g. Close price times Volume), aligned to the same index as
        `returns`.
        scale (float, optional): A multiplier applied to the resulting ratio purely for
        readability, since the raw ratio is otherwise extremely small. Defaults to
        1,000,000.

    Returns:
        pd.Series | pd.DataFrame: Amihud Illiquidity values as float if `returns` is a
        pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        TypeError: If `returns` is not a pd.Series or pd.DataFrame.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_amihud_illiquidity(
                returns.loc[sub_period], dollar_volume.loc[sub_period], scale
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        illiquidity = pd.concat(period_data_list, axis=1)

        return illiquidity.T

    if isinstance(returns, pd.DataFrame):
        return pd.Series(
            {
                column: get_amihud_illiquidity(
                    returns[column], dollar_volume[column], scale
                )
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        daily_illiquidity = drop_non_finite(returns.abs() / dollar_volume)

        return daily_illiquidity.mean() * scale

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


@by_own_trading_days
def get_roll_spread(
    close_prices: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Roll (1984) implied bid-ask spread.

    Roll's model shows that, under a stylized microstructure model in which the true
    (efficient) price follows a random walk and observed trade prices randomly bounce
    between the bid and the ask, the effective spread can be backed out purely from
    the serial covariance of consecutive price changes, without needing any actual
    quote data:

    - Spread = 2 * SQRT( -Cov(dP_t, dP_(t-1)) )

    Where `dP_t = P_t - P_(t-1)` is the price change on day `t`. The bid-ask bounce
    mechanically induces negative first-order serial covariance in `dP_t` (a trade at
    the ask followed by one at the bid looks like a price drop, and vice versa),
    which is why a negative covariance is required for the square root to be
    real-valued. If the estimated covariance is zero or positive -- which can happen
    in practice, especially over short windows or for assets whose price changes are
    dominated by genuine (positively autocorrelated, e.g. trending) information flow
    rather than bid-ask bounce -- Roll's model breaks down and no spread estimate can
    be backed out; NaN is returned in that case rather than an arbitrary or complex
    number.

    Also known as: Roll's implied spread, Roll measure.

    For more information about the method, see the following paper:

    - Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in
    an Efficient Market." The Journal of Finance, 39(4), 1127-1139.

    Args:
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close
        prices.

    Returns:
        pd.Series | pd.DataFrame: The Roll Spread (in price units), the Relative Roll
        Spread (the Roll Spread divided by the mean price, as a decimal: 0.005 is 0.5%
        of the price), the underlying lag-1 autocovariance, and
        whether that autocovariance was negative (i.e. whether a valid estimate could
        be backed out).

    Raises:
        TypeError: If `close_prices` is not a pd.Series or pd.DataFrame.
    """
    if isinstance(close_prices, pd.DataFrame):
        if close_prices.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = close_prices.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_roll_spread(close_prices.loc[sub_period])

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_roll_spread(close_prices[column])
                for column in close_prices.columns
            }
        )
    if isinstance(close_prices, pd.Series):
        prices = close_prices.dropna()
        price_changes = prices.diff().dropna().to_numpy()

        if len(price_changes) <= MINIMUM_ROLL_OBSERVATIONS:
            return pd.Series(
                {
                    "Roll Spread": np.nan,
                    "Relative Roll Spread": np.nan,
                    "Autocovariance": np.nan,
                    "Valid Estimate": False,
                }
            )

        autocovariance = np.cov(price_changes[:-1], price_changes[1:], ddof=1)[0, 1]

        if autocovariance >= 0:
            return pd.Series(
                {
                    "Roll Spread": np.nan,
                    "Relative Roll Spread": np.nan,
                    "Autocovariance": autocovariance,
                    "Valid Estimate": False,
                }
            )

        roll_spread = 2 * np.sqrt(-autocovariance)
        mean_price = prices.mean()

        return pd.Series(
            {
                "Roll Spread": roll_spread,
                "Relative Roll Spread": roll_spread / mean_price,
                "Autocovariance": autocovariance,
                "Valid Estimate": True,
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_corwin_schultz_spread(
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Corwin and Schultz (2012) high-low bid-ask spread per period.

    The daily High is almost always a trade at the ask and the daily Low one at the bid, so
    the High/Low ratio holds both the volatility and the spread. The variance part grows
    with the length of the window while the spread part does not, so comparing the range
    of two single days with the range over both days separates the two:

    - β = [ln(H_t / L_t)]^2 + [ln(H_(t+1) / L_(t+1))]^2
    - γ = [ln(max(H_t, H_(t+1)) / min(L_t, L_(t+1)))]^2
    - α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
    - S = 2 * (e^α - 1) / (1 + e^α)

    Before β and γ are calculated, the second day is adjusted for an overnight return as in
    section III.B of the paper: when its Low is above the previous Close, its High and Low
    are lowered by the gap (Low - previous Close), and when its High is below the previous
    Close, they are raised by the gap (previous Close - High). A two-day spread below zero
    (the range over both days is wide relative to the single days, mostly from a move in
    the price) is set to 0, the paper's main convention, before the spreads of a period
    are averaged.

    The pairs are consecutive trading days of the asset itself (the rows with a High, Low
    and Close, which are each positive) and both days of a pair fall within the period.

    For more information about the method, see the following paper:

    - Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from
    Daily High and Low Prices." The Journal of Finance, 67(2), 719-760.

    Args:
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close
        prices, used for the overnight adjustment.
        period (str, optional): The period to average the two-day spreads over: weekly,
        monthly, quarterly or yearly. Defaults to "monthly".
        rolling (int | None, optional): Average the two-day spreads of the last `rolling`
        periods instead of those of a single period. A window with a period without
        spreads is NaN. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The spread as a decimal of the price (0.005 is 0.5%), with
        the periods as the index (and the assets as columns for a DataFrame).

    Raises:
        TypeError: If the prices are not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(high_prices, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_corwin_schultz_spread(
                    high_prices[column],
                    low_prices[column],
                    close_prices[column],
                    period,
                    rolling,
                )
                for column in high_prices.columns
            }
        )
    if not isinstance(high_prices, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    prices = pd.concat([high_prices, low_prices, close_prices], axis=1)
    # A bad bar (a price of 0 or below) has no logarithm, and a day without a High, Low or Close is not a trading day of the asset, so both leave the pairs.  # noqa: E501
    prices = prices.where(prices > 0).dropna()
    periods = to_period_index(prices.index).asfreq(PERIOD_TRANSLATION[period])

    high, low, close = (prices.iloc[:, number].to_numpy() for number in range(3))

    # The overnight adjustment of the second day: only one of the two gaps can be positive.
    shift = np.maximum(close[:-1] - high[1:], 0) - np.maximum(low[1:] - close[:-1], 0)
    second_high = high[1:] + shift
    second_low = low[1:] + shift

    beta = np.log(high[:-1] / low[:-1]) ** 2 + np.log(second_high / second_low) ** 2
    gamma = (
        np.log(np.maximum(high[:-1], second_high) / np.minimum(low[:-1], second_low))
        ** 2
    )
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / CORWIN_SCHULTZ_DENOMINATOR - np.sqrt(
        gamma / CORWIN_SCHULTZ_DENOMINATOR
    )
    spread = np.maximum(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0)

    same_period = np.asarray(periods[1:] == periods[:-1])
    spreads = pd.Series(spread[same_period], index=periods[1:][same_period])

    if spreads.empty:
        return pd.Series(dtype=float, name=high_prices.name)

    grouped = spreads.groupby(level=0)

    if not rolling:
        return grouped.mean().rename(high_prices.name)

    # The mean of every two-day spread in the window, not the mean of the period means.
    full_range = pd.period_range(
        spreads.index.min(), spreads.index.max(), freq=spreads.index.freq
    )
    totals = grouped.sum().reindex(full_range).rolling(rolling).sum()
    counts = grouped.count().reindex(full_range).rolling(rolling).sum()

    return (totals / counts).rename(high_prices.name)


def get_share_turnover(
    volume: pd.Series | pd.DataFrame,
    shares_outstanding: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Share Turnover per period: the number of shares traded relative to the
    number of shares outstanding.

    Datar, Naik and Radcliffe (1998) use the turnover as a liquidity proxy: a share that
    changes hands often can be traded at a lower cost, and they find that stocks with a
    lower turnover earn higher returns. The turnover of a period is the sum of the daily
    turnover within it, i.e. the share of the outstanding shares traded in the period:

    - Daily Turnover = Volume_t / Shares Outstanding_t
    - Turnover = SUM(Daily Turnover within the period)

    The paper measures the turnover per month and averages it over the three months
    before the month of the returns; `rolling=3` with monthly periods gives that average.

    A period in which a trading day has a Volume but no (positive) number of shares
    outstanding is NaN, as its turnover would be incomplete.

    For more information about the method, see the following paper:

    - Datar, V. T., Naik, N. Y., & Radcliffe, R. (1998). "Liquidity and Stock Returns: An
    Alternative Test." Journal of Financial Markets, 1(2), 203-219.

    Args:
        volume (pd.Series | pd.DataFrame): A Series or Dataframe of daily Volume, NaN on
        the days the asset did not trade.
        shares_outstanding (pd.Series | pd.DataFrame): The shares outstanding on each day,
        aligned to the same index (and columns) as `volume`.
        period (str, optional): The period to sum the daily turnover over: weekly, monthly,
        quarterly or yearly. Defaults to "monthly".
        rolling (int | None, optional): The mean of the turnover of the last `rolling`
        periods instead of the turnover per period. A window with a period without any
        trading day is NaN. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The turnover as a decimal (0.1 is 10% of the shares
        outstanding traded), with the periods as the index.

    Raises:
        TypeError: If `volume` is not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if not isinstance(volume, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    daily_turnover = volume / shares_outstanding.where(shares_outstanding > 0)
    dates = to_period_index(volume.index).asfreq(PERIOD_TRANSLATION[period])

    turnover = daily_turnover.groupby(dates).sum(min_count=1)
    complete = (
        daily_turnover.notna().groupby(dates).sum()
        == volume.notna().groupby(dates).sum()
    )
    turnover = turnover.where(complete)

    if rolling:
        # A period without any trading day in the data (a halt) is a row of its own, so a window always spans `rolling` periods and one that holds such a period is NaN.  # noqa: E501
        full_range = pd.period_range(
            turnover.index.min(),
            turnover.index.max(),
            freq=turnover.index.freq,
            name=turnover.index.name,
        )
        turnover = turnover.reindex(full_range).rolling(rolling).mean()

    return turnover


def get_idiosyncratic_volatility(
    excess_returns: pd.Series | pd.DataFrame,
    factors: pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
    annualize: bool = False,
    minimum_observations: int = MINIMUM_IDIOSYNCRATIC_OBSERVATIONS,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Idiosyncratic Volatility per period: the standard deviation of the
    residuals of a regression of the daily excess returns on the daily factor returns
    within each period.

    Ang, Hodrick, Xing and Zhang (2006) regress the daily excess returns of a stock on the
    Fama and French three factors within each month and find that stocks with a high
    volatility of the residuals earn low returns the next month:

    - Excess Return_t = α + β_1 * (Mkt-RF)_t + β_2 * SMB_t + β_3 * HML_t + ε_t
    - Idiosyncratic Volatility = Std(ε_t) * √(Trading Days per Period)

    Where Std is the sample standard deviation (ddof=1) of the residuals and the trading
    days per period are 252 / 12 (monthly), 252 / 4 (quarterly) or 252 (yearly), the
    scaling of the Volatility of the Risk module; with `annualize` the scaling is √252 for
    every period. Any set of factors works the same way, e.g. the excess return of a
    benchmark alone for the market model (CAPM).

    The regression runs on the days on which the asset and every factor have a return, and
    a period with fewer than `minimum_observations` of those days (or no more than the
    number of coefficients) is NaN. The periods run up to the last date of `excess_returns`, so a period
    after the last factor return is NaN as well.

    For more information about the method, see the following paper:

    - Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). "The Cross-Section of
    Volatility and Expected Returns." The Journal of Finance, 61(1), 259-299.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily excess
        returns.
        factors (pd.DataFrame): The daily factor returns, one column per factor, on the
        same dates as `excess_returns`.
        period (str, optional): The period each regression runs within: weekly, monthly,
        quarterly or yearly. Defaults to "monthly".
        rolling (int | None, optional): Regress on the days of the last `rolling` periods
        instead of a single period, from the period in which the asset's history first
        spans `rolling` periods. Defaults to None.
        annualize (bool, optional): Whether to scale the daily standard deviation by √252
        instead of the square root of the trading days per period. Defaults to False.
        minimum_observations (int, optional): The fewest days a regression runs on.
        Defaults to 15.

    Returns:
        pd.Series | pd.DataFrame: The Idiosyncratic Volatility with the periods as the
        index (and the assets as columns for a DataFrame).

    Raises:
        TypeError: If `excess_returns` is not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(excess_returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_idiosyncratic_volatility(
                    excess_returns[column],
                    factors,
                    period,
                    rolling,
                    annualize,
                    minimum_observations,
                )
                for column in excess_returns.columns
            }
        )
    if not isinstance(excess_returns, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    data = pd.concat(
        [excess_returns.rename("Excess Return"), factors.reindex(excess_returns.index)],
        axis=1,
    ).dropna()

    if data.empty:
        return pd.Series(dtype=float, name=excess_returns.name)

    periods = to_period_index(data.index).asfreq(PERIOD_TRANSLATION[period])
    # The periods run up to the last date of the returns, so the periods after the last
    # factor return (the Fama-French data lag a month or more, and so does their risk-free
    # rate the excess returns are taken over) are NaN rows rather than missing, also with a
    # rolling window that still holds earlier days.
    last_period = (
        to_period_index(excess_returns.index).asfreq(PERIOD_TRANSLATION[period]).max()
    )
    window = rolling if rolling else 1
    scale = np.sqrt(VOLATILITY_WINDOW_TRANSLATION["yearly" if annualize else period])
    minimum = max(minimum_observations, factors.shape[1] + 2)

    target = data["Excess Return"].to_numpy()
    design = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy()])

    volatility = {}

    # A rolling window starts once it holds `rolling` periods of the asset's history.
    for end in pd.period_range(
        periods.min() + window - 1, periods.max(), freq=periods.freq
    ):
        rows = np.asarray((periods > end - window) & (periods <= end))

        if rows.sum() < minimum:
            volatility[end] = np.nan
            continue

        coefficients = np.linalg.lstsq(design[rows], target[rows], rcond=None)[0]
        residuals = target[rows] - design[rows] @ coefficients
        volatility[end] = residuals.std(ddof=1) * scale

    volatility = pd.Series(volatility, name=excess_returns.name, dtype=float)

    if last_period > periods.max():
        volatility = volatility.reindex(
            pd.period_range(periods.min() + window - 1, last_period, freq=periods.freq)
        )

    return volatility.rename_axis(excess_returns.index.name)
