"""Market (Trading) Liquidity Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# The Roll Spread needs a few consecutive changes for a lag-1 autocovariance.
MINIMUM_ROLL_OBSERVATIONS = 3


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
    assets, it is conventionally scaled up by `scale` (1,000,000 by default) purely
    for readability.

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
        daily_illiquidity = (returns.abs() / dollar_volume).replace(
            [np.inf, -np.inf], np.nan
        )

        return daily_illiquidity.mean() * scale

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


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
        pd.Series | pd.DataFrame: The Roll Spread (in price units), the Roll Spread
        as a percentage of the mean price, the underlying lag-1 autocovariance, and
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
                    "Roll Spread (%)": np.nan,
                    "Autocovariance": np.nan,
                    "Valid Estimate": False,
                }
            )

        autocovariance = np.cov(price_changes[:-1], price_changes[1:], ddof=1)[0, 1]

        if autocovariance >= 0:
            return pd.Series(
                {
                    "Roll Spread": np.nan,
                    "Roll Spread (%)": np.nan,
                    "Autocovariance": autocovariance,
                    "Valid Estimate": False,
                }
            )

        roll_spread = 2 * np.sqrt(-autocovariance)
        mean_price = prices.mean()

        return pd.Series(
            {
                "Roll Spread": roll_spread,
                "Roll Spread (%)": roll_spread / mean_price * 100,
                "Autocovariance": autocovariance,
                "Valid Estimate": True,
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")
