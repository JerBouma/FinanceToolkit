"""Market (Trading) Liquidity Model"""

__docformat__ = "google"


import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.risk.helpers import (
    average_two_day_spreads,
    get_corwin_schultz_pairs,
    get_days_per_period,
    get_slope_through_origin,
    sum_per_period,
)
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

# k_2 = E[ln(H / L)] / σ = √(8 / π) for a Brownian motion within a day, the constant of the
# Bekker-Parkinson volatility in Corwin and Schultz (2012).
BEKKER_PARKINSON_K2 = np.sqrt(8 / np.pi)

# EDGE needs at least two pairs of days with a price change, as in the authors' implementation.
MINIMUM_EDGE_PRICE_CHANGES = 2

# The price impact regressions have one coefficient, so they need two observations.
MINIMUM_LAMBDA_OBSERVATIONS = 2

# Easley, López de Prado and O'Hara (2012) measure VPIN over 50 volume buckets.
VPIN_BUCKETS = 50

# The share of a bucket by which the cumulative volume may fall short of a bucket edge through
# floating point rounding and still complete the bucket on that day.
VPIN_EDGE_TOLERANCE = 1e-9

# Ang, Hodrick, Xing and Zhang (2006) require at least 15 daily returns within a month.
MINIMUM_IDIOSYNCRATIC_OBSERVATIONS = 15


def get_amihud_illiquidity(
    returns: pd.Series | pd.DataFrame,
    dollar_volume: pd.Series | pd.DataFrame,
    scale: float = 1_000_000,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Amihud (2002) Illiquidity ratio: how much the price moves per dollar traded,
    high for an illiquid asset.

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
    Calculate the Roll (1984) implied bid-ask spread: the cost of a round trip implied by the
    bounce of the price between the bid and the ask.

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
    Calculate the Corwin and Schultz (2012) high-low bid-ask spread per period: the cost of a
    round trip estimated from the daily High and Low.

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

    Shares similar logic:
    - RiskLabAI, corwin_schultz_estimator: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/microstructural_features/corwin_schultz.py#L106

    Args:
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close
        prices, used for the overnight adjustment.
        period (str, optional): The period to average the two-day spreads over.
        Options are: "weekly", "monthly", "quarterly", "yearly". Defaults to "monthly".
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

    pairs = get_corwin_schultz_pairs(high_prices, low_prices, close_prices, period)
    beta, gamma = pairs["Beta"], pairs["Gamma"]

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / CORWIN_SCHULTZ_DENOMINATOR - np.sqrt(
        gamma / CORWIN_SCHULTZ_DENOMINATOR
    )
    spreads = np.maximum(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0)

    return average_two_day_spreads(spreads, rolling, high_prices.name)


def get_bekker_parkinson_volatility(
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Bekker-Parkinson Volatility per period: the volatility that remains in the
    High/Low range of two consecutive days once the bid-ask spread of Corwin and Schultz
    (2012) is taken out.

    Corwin and Schultz write the expected squared log range of one day and of two days as
    a function of the spread (α) and the volatility (σ). The spread is the part that does not
    grow with the length of the window, and solving the same two equations for σ instead of
    α gives the volatility:

    - β = [ln(H_t / L_t)]^2 + [ln(H_(t+1) / L_(t+1))]^2
    - γ = [ln(max(H_t, H_(t+1)) / min(L_t, L_(t+1)))]^2
    - k_2 = √(8 / π)
    - σ = (√(β / 2) - √β) / (k_2 * (3 - 2√2)) + √(γ / (k_2^2 * (3 - 2√2)))

    López de Prado (2018, section 19.3.4 and snippet 19.2) derives it this way, and
    RiskLabAI names it the Bekker-Parkinson volatility: the range-based volatility of
    Parkinson (1980) corrected for the spread. The volatility of each pair is set to 0 when it is negative (the range over
    the two days is narrow relative to the single days), as in López de Prado's snippet
    19.2, and the volatilities of the pairs of a period are averaged. The result is the
    volatility of the log price (continuously compounded) over one day, scaled to the period
    as the Parkinson Volatility of `realized_volatility_model.get_parkinson_volatility` is:
    multiplied by the square root of the trading days in the period (252 / 12 for monthly).

    The pairs, the overnight adjustment of the second day and the rule that both days fall
    within the period are those of `get_corwin_schultz_spread`.

    For more information about the method, see the following sources:

    - Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from
    Daily High and Low Prices." The Journal of Finance, 67(2), 719-760.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
    19.3.4.
    - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the
    Rate of Return." Journal of Business, 53(1), 61-65.

    RiskLabAI's estimate averages β over a rolling window of days while taking γ of the last
    pair only and does not adjust for the overnight return; here σ is taken per pair (with the
    overnight adjustment) and averaged per period, as the Corwin-Schultz spread is. On a single
    pair without an overnight gap the two use the same formula.

    Shares similar logic:
    - RiskLabAI, bekker_parkinson_volatility_estimates: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/microstructural_features/bekker_parkinson_volatility_estimator.py#L51

    Args:
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close
        prices, used for the overnight adjustment.
        period (str, optional): The period to average the two-day volatilities over and
        scale to. Options are: "weekly", "monthly", "quarterly", "yearly". Defaults to "monthly".
        rolling (int | None, optional): Average the two-day volatilities of the last
        `rolling` periods instead of those of a single period. A window with a period
        without pairs is NaN. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The volatility as a decimal, with the periods as the index
        (and the assets as columns for a DataFrame).

    Raises:
        TypeError: If the prices are not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(high_prices, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_bekker_parkinson_volatility(
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

    pairs = get_corwin_schultz_pairs(high_prices, low_prices, close_prices, period)
    beta, gamma = pairs["Beta"], pairs["Gamma"]

    volatilities = np.maximum(
        (np.sqrt(beta / 2) - np.sqrt(beta))
        / (BEKKER_PARKINSON_K2 * CORWIN_SCHULTZ_DENOMINATOR)
        + np.sqrt(gamma / (BEKKER_PARKINSON_K2**2 * CORWIN_SCHULTZ_DENOMINATOR)),
        0,
    )
    days_per_period = get_days_per_period(high_prices.dropna(), period)

    return average_two_day_spreads(volatilities, rolling, high_prices.name) * np.sqrt(
        days_per_period
    )


def get_abdi_ranaldo_spread(
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Abdi and Ranaldo (2017) close-high-low bid-ask spread per period: the cost of
    a round trip estimated from the distance of the Close to the middle of the daily range.

    The mid-range (the average of the log High and log Low) of a day is a proxy for the
    efficient price that holds no spread, while the Close is a trade at the bid or the ask.
    The distance of the Close to the mid-range of its own day and to that of the next day
    therefore moves with the spread, and their product removes the volatility of the
    efficient price between the two days:

    - η_t = (ln H_t + ln L_t) / 2
    - c_t = ln C_t
    - S_t^2 = 4 * (c_t - η_t) * (c_t - η_(t+1))
    - S_t = √(max(S_t^2, 0))
    - S = Mean(S_t within the period)

    This is the "two-day corrected" version of the paper (equation 11), which the authors
    use in their empirical analysis: a two-day estimate of the squared spread below zero
    is set to 0 before the square root is taken and the two-day spreads of a period are
    averaged. Unlike the Corwin-Schultz spread, the estimator needs no adjustment for the
    overnight return.

    The pairs are consecutive trading days of the asset itself (the rows with a High, Low
    and Close, which are each positive) and both days of a pair fall within the period.

    For more information about the method, see the following paper:

    - Abdi, F., & Ranaldo, A. (2017). "A Simple Estimation of Bid-Ask Spreads from Daily
    Close, High, and Low Prices." The Review of Financial Studies, 30(12), 4437-4480.

    Args:
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close
        prices.
        period (str, optional): The period to average the two-day spreads over.
        Options are: "weekly", "monthly", "quarterly", "yearly". Defaults to "monthly".
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
                column: get_abdi_ranaldo_spread(
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
    prices = np.log(prices.where(prices > 0).dropna())
    periods = to_period_index(prices.index).asfreq(PERIOD_TRANSLATION[period])

    high, low, close = (prices.iloc[:, number].to_numpy() for number in range(3))
    mid_range = (high + low) / 2

    squared_spread = 4 * (close[:-1] - mid_range[:-1]) * (close[:-1] - mid_range[1:])
    spread = np.sqrt(np.maximum(squared_spread, 0))

    same_period = np.asarray(periods[1:] == periods[:-1])
    spreads = pd.Series(spread[same_period], index=periods[1:][same_period])

    return average_two_day_spreads(spreads, rolling, high_prices.name)


def get_edge_spread(
    open_prices: pd.Series | pd.DataFrame,
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the EDGE bid-ask spread of Ardia, Guidotti and Kroencke (2024) per period: the
    cost of a round trip estimated from the Open, High, Low and Close.

    EDGE (the Efficient Discrete Generalized Estimator) combines four moment conditions on
    the Open, High, Low and Close: two measure the spread at the open (OHL and OHLC) and
    two at the close (CHL, a generalization of Abdi and Ranaldo (2017), and CHLO). They
    are paired into two conditions and weighted by the Generalized Method of Moments, which
    minimizes the variance of the estimate. With the log prices o, h, l and c and the
    mid-range η = (h + l) / 2 of two consecutive trading days t-1 and t:

    - τ_t = 1 if h_t ≠ l_t or l_t ≠ c_(t-1), else 0
    - π_o = -8 / (P[o_t ≠ h_t, τ_t = 1] + P[o_t ≠ l_t, τ_t = 1])
    - π_c = -8 / (P[c_(t-1) ≠ h_(t-1), τ_t = 1] + P[c_(t-1) ≠ l_(t-1), τ_t = 1])
    - x_1 = π_o / 2 * (η_t - o_t) * (o_t - η_(t-1)) + π_c / 2 * (η_t - c_(t-1)) * (c_(t-1) - η_(t-1))
    - x_2 = π_o / 2 * (η_t - o_t) * (o_t - c_(t-1)) + π_c / 2 * (o_t - c_(t-1)) * (c_(t-1) - η_(t-1))
    - w_1 = Var[x_2] / (Var[x_1] + Var[x_2]) and w_2 = Var[x_1] / (Var[x_1] + Var[x_2])
    - S^2 = w_1 * E[x_1] + w_2 * E[x_2]
    - S = √(max(S^2, 0))

    The returns η_t - o_t, η_t - c_(t-1) and o_t - c_(t-1) in the first factor of each
    product are de-meaned over the days with τ_t = 1, as in the authors' implementation,
    and the probabilities, expectations and (population) variances are the averages over
    the pairs of the period. When Var[x_1] + Var[x_2] is not positive, the two expectations
    get equal weights.

    A squared spread below zero is set to 0 after the estimate of the period, as the paper
    does (equation 26). Unlike resetting each two-day estimate, which the Corwin-Schultz
    and Abdi-Ranaldo spreads do, this keeps the estimate consistent: its bias declines as
    the period gets longer. A period with fewer than two pairs with τ_t = 1, or in which
    the Open (or the previous Close) equals both the High and the Low on every pair with
    τ_t = 1 (only single-price days), is NaN.

    The pairs are consecutive trading days of the asset itself (the rows with an Open,
    High, Low and Close, which are each positive) and both days of a pair fall within the
    period.

    For more information about the method, see the following paper:

    - Ardia, D., Guidotti, E., & Kroencke, T. A. (2024). "Efficient Estimation of Bid-Ask
    Spreads from Open, High, Low, and Close Prices." Journal of Financial Economics, 161,
    103916.

    Shares similar logic:
    - RiskLabAI, edge_estimator: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/microstructural_features/edge.py#L42

    Args:
        open_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Open prices.
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close
        prices.
        period (str, optional): The period to estimate the spread over. Options are: "weekly", "monthly", "quarterly",
        "yearly". Defaults to "monthly".
        rolling (int | None, optional): Estimate the spread from the pairs of the last
        `rolling` periods together instead of those of a single period. A window with a
        period without pairs is NaN. Defaults to None.

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
                column: get_edge_spread(
                    open_prices[column],
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

    prices = pd.concat([open_prices, high_prices, low_prices, close_prices], axis=1)
    # A bad bar (a price of 0 or below) has no logarithm, and a day without all four prices is not a trading day of the asset, so both leave the pairs.  # noqa: E501
    prices = np.log(prices.where(prices > 0).dropna())
    periods = to_period_index(prices.index).asfreq(PERIOD_TRANSLATION[period])

    open_, high, low, close = (prices.iloc[:, number].to_numpy() for number in range(4))
    mid_range = (high + low) / 2

    # Day t-1 is the first day of a pair and day t the second, as in the paper.
    previous_high, previous_low, previous_close = high[:-1], low[:-1], close[:-1]
    previous_mid_range = mid_range[:-1]
    open_, high, low, mid_range = open_[1:], high[1:], low[1:], mid_range[1:]

    # τ_t marks the pairs with a price change: a day without trades repeats the previous Close as its Open, High, Low and Close.  # noqa: E501
    tau = ((high != low) | (low != previous_close)).astype(float)
    pairs = pd.DataFrame(
        {
            "Mid To Open": mid_range - open_,
            "Open To Previous Mid": open_ - previous_mid_range,
            "Mid To Previous Close": mid_range - previous_close,
            "Previous Close To Previous Mid": previous_close - previous_mid_range,
            "Open To Previous Close": open_ - previous_close,
            "Tau": tau,
            "Open Not High": tau * (open_ != high),
            "Open Not Low": tau * (open_ != low),
            "Previous Close Not High": tau * (previous_close != previous_high),
            "Previous Close Not Low": tau * (previous_close != previous_low),
        },
        index=periods[1:],
    )
    pairs = pairs[np.asarray(periods[1:] == periods[:-1])]

    if pairs.empty:
        return pd.Series(dtype=float, name=high_prices.name)

    # x_1 and x_2 are each π_o / 2 * (r_a - ā_a * τ_t) * r_b + π_c / 2 * (r_c - ā_c * τ_t) * r_d, with ā the mean return over the pairs with τ_t = 1 of the period. As the product of the coefficients (π_o / 2, -π_o / 2 * ā_a, π_c / 2, -π_c / 2 * ā_c) and z = (r_a * r_b, τ_t * r_b, r_c * r_d, τ_t * r_d), E[x] and E[x^2] follow from the means of z and of z * z' per period, so the pairs are summed once rather than revisited per period or window.  # noqa: E501
    conditions = {
        "First": [
            "Mid To Open",
            "Open To Previous Mid",
            "Mid To Previous Close",
            "Previous Close To Previous Mid",
        ],
        "Second": [
            "Mid To Open",
            "Open To Previous Close",
            "Open To Previous Close",
            "Previous Close To Previous Mid",
        ],
    }
    blocks, labels = [pairs.to_numpy()], list(pairs.columns)
    for condition, (r_a, r_b, r_c, r_d) in conditions.items():
        z = np.column_stack(
            [
                pairs[r_a] * pairs[r_b],
                pairs["Tau"] * pairs[r_b],
                pairs[r_c] * pairs[r_d],
                pairs["Tau"] * pairs[r_d],
            ]
        )
        blocks += [z, (z[:, :, None] * z[:, None, :]).reshape(len(z), 16)]
        labels += [f"{condition} {i}" for i in range(4)]
        labels += [f"{condition} {i}{j}" for i in range(4) for j in range(4)]

    moments = pd.DataFrame(np.hstack(blocks), index=pairs.index, columns=labels)
    grouped = moments.groupby(level=0)
    sums, counts = grouped.sum(), grouped.size()

    # Per period, or per window of `rolling` periods with every pair in it together rather than the mean of the period estimates. A window with a period without pairs is NaN.  # noqa: E501
    if rolling:
        full_range = pd.period_range(
            sums.index.min(), sums.index.max(), freq=sums.index.freq
        )
        sums = sums.reindex(full_range).rolling(rolling).sum()
        counts = counts.reindex(full_range).rolling(rolling).sum()

    means = sums.div(counts, axis=0)

    tau_share = means["Tau"]
    open_probability = means["Open Not High"] + means["Open Not Low"]
    close_probability = (
        means["Previous Close Not High"] + means["Previous Close Not Low"]
    )

    # π_o / 2 and π_c / 2 are -4 divided by the probabilities.
    open_weight = -4 / open_probability
    close_weight = -4 / close_probability

    # A period that is not estimable has an infinite weight; it is set to NaN below.
    with np.errstate(divide="ignore", invalid="ignore"):
        expectations, variances = {}, {}
        for condition, (r_a, _, r_c, _) in conditions.items():
            # The second and fourth coefficients de-mean r_a and r_c over the pairs with τ_t = 1 (a pair without a price change keeps its return of 0).  # noqa: E501
            coefficients = np.column_stack(
                [
                    open_weight,
                    -open_weight * means[r_a] / tau_share,
                    close_weight,
                    -close_weight * means[r_c] / tau_share,
                ]
            )
            mean_z = means[[f"{condition} {i}" for i in range(4)]].to_numpy()
            mean_zz = (
                means[[f"{condition} {i}{j}" for i in range(4) for j in range(4)]]
                .to_numpy()
                .reshape(-1, 4, 4)
            )

            expectations[condition] = (coefficients * mean_z).sum(axis=1)
            variances[condition] = (
                np.einsum("wi,wij,wj->w", coefficients, mean_zz, coefficients)
                - expectations[condition] ** 2
            )

        total_variance = variances["First"] + variances["Second"]
        squared_spread = np.where(
            total_variance > 0,
            (
                variances["Second"] * expectations["First"]
                + variances["First"] * expectations["Second"]
            )
            / total_variance,
            (expectations["First"] + expectations["Second"]) / 2,
        )

    estimable = (
        (sums["Tau"] >= MINIMUM_EDGE_PRICE_CHANGES)
        & (open_probability != 0)
        & (close_probability != 0)
    )

    # A negative squared spread of the period (or window) is set to 0 (equation 26).
    return pd.Series(
        np.sqrt(np.maximum(squared_spread, 0)), index=means.index, name=high_prices.name
    ).where(estimable)


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
    before the month of the returns; a rolling window of 3 monthly periods gives that average.

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
        period (str, optional): The period to sum the daily turnover over. Options are: "weekly", "monthly",
        "quarterly", "yearly". Defaults to "monthly".
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

    Where Std is the sample standard deviation (with n - 1 in the denominator) of the residuals and the trading
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
        period (str, optional): The period each regression runs within. Options are: "weekly",
        "monthly", "quarterly", "yearly". Defaults to "monthly".
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


def get_kyle_lambda(
    close_prices: pd.Series | pd.DataFrame,
    volume: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
    scale: float = 1_000_000,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Kyle's lambda per period: the change of the price per share of signed volume,
    as the slope of a regression of the price changes on the signed volume.

    In the model of Kyle (1985), market makers cannot tell informed from uninformed orders
    and move the price by λ for every unit of net order flow, so λ measures the depth of the
    market: a high λ is an illiquid market in which an order moves the price a lot. López de
    Prado (2018, section 19.4.1) estimates it from bars, signing each bar's volume with the
    tick rule:

    - ΔP_t = P_t - P_(t-1)
    - b_t = sign(ΔP_t), or b_(t-1) when ΔP_t = 0
    - ΔP_t = λ * (b_t * V_t) + ε_t
    - λ = Σ(ΔP_t * b_t * V_t) / Σ((b_t * V_t)^2)

    Where P is the Close and V the Volume of a day, and the regression runs without an
    intercept on the days within the period, as in the book. λ is multiplied by `scale` for
    readability: with the default of 1,000,000 it is the change of the price per million
    shares of net buying.

    For more information about the method, see the following sources:

    - Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." Econometrica, 53(6),
    1315-1335.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
    19.4.1.

    The lambda is the regression slope of the book rather than the mean of the ratio
    ΔP_t / (b_t * V_t), which would weigh every day equally however small its volume.

    Args:
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        volume (pd.Series | pd.DataFrame): The daily Volume on the same dates.
        period (str, optional): The period each regression runs within. Options are: "weekly",
        "monthly", "quarterly", "yearly". Defaults to "monthly".
        rolling (int | None, optional): Regress on the days of the last `rolling` periods
        instead of a single period. A window with a period without days is NaN. Defaults to
        None.
        scale (float, optional): A multiplier applied to λ for readability. Defaults to
        1,000,000.

    Returns:
        pd.Series | pd.DataFrame: Kyle's lambda with the periods as the index (and the assets
        as columns for a DataFrame).

    Raises:
        TypeError: If the prices are not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(close_prices, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_kyle_lambda(
                    close_prices[column], volume[column], period, rolling, scale
                )
                for column in close_prices.columns
            }
        )
    if not isinstance(close_prices, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    # The asset's own trading days: a day without a (positive) price or a Volume is left out.
    data = pd.concat([close_prices, volume], axis=1)
    data = data[(data.iloc[:, 0] > 0) & (data.iloc[:, 1] >= 0)]
    change = data.iloc[:, 0].diff()
    # The tick rule: a day without a price change keeps the sign of the last change.
    sign = np.sign(change).replace(0, np.nan).ffill()

    return (
        get_slope_through_origin(
            change.rename(close_prices.name),
            sign * data.iloc[:, 1],
            period,
            rolling,
            MINIMUM_LAMBDA_OBSERVATIONS,
        )
        * scale
    )


def get_amihud_lambda(
    prices: pd.Series | pd.DataFrame,
    dollar_volume: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
    scale: float = 1_000_000,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Amihud's lambda per period: the absolute log return per dollar traded, as the
    slope of a regression of the absolute log returns on the dollar volume.

    Amihud (2002) measures illiquidity as the mean of the ratio of the absolute return to the
    dollar volume (`get_amihud_illiquidity`). López de Prado (2018, section 19.4.2) estimates
    the same price impact as a regression slope instead:

    - r_t = ln(P_t / P_(t-1))
    - |r_t| = λ * DV_t + ε_t
    - λ = Σ(|r_t| * DV_t) / Σ(DV_t^2)

    Where P is the price, DV the dollar volume of a day and the regression runs without an
    intercept on the days within the period, as in the book. Unlike the mean of the ratios,
    the slope weighs the days by their dollar volume, so a quiet day with a large move does
    not dominate it. λ is multiplied by `scale` for readability: with the default of
    1,000,000 it is the absolute log return per million dollars traded.

    For more information about the method, see the following sources:

    - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series
    Effects." Journal of Financial Markets, 5(1), 31-56.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
    19.4.2.

    The lambda is the regression slope of the book; the mean of the ratio |r_t| / DV_t is
    Amihud's ratio instead (`get_amihud_illiquidity`).

    Args:
        prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily prices for the
        returns (e.g. the Adjusted Close).
        dollar_volume (pd.Series | pd.DataFrame): The daily dollar volume (e.g. the Close
        times the Volume) on the same dates.
        period (str, optional): The period each regression runs within. Options are: "weekly",
        "monthly", "quarterly", "yearly". Defaults to "monthly".
        rolling (int | None, optional): Regress on the days of the last `rolling` periods
        instead of a single period. A window with a period without days is NaN. Defaults to
        None.
        scale (float, optional): A multiplier applied to λ for readability. Defaults to
        1,000,000.

    Returns:
        pd.Series | pd.DataFrame: Amihud's lambda with the periods as the index (and the
        assets as columns for a DataFrame).

    Raises:
        TypeError: If the prices are not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(prices, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_amihud_lambda(
                    prices[column], dollar_volume[column], period, rolling, scale
                )
                for column in prices.columns
            }
        )
    if not isinstance(prices, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    data = pd.concat([prices, dollar_volume], axis=1)
    data = data[(data.iloc[:, 0] > 0) & (data.iloc[:, 1] >= 0)]
    log_return = np.log(data.iloc[:, 0]).diff()

    return (
        get_slope_through_origin(
            log_return.abs().rename(prices.name),
            data.iloc[:, 1],
            period,
            rolling,
            MINIMUM_LAMBDA_OBSERVATIONS,
        )
        * scale
    )


def get_hasbrouck_lambda(
    prices: pd.Series | pd.DataFrame,
    dollar_volume: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
    scale: float = 1_000_000,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Hasbrouck's lambda per period: the log return per square root of the signed
    dollar volume, as the slope of a regression of the log returns on the signed square root
    of the dollar volume.

    Hasbrouck (2009) estimates the price impact of trading with the square root of the
    signed dollar volume, which captures that the impact grows less than proportionally with
    the size of an order. López de Prado (2018, section 19.4.3) estimates it from bars,
    signing each bar with the tick rule:

    - r_t = ln(P_t / P_(t-1))
    - b_t = sign(r_t), or b_(t-1) when r_t = 0
    - r_t = λ * (b_t * √DV_t) + ε_t
    - λ = Σ(r_t * b_t * √DV_t) / Σ(DV_t)

    Where P is the price, DV the dollar volume of a day and the regression runs without an
    intercept on the days within the period, as in the book. λ is multiplied by `scale` for
    readability.

    For more information about the method, see the following sources:

    - Hasbrouck, J. (2009). "Trading Costs and Returns for U.S. Equities: Estimating
    Effective Costs from Daily Data." The Journal of Finance, 64(3), 1445-1477.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
    19.4.3.

    The lambda is the regression slope of the book rather than the mean of the ratio
    r_t / (b_t * √DV_t), and it regresses the signed log return r_t, as in the book, rather
    than its absolute value.

    Args:
        prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily prices for the
        returns (e.g. the Adjusted Close).
        dollar_volume (pd.Series | pd.DataFrame): The daily dollar volume (e.g. the Close
        times the Volume) on the same dates.
        period (str, optional): The period each regression runs within. Options are: "weekly",
        "monthly", "quarterly", "yearly". Defaults to "monthly".
        rolling (int | None, optional): Regress on the days of the last `rolling` periods
        instead of a single period. A window with a period without days is NaN. Defaults to
        None.
        scale (float, optional): A multiplier applied to λ for readability. Defaults to
        1,000,000.

    Returns:
        pd.Series | pd.DataFrame: Hasbrouck's lambda with the periods as the index (and the
        assets as columns for a DataFrame).

    Raises:
        TypeError: If the prices are not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(prices, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_hasbrouck_lambda(
                    prices[column], dollar_volume[column], period, rolling, scale
                )
                for column in prices.columns
            }
        )
    if not isinstance(prices, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    data = pd.concat([prices, dollar_volume], axis=1)
    data = data[(data.iloc[:, 0] > 0) & (data.iloc[:, 1] >= 0)]
    log_return = np.log(data.iloc[:, 0]).diff()
    sign = np.sign(log_return).replace(0, np.nan).ffill()

    return (
        get_slope_through_origin(
            log_return.rename(prices.name),
            sign * np.sqrt(data.iloc[:, 1]),
            period,
            rolling,
            MINIMUM_LAMBDA_OBSERVATIONS,
        )
        * scale
    )


def get_roll_impact(
    close_prices: pd.Series | pd.DataFrame,
    dollar_volume: pd.Series | pd.DataFrame,
    period: str = "monthly",
    rolling: int | None = None,
    scale: float = 1_000_000,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Roll Impact per period: the Roll (1984) spread of the period divided by
    its mean dollar volume, the spread paid per dollar traded.

    López de Prado (2018, section 19.3.2) divides the Roll measure by the dollar volume, so
    that the spread an asset pays relates to how much of it trades:

    - ΔP_t = P_t - P_(t-1)
    - Roll Spread = 2 * √(-Cov(ΔP_t, ΔP_(t-1)))
    - Roll Impact = Roll Spread / Mean(DV_t)

    Where P is the Close, DV the dollar volume of a day, Cov the sample autocovariance
    (with n - 1 in the denominator) of the consecutive price changes within the period and the mean runs over the
    days of the period. A period with an autocovariance of zero or above has no Roll spread
    and is NaN, as in `get_roll_spread`, and so is a period with fewer than three pairs of
    consecutive price changes. The changes are those between the prices within the period,
    so a pair of changes spans three days of the period (also in a rolling window). The
    result is multiplied by `scale` for readability: with the default of 1,000,000 it is the
    spread in price units per million dollars traded a day.

    For more information about the method, see the following sources:

    - Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in an
    Efficient Market." The Journal of Finance, 39(4), 1127-1139.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
    19.3.2.

    A positive autocovariance gives NaN rather than the spread of its absolute value,
    since Roll's model has no spread then, and the spread is divided by the mean dollar
    volume of the days it is estimated from.

    Args:
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        dollar_volume (pd.Series | pd.DataFrame): The daily dollar volume (e.g. the Close
        times the Volume) on the same dates.
        period (str, optional): The period to estimate the impact over. Options are: "weekly", "monthly", "quarterly",
        "yearly". Defaults to "monthly".
        rolling (int | None, optional): Estimate from the days of the last `rolling` periods
        instead of a single period. A window with a period without days is NaN. Defaults to
        None.
        scale (float, optional): A multiplier applied to the result for readability.
        Defaults to 1,000,000.

    Returns:
        pd.Series | pd.DataFrame: The Roll Impact with the periods as the index (and the
        assets as columns for a DataFrame).

    Raises:
        TypeError: If the prices are not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(close_prices, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_roll_impact(
                    close_prices[column], dollar_volume[column], period, rolling, scale
                )
                for column in close_prices.columns
            }
        )
    if not isinstance(close_prices, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    data = pd.concat([close_prices, dollar_volume], axis=1)
    data = data[(data.iloc[:, 0] > 0) & (data.iloc[:, 1] >= 0)]
    periods = to_period_index(data.index).asfreq(PERIOD_TRANSLATION[period])

    if len(data) <= MINIMUM_ROLL_OBSERVATIONS:
        return pd.Series(dtype=float, name=close_prices.name)

    change = data.iloc[:, 0].diff().to_numpy()
    previous, current = change[1:-1], change[2:]
    # A pair is two consecutive changes of the prices within the period (the three days fall within it), as the Roll Spread of a period in `get_roll_spread` takes the changes of the period's prices.  # noqa: E501
    same_period = np.asarray(periods[2:] == periods[:-2])

    pair_sums = sum_per_period(
        pd.DataFrame(
            {
                "Previous": previous,
                "Current": current,
                "Product": previous * current,
                "Count": 1.0,
            },
            index=periods[2:],
        )[same_period],
        rolling,
    )
    day_sums = sum_per_period(
        pd.DataFrame(
            {"Dollar Volume": data.iloc[:, 1].to_numpy(), "Days": 1.0}, index=periods
        ),
        rolling,
    )
    pair_sums = pair_sums.reindex(day_sums.index)

    count = pair_sums["Count"]
    autocovariance = (
        pair_sums["Product"] - pair_sums["Previous"] * pair_sums["Current"] / count
    ) / (count - 1)
    estimable = (count >= MINIMUM_ROLL_OBSERVATIONS) & (autocovariance < 0)
    roll_spread = 2 * np.sqrt(-autocovariance.where(estimable))
    mean_dollar_volume = day_sums["Dollar Volume"] / day_sums["Days"]

    return (
        roll_spread / mean_dollar_volume.where(mean_dollar_volume > 0) * scale
    ).rename(close_prices.name)


def get_bulk_volume_classification(
    prices: pd.Series | pd.DataFrame,
    volume: pd.Series | pd.DataFrame,
    window: int = VPIN_BUCKETS,
    degrees_of_freedom: float | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the buy volume of each bar with the Bulk Volume Classification of Easley,
    López de Prado and O'Hara (2012): the share of the volume that was buying, from the price change.

    Rather than signing every trade, the Bulk Volume Classification splits the volume of a
    bar into buys and sells by how large the price change of the bar is relative to the
    usual price change: a bar with a large rise is mostly buying, one without a change is
    half buying and half selling:

    - ΔP_t = P_t - P_(t-1)
    - V^B_t = V_t * Z(ΔP_t / σ_ΔP)
    - V^S_t = V_t - V^B_t

    Where Z is the cumulative distribution function of the standard normal distribution (or
    of the Student-t distribution with `degrees_of_freedom`, as in Easley, López de Prado and
    O'Hara (2016)), and σ_ΔP the sample standard deviation (with n - 1 in the denominator) of the price changes of
    the last `window` bars, the bar itself included. The papers classify one-minute or other
    intraday bars; on daily bars each day is classified as a whole.

    For more information about the method, see the following papers:

    - Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a
    High-Frequency World." The Review of Financial Studies, 25(5), 1457-1493.
    - Easley, D., López de Prado, M., & O'Hara, M. (2016). "Discerning Information from Trade
    Data." Journal of Financial Economics, 120(2), 269-285.

    Args:
        prices (pd.Series | pd.DataFrame): A Series or Dataframe of the prices of the bars.
        volume (pd.Series | pd.DataFrame): The Volume of the bars on the same dates.
        window (int, optional): The number of bars the standard deviation of the price changes
        is taken over. Defaults to 50.
        degrees_of_freedom (float | None, optional): The degrees of freedom of a Student-t
        distribution for Z instead of the standard normal distribution. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The buy volume of each bar, NaN before the first full
        window.

    Raises:
        TypeError: If the prices are not a pd.Series or pd.DataFrame.
    """
    if isinstance(prices, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_bulk_volume_classification(
                    prices[column], volume[column], window, degrees_of_freedom
                )
                for column in prices.columns
            }
        )
    if not isinstance(prices, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    data = pd.concat([prices, volume], axis=1)
    data = data[(data.iloc[:, 0] > 0) & (data.iloc[:, 1] >= 0)]
    change = data.iloc[:, 0].diff()
    deviation = change.rolling(window).std()
    standardized = (change / deviation.where(deviation > 0)).to_numpy()

    probability = (
        stats.norm.cdf(standardized)
        if degrees_of_freedom is None
        else stats.t.cdf(standardized, degrees_of_freedom)
    )

    return (data.iloc[:, 1] * probability).rename(prices.name)


def get_vpin(
    prices: pd.Series | pd.DataFrame,
    volume: pd.Series | pd.DataFrame,
    buckets: int = VPIN_BUCKETS,
    bucket_volume: float | None = None,
    window: int = VPIN_BUCKETS,
    degrees_of_freedom: float | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Volume-Synchronized Probability of Informed Trading (VPIN) of Easley, López de
    Prado and O'Hara (2012) on daily bars: how one-sided, and thus toxic, the order flow is.

    VPIN measures the toxicity of the order flow: the share of the volume that is one-sided,
    as informed traders buy (or sell) persistently while uninformed trading roughly
    balances. The volume is split into buckets of the same volume V (a volume clock rather
    than a time clock), the volume of each bucket is classified into buys and sells with the
    Bulk Volume Classification and VPIN is the mean order imbalance over the last n buckets:

    - V^B_τ = Σ V_t * Z(ΔP_t / σ_ΔP) over the volume in bucket τ
    - V^S_τ = V - V^B_τ
    - VPIN = Σ |V^S_τ - V^B_τ| / (n * V) over the last n buckets

    A day that fills more than the rest of a bucket carries its excess volume into the next
    bucket, as in the paper, with its own buy share, so every bucket holds exactly V. VPIN
    is updated when a bucket completes; on a day without a completed bucket it is that of
    the last one, and before n buckets are complete it is NaN. It lies between 0 and 1.

    VPIN was designed for intraday data: the paper fills 50 buckets a day from one-minute
    bars, so it measures the toxicity within a day. On daily bars the finest classification
    is a whole day, which nets the buying and selling within it, so this is a coarse
    approximation of the measure: a slow-moving estimate of the one-sidedness of the
    volume over weeks rather than the flow toxicity that warns of an intraday liquidity
    crash. Use it to compare assets and periods, not as the published intraday VPIN.

    For more information about the method, see the following sources:

    - Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a
    High-Frequency World." The Review of Financial Studies, 25(5), 1457-1493.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
    19.5.2.

    The buckets are formed from the days, so each holds the same volume, as in the paper,
    and the mean imbalance of the last n buckets is divided by that volume.

    Shares similar logic:
    - RiskLabAI, volume_synchronized_pin: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/microstructural_features/order_flow.py#L157

    Args:
        prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily prices.
        volume (pd.Series | pd.DataFrame): The daily Volume on the same dates.
        buckets (int, optional): The number of buckets n VPIN is the mean over. Defaults to 50,
        as in the paper.
        bucket_volume (float | None, optional): The volume V of a bucket. Defaults to None,
        which takes the mean daily Volume of each asset over its first `window` days, the
        days before the first one classified, so that a bucket holds about a day of trading,
        VPIN spans about `buckets` days and no value uses the volume of later days. Easley,
        López de Prado and O'Hara (2012) also fix V in advance, at a fiftieth of the average
        daily volume.
        window (int, optional): The number of days the standard deviation of the price
        changes of the classification is taken over. Defaults to 50.
        degrees_of_freedom (float | None, optional): The degrees of freedom of a Student-t
        distribution for the classification instead of the standard normal distribution.
        Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: VPIN per day, with the dates as the index (and the assets
        as columns for a DataFrame).

    Raises:
        TypeError: If the prices are not a pd.Series or pd.DataFrame.
    """
    if isinstance(prices, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_vpin(
                    prices[column],
                    volume[column],
                    buckets,
                    bucket_volume,
                    window,
                    degrees_of_freedom,
                )
                for column in prices.columns
            }
        )
    if not isinstance(prices, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    data = pd.concat([prices, volume], axis=1)
    data = data[(data.iloc[:, 0] > 0) & (data.iloc[:, 1] >= 0)]
    buy_volume = get_bulk_volume_classification(
        data.iloc[:, 0], data.iloc[:, 1], window, degrees_of_freedom
    )
    vpin = pd.Series(np.nan, index=data.index, name=prices.name)
    # The mean Volume of the first `window` days, the days before the first classified one,
    # so the volume clock is fixed in advance (as in the paper) and no VPIN uses later volume.
    size = bucket_volume if bucket_volume else data.iloc[:window, 1].mean()

    # The days that fill the buckets: classified and with volume (a day without volume adds nothing to a bucket).  # noqa: E501
    filled = buy_volume.notna() & (data.iloc[:, 1] > 0)

    if not size > 0 or not filled.any():
        return vpin

    cumulative_volume = np.concatenate([[0], data.iloc[:, 1][filled].cumsum()])
    cumulative_buy_volume = np.concatenate([[0], buy_volume[filled].cumsum()])
    completed = int(cumulative_volume[-1] // size)

    if completed < buckets:
        return vpin

    # The buy share of a day is the same for all of its volume, so the cumulative buy volume is linear within a day and its value at each bucket edge follows by interpolation, splitting a day over two buckets.  # noqa: E501
    edges = size * np.arange(completed + 1)
    bucket_buy_volume = np.diff(
        np.interp(edges, cumulative_volume, cumulative_buy_volume)
    )
    imbalance = np.abs(size - 2 * bucket_buy_volume)
    bucket_vpin = pd.Series(imbalance).rolling(buckets).sum().to_numpy() / (
        buckets * size
    )

    # The day on which each bucket completes (a day that ends exactly on an edge completes it, also after rounding of the sums); on a day that completes several, the last one counts.  # noqa: E501
    completion = (
        np.searchsorted(
            cumulative_volume, edges[1:] - VPIN_EDGE_TOLERANCE * size, side="left"
        )
        - 1
    )
    completion_day = data.index[filled][np.minimum(completion, filled.sum() - 1)]
    bucket_vpin = pd.Series(bucket_vpin, index=completion_day)
    bucket_vpin = bucket_vpin[~bucket_vpin.index.duplicated(keep="last")]

    return (
        bucket_vpin.reindex(data.index)
        .ffill()
        .reindex(prices.index)
        .rename(prices.name)
    )
