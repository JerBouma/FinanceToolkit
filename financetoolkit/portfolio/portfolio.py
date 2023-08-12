"""Portfolio Module"""
__docformat__ = "google"

import numpy as np
import pandas as pd


def get_tracking_error(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculate the tracking error, a measure of the deviation of a portfolio's returns
    from its benchmark.

    Args:
        portfolio_returns (pd.Series): A pandas series of portfolio returns.
        benchmark_returns (pd.Series): A pandas series of benchmark returns.
    Returns:
        pd.Series: A pandas series of tracking error.
    """
    return np.std(portfolio_returns - benchmark_returns)


def get_payoff_ratio(returns: pd.Series) -> pd.Series:
    """
    Calculate the Payoff Ratio, a ratio that measures the ability of a trading strategy
    to generate profits relative to its losses.

    Args:
        returns (pd.Series): A series of returns.

    Returns:
        float | pd.Series: The Payoff Ratio value.
    """
    return np.sum(returns.where(returns > 0, 0)) / -np.sum(
        returns.where(returns < 0, 0)
    )


def get_profit_factor(returns: pd.Series) -> pd.Series:
    """
    Calculate the profit factor, which measures the gross profit of winning trades
    divided by the gross loss of losing trades. A value greater than 1 indicates
    profitability, while a value less than 1 indicates unprofitability.

    Args:
        returns (pd.Series): Series of returns.
        rolling (int, optional): The rolling window size. Default is 0, which
            means that the calculation is not done on a rolling window.

    Returns:
        pd.Series: Series of profit factor values.
    """
    return returns[returns > 0].sum() / abs(returns[returns < 0].sum())


def get_jensens_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: pd.Series = 0.0,
) -> pd.Series:
    """
    Calculates the Jensen's alpha for the given returns series relative to the given benchmark returns series.

    Args:
        returns (pd.Series): A series of asset returns.
        benchmark_returns (pd.Series): A series of benchmark returns.
        risk_free_rate (float or pd.Series): The annual risk-free rate, defaults to 0.0.
        rolling_window (int): The rolling window size, defaults to 0 which means no rolling.

    Returns:
        pd.Series: The calculated Jensen's alpha series.
    """
    excess_returns = returns - (risk_free_rate / 252)
    excess_benchmark_returns = benchmark_returns - (risk_free_rate / 252)

    _, alpha = np.polyfit(excess_benchmark_returns, excess_returns, 1)

    return alpha


def get_gain_to_pain_ratio(returns: pd.Series) -> pd.Series:
    """
    Calculate the gain to pain ratio, a risk-adjusted return ratio that measures the
    return of an investment relative to the drawdown experienced over the same period.

    Args:
        returns (pd.Series): A pandas series of returns.

    Returns:
        pd.Series: A pandas series of gain to pain ratio.
    """
    return returns.sum() / np.abs(returns[returns < 0].sum())


def get_max_drawdown(returns: pd.Series) -> pd.Series:
    """
    The maximum drawdown is a measure of the largest loss from
    a peak to a through of a portfolio, expressed as a percentage.

    Args:
        returns (pd.Series): A pandas series of returns.

    Returns:
        pd.Series: A pandas series of gain to pain ratio.
    """
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    return max_drawdown


def get_tail_ratio(returns: pd.Series) -> pd.Series:
    """
    Calculate the tail ratio, a measure of the ratio of the average of the positive
    returns to the absolute value of the average of the negative returns.

    Args:
        returns (pd.Series): A pandas series of returns.

    Returns:
        pd.Series: A pandas series of tail ratio.
    """
    return np.mean(returns[returns > 0]) / np.abs(np.mean(returns[returns < 0]))


def get_common_sense_ratio(returns: pd.Series) -> pd.Series:
    """
    Calculate the common sense ratio, a risk-adjusted return ratio that measures the
    return of an investment relative to its maximum drawdown.

    Args:
        returns (pd.Series): A pandas series of returns.

    Returns:
        pd.Series: A pandas series of common sense ratio.
    """
    return returns.sum() / np.abs(returns.min())


def get_calmar_ratio(returns: pd.Series) -> pd.Series:
    """
    The Calmar Ratio is a measure of risk-adjusted performance
    of an investment strategy. It is calculated by dividing the average
    annual rate of return by the maximum drawdown over a specified period.

    Args:
        returns (pd.Series): A series of asset returns.

    Returns:
        float: The calculated Calmar ratio series.
    """
    average_return = returns.mean()
    maximum_drawdown = get_max_drawdown(returns)

    if min(returns) > 0:
        raise ValueError(
            "Calmar ratio can not be calculated due to zero negative returns."
        )

    return average_return / maximum_drawdown


def get_kelly_criterion(
    win_probability: pd.Series, win_loss_ratio: pd.Series
) -> pd.Series:
    """
    Calculates the Kelly criterion for the given win and loss probabilities.

    The win/loss ratio rpresents the ratio of the amount won on a winning
    bet to the amount lost on a losing bet.

    The Kelly Criterion is a decimal value between 0 and 1, which
    represents the percentage of your bankroll that should be bet
    on each round.

    Args:
        win_probability (float or pd.Series): A series of win probabilities.
        win_loss_ratio (float or pd.Series): A series with win/loss ratios.

    Returns:
        pd.Series: The calculated Kelly criterion series.
    """
    p = win_probability
    r = win_loss_ratio

    kelly_criterion = (p * r - (1 - p)) / r

    return kelly_criterion
