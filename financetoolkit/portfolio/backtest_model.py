"""Portfolio Backtest Model"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.risk import risk_model
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# pylint: disable=too-many-arguments,too-many-locals

# The period of each rebalancing frequency, whose last trading day is a rebalancing date.
REBALANCING_PERIODS = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}


def get_rebalancing_dates(
    index: pd.PeriodIndex, rebalancing: str, lookback: int
) -> list:
    """
    The rebalancing dates of a walk-forward backtest: the last trading day of every
    month, quarter or year that has at least `lookback` returns up to and including it,
    except the last day of the data, after which there is nothing to hold.

    Args:
        index (pd.PeriodIndex): The trading days of the returns.
        rebalancing (str): "monthly", "quarterly" or "yearly".
        lookback (int): The number of returns each estimation needs.

    Returns:
        list: The rebalancing dates, in order.
    """
    if rebalancing not in REBALANCING_PERIODS:
        raise ValueError(
            f"The rebalancing must be one of {', '.join(REBALANCING_PERIODS)}, not "
            f"'{rebalancing}'."
        )
    period_ends = (
        pd.Series(index, index=index)
        .groupby(index.asfreq(REBALANCING_PERIODS[rebalancing]))
        .tail(1)
    )
    positions = index.get_indexer(period_ends.to_numpy())
    return [
        index[position]
        for position in positions
        if lookback - 1 <= position < len(index) - 1
    ]


def get_walk_forward_backtest(
    returns: pd.DataFrame,
    strategies: dict[str, Callable[[pd.DataFrame], pd.Series]],
    rebalancing_dates: list,
    lookback: int,
    transaction_cost: float = 0.0,
    maximum_turnover: float | None = None,
    holding_returns: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    A walk-forward (out-of-sample) backtest of weighting strategies. On every rebalancing
    date each strategy sets its weights from the `lookback` returns up to and including
    that date, and the portfolio holds them from the next period on, so no estimate uses
    a return it is then measured on. Between rebalancing dates the weights drift with
    the returns of the assets, w_i,t+1 = w_i,t (1 + r_i,t+1) / (1 + r_p,t+1), as a
    portfolio that does not trade does.

    The turnover of a rebalancing is Σ |w_i,new - w_i,drifted|, the weight bought and
    sold, and its cost is the transaction cost times the turnover, taken from the
    portfolio's value on the rebalancing date: the next period's return is
    (1 - c × turnover)(1 + r_p) - 1 (DeMiguel, Garlappi and Uppal, 2009). The first
    investment is not counted, as in the paper, so every strategy starts from the same
    value. With a maximum turnover a strategy trades from its drifted weights towards its
    new weights only as far as the limit allows, w = w_drifted + min(1, τ / turnover)
    (w_new - w_drifted), which keeps the weights summing to one and every strategy
    comparable.

    The estimates use the periods in which every asset has a return. The holdings earn
    the returns of every period, so a day on which one market is closed (a 0% return for
    its assets, the next day covering both) still counts for the others.

    When a strategy finds no weights on a rebalancing date (for example the maximum
    Sharpe ratio portfolio when no asset beat the risk free rate over the window), it
    keeps its drifted weights until the next date, or holds equal weights when that
    happens on the first date.

    See DeMiguel, V., Garlappi, L. and Uppal, R. (2009), "Optimal Versus Naive
    Diversification: How Inefficient Is the 1/N Portfolio Strategy?", The Review of
    Financial Studies, 22(5), 1915-1953.

    Args:
        returns (pd.DataFrame): The returns per period of the assets, without missing
            values.
        strategies (dict[str, Callable[[pd.DataFrame], pd.Series]]): Per strategy name, a
            function that turns the returns of an estimation window into weights.
        rebalancing_dates (list): The dates on which the strategies set new weights, each
            with at least `lookback` returns up to and including it.
        lookback (int): The number of returns in each estimation window.
        transaction_cost (float, optional): The cost per unit of weight traded, e.g.
            0.001 for 10 basis points. Defaults to 0.
        maximum_turnover (float | None, optional): The highest turnover of a rebalancing.
            Defaults to None, no limit.
        holding_returns (pd.DataFrame | None, optional): The returns per period the
            portfolios earn, with the same assets, every date of `returns` and 0 on the
            days an asset did not trade. Defaults to None, which uses `returns`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The returns per period after
        costs from the day after the first rebalancing date, one column per strategy;
        the weights set on each rebalancing date, one column per strategy and asset; and
        the turnover of each rebalancing per strategy (NaN for the first investment).
    """
    if not rebalancing_dates:
        raise ValueError(
            "No rebalancing date has enough returns before it; use a longer date range "
            "or a shorter lookback."
        )
    if transaction_cost < 0:
        raise ValueError("The transaction cost cannot be negative.")
    if maximum_turnover is not None and maximum_turnover < 0:
        raise ValueError("The maximum turnover cannot be negative.")

    holding_returns = returns if holding_returns is None else holding_returns
    data = holding_returns[returns.columns].to_numpy(dtype=float)
    assets = returns.columns
    windows = [returns.index.get_loc(date) for date in rebalancing_dates]
    if windows[0] < lookback - 1:
        raise ValueError(
            "The first rebalancing date has fewer returns than the lookback."
        )
    positions = [holding_returns.index.get_loc(date) for date in rebalancing_dates]
    # Each holding period runs from the day after a rebalancing date up to and including
    # the next one (the last up to the end of the data).
    ends = positions[1:] + [len(data) - 1]

    net_returns, weight_rows, turnover_rows = {}, {}, {}
    for name, strategy in strategies.items():
        path = np.empty(len(data) - positions[0] - 1)
        held = None
        targets, turnovers, failures = [], [], 0
        for window, position, end in zip(windows, positions, ends, strict=True):
            try:
                target = (
                    strategy(returns.iloc[window - lookback + 1 : window + 1])
                    .reindex(assets)
                    .fillna(0)
                    .to_numpy(dtype=float)
                )
                if not np.all(np.isfinite(target)):
                    raise ValueError("The weights are not finite.")
            except ValueError:
                failures += 1
                target = (
                    held if held is not None else np.full(len(assets), 1 / len(assets))
                )

            if held is None:
                weights, turnover = target, np.nan
            else:
                trade = target - held
                turnover = float(np.abs(trade).sum())
                if maximum_turnover is not None and turnover > maximum_turnover:
                    trade *= maximum_turnover / turnover
                    turnover = maximum_turnover
                weights = held + trade
            targets.append(weights)
            turnovers.append(turnover)

            # The value of each position relative to the portfolio's value on the
            # rebalancing date, compounded over the holding period.
            growth = np.cumprod(1 + data[position + 1 : end + 1], axis=0) * weights
            value = growth.sum(axis=1)
            period_returns = np.diff(np.concatenate([[1.0], value])) / np.concatenate(
                [[1.0], value[:-1]]
            )
            if not np.isnan(turnover):
                period_returns[0] = (1 - transaction_cost * turnover) * (
                    1 + period_returns[0]
                ) - 1
            start = position - positions[0]
            path[start : start + len(period_returns)] = period_returns
            held = growth[-1] / value[-1]

        if failures:
            logger.warning(
                "%s found no weights on %d of the %d rebalancing dates and kept its "
                "weights on those dates.",
                name,
                failures,
                len(positions),
            )
        net_returns[name] = path
        weight_rows[name] = pd.DataFrame(
            targets, index=rebalancing_dates, columns=assets
        )
        turnover_rows[name] = turnovers

    index = holding_returns.index[positions[0] + 1 :]
    weights_table = pd.concat(weight_rows, axis=1)
    weights_table.index.name = "Date"
    turnover_table = pd.DataFrame(turnover_rows, index=rebalancing_dates)
    turnover_table.index.name = "Date"

    return pd.DataFrame(net_returns, index=index), weights_table, turnover_table


def get_backtest_statistics(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """
    The performance of the strategies of a backtest: the annualized (geometric) return,
    the annualized volatility (ddof=1), the Sharpe ratio (the annualized average return
    above the risk free rate over the annualized volatility), the maximum drawdown of
    the compounded returns, the average turnover per rebalancing (DeMiguel, Garlappi and
    Uppal, 2009) and the transaction costs per year.

    Each strategy is measured over the periods in which it has a return, so a benchmark
    that misses some days does not shorten the others.

    Args:
        returns (pd.DataFrame): The returns per period after costs, one column per
            strategy.
        turnover (pd.DataFrame): The turnover of each rebalancing per strategy, NaN for
            the first investment; a strategy without turnover (a benchmark) has none.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.
        risk_free_rate (float, optional): The average annualized risk free rate over the
            periods. Defaults to 0.
        transaction_cost (float, optional): The cost per unit of weight traded. Defaults
            to 0.

    Returns:
        pd.DataFrame: Per strategy the Annualized Return, Annualized Volatility, Sharpe
        Ratio, Maximum Drawdown, Average Turnover and Annualized Costs.
    """
    years = returns.count() / periods_per_year
    annualized_return = (1 + returns).prod(min_count=1) ** (1 / years) - 1
    volatility = returns.std() * np.sqrt(periods_per_year)
    average_turnover = turnover.reindex(columns=returns.columns).mean()

    return pd.DataFrame(
        {
            "Annualized Return": annualized_return,
            "Annualized Volatility": volatility,
            "Sharpe Ratio": (returns.mean() * periods_per_year - risk_free_rate)
            / volatility,
            "Maximum Drawdown": risk_model.get_max_drawdown(returns),
            "Average Turnover": average_turnover,
            "Annualized Costs": turnover.reindex(columns=returns.columns).sum(
                min_count=1
            )
            * transaction_cost
            / years,
        }
    )


def get_in_sample_expectations(weights: pd.DataFrame, estimates: dict) -> pd.DataFrame:
    """
    What the weights of a walk-forward backtest promised when they were set: on every
    rebalancing date the expected return and Sharpe ratio of the weights by the estimates
    of their own estimation window, averaged over the dates. Next to the realised (out of
    sample) return of the backtest, it shows how much of an optimizer's promise held.

    The formula is as follows:

    - In-Sample Return = average over the dates t of w(t) · mu(t)
    - In-Sample Sharpe Ratio = average over the dates t of (w(t) · mu(t) - rf(t)) /
      sqrt(w(t)' Sigma(t) w(t))

    Where w(t) are the weights set on date t and mu(t), Sigma(t) and rf(t) the annualized
    expected returns, covariance matrix and risk free rate estimated from the window up to
    and including t, with the estimators of the backtest.

    Args:
        weights (pd.DataFrame): The weights set on each rebalancing date, one column per
            strategy and asset (see get_walk_forward_backtest).
        estimates (dict): Per rebalancing date the risk free rate, the expected returns
            (pd.Series) and the covariance matrix (pd.DataFrame) of its window.

    Returns:
        pd.DataFrame: Per strategy the In-Sample Return and In-Sample Sharpe Ratio.
    """
    expectations = {}

    for strategy in weights.columns.get_level_values(0).unique():
        returns, sharpe_ratios = [], []

        for date, row in weights[strategy].iterrows():
            risk_free_rate, mu, sigma = estimates[date]
            weight = row.reindex(mu.index).fillna(0).to_numpy(dtype=float)
            expected = float(weight @ mu.to_numpy(dtype=float))
            variance = float(weight @ sigma.to_numpy(dtype=float) @ weight)
            returns.append(expected)
            sharpe_ratios.append(
                (expected - risk_free_rate) / np.sqrt(variance)
                if variance > 0
                else np.nan
            )

        expectations[strategy] = {
            "In-Sample Return": np.nanmean(returns),
            "In-Sample Sharpe Ratio": np.nanmean(sharpe_ratios),
        }

    return pd.DataFrame(expectations).T
