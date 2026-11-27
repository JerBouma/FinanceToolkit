"""Portfolio Backtest Model"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.portfolio.construction.optimization import optimization_model
from financetoolkit.portfolio.construction.optimization.optimization_model import (
    ZERO_VOLATILITY_TOLERANCE,
)
from financetoolkit.risk import risk_model
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# pylint: disable=too-many-arguments,too-many-locals

# The period of each rebalancing frequency, whose last trading day is a rebalancing date.
REBALANCING_PERIODS = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}


def get_rebalancing_dates(
    index: pd.PeriodIndex, rebalancing: str, lookback: int
) -> list:
    """
    The rebalancing dates of a walk-forward backtest.

    They fall on the last trading day of every month, quarter or year that has at least
    `lookback` returns up to and including it, except the last day of the data, after which
    there is nothing to hold.

    Args:
        index (pd.PeriodIndex): The trading days of the returns.
        rebalancing (str): How often the weights are set anew. Options are: "monthly" (the last
            trading day of every month), "quarterly" (of every quarter), "yearly" (of every year).
        lookback (int): The number of returns each estimation needs.

    Returns:
        list: The rebalancing dates, in order.
    """
    check_option("rebalancing", rebalancing, REBALANCING_PERIODS)
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
    A walk-forward (out-of-sample) backtest of weighting strategies: how each would have
    performed with weights set from only the returns known at the time.

    On every rebalancing
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
    keeps its drifted weights until the next date. Before the first date on which it
    finds weights it holds nothing and has no returns (NaN), so it starts later than the
    others rather than with weights its constraints may not allow, and a strategy that
    finds no weights on any date has no returns at all.

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
        costs from the day after the first rebalancing date, one column per strategy
        (NaN before its first weights); the weights set on each rebalancing date, one
        column per strategy and asset (NaN before its first weights); and the turnover of
        each rebalancing per strategy (NaN for the first investment).
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
        path = np.full(len(data) - positions[0] - 1, np.nan)
        held = None
        targets, turnovers, failures, reason = [], [], 0, ""
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
            except InvalidArgumentError:
                # A mistake in the call fails every date alike, so it is not a failed date.
                raise
            except ValueError as error:
                failures += 1
                reason = reason or str(error)
                target = held

            if target is None:
                # Nothing is held before the first weights, so there is no return yet.
                targets.append(np.full(len(assets), np.nan))
                turnovers.append(np.nan)
                continue
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

        if failures == len(positions):
            logger.warning(
                "%s found no weights on any of the %d rebalancing dates (%s), so it has "
                "no returns.",
                name,
                failures,
                reason,
            )
        elif failures:
            logger.warning(
                "%s found no weights on %d of the %d rebalancing dates (%s); it starts "
                "with its first weights, on %s, and keeps its weights on the later dates "
                "without new ones.",
                name,
                failures,
                len(positions),
                reason,
                next(
                    date
                    for date, target in zip(rebalancing_dates, targets, strict=True)
                    if not np.isnan(target).all()
                ),
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
    periods_per_year: float | pd.Series,
    risk_free_rate: float = 0.0,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """
    The performance of the strategies of a backtest.

    These are the annualized (geometric) return, the annualized volatility (ddof=1), the
    Sharpe ratio (the annualized average return above the risk free rate over the annualized
    volatility), the maximum drawdown of the compounded returns, the average turnover per
    rebalancing (DeMiguel, Garlappi and Uppal, 2009) and the transaction costs per year.
    Each strategy is measured over the periods in which it has a return, so a benchmark that
    misses some days does not shorten the others.

    Args:
        returns (pd.DataFrame): The returns per period after costs, one column per
            strategy.
        turnover (pd.DataFrame): The turnover of each rebalancing per strategy, NaN for
            the first investment; a strategy without turnover (a benchmark) has none.
        periods_per_year (float | pd.Series): The number of periods in a year, e.g. 252
            for daily data, or per strategy (365 for a strategy with returns on every
            calendar day).
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
            # A strategy without volatility (all cash) has no Sharpe ratio, not ±inf.
            "Sharpe Ratio": (returns.mean() * periods_per_year - risk_free_rate)
            / volatility.where(volatility > ZERO_VOLATILITY_TOLERANCE),
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
    What the weights of a walk-forward backtest promised when they were set.

    On every rebalancing date the expected return and Sharpe ratio of the weights by the
    estimates of their own estimation window, averaged over the dates. Next to the realised
    (out of sample) return of the backtest, it shows how much of an optimizer's promise
    held.

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
            # A strategy holds nothing before its first weights.
            if row.isna().all():
                continue
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
            "In-Sample Return": np.nanmean(returns) if returns else np.nan,
            "In-Sample Sharpe Ratio": (
                np.nanmean(sharpe_ratios) if sharpe_ratios else np.nan
            ),
        }

    return pd.DataFrame(expectations).T


def get_window_estimates(
    window: pd.DataFrame,
    estimates: dict,
    risk_free_rate: Callable[[pd.Index], float],
    periods_per_year: float,
    covariance: str,
    covariance_span: int,
    expected_returns: str,
    expected_returns_span: int,
    benchmark: pd.Series | None,
    factors: pd.DataFrame | None,
) -> tuple:
    """
    Estimate the risk free rate, expected returns and covariance matrix of an estimation
    window of a walk-forward backtest. The estimates of a window are the same for every
    method, so they are made once and kept in estimates per last date of the window.

    Args:
        window (pd.DataFrame): The returns of the estimation window, one column per asset.
        estimates (dict): The estimates made so far per last date of their window, to which
            this window's are added.
        risk_free_rate (Callable[[pd.Index], float]): Gives the annualized risk free rate over
            the dates of a window.
        periods_per_year (float): The number of return periods in a year.
        covariance (str): The covariance estimator.
        covariance_span (int): The span of an exponentially weighted covariance.
        expected_returns (str): The expected returns estimator.
        expected_returns_span (int): The span of exponentially weighted expected returns.
        benchmark (pd.Series | None): The benchmark returns, for the CAPM expected returns and
            the single factor covariance.
        factors (pd.DataFrame | None): The factor returns, for the factor covariance.

    Returns:
        tuple: The annualized risk free rate, the expected returns (pd.Series) and the
        covariance matrix (pd.DataFrame) of the window.
    """
    if window.index[-1] not in estimates:
        rate = risk_free_rate(window.index)
        estimates[window.index[-1]] = (
            rate,
            *optimization_model.get_expected_returns_and_covariance(
                window,
                periods_per_year,
                covariance,
                covariance_span,
                expected_returns,
                expected_returns_span,
                benchmark,
                rate,
                factors,
            ),
        )
    return estimates[window.index[-1]]


def get_strategy_weights(
    window: pd.DataFrame,
    method: str,
    generator: np.random.Generator,
    estimate: Callable[[pd.DataFrame], tuple],
    periods_per_year: float,
    minimum_weight: float,
    maximum_weight: float,
) -> pd.Series:
    """
    Set the weights of one weighting method of a walk-forward backtest from the returns of
    its estimation window.

    Args:
        window (pd.DataFrame): The returns of the estimation window, one column per asset.
        method (str): The weighting method, such as "equal_weight", "random",
            "minimum_variance" or "maximum_sharpe"; any other method sets mean-variance
            weights.
        generator (np.random.Generator): The random generator of the "random" method, one
            per backtest, drawing a new portfolio on every date.
        estimate (Callable[[pd.DataFrame], tuple]): Gives the annualized risk free rate,
            expected returns and covariance matrix of a window (see get_window_estimates).
        periods_per_year (float): The number of return periods in a year.
        minimum_weight (float): The minimum weight of an asset.
        maximum_weight (float): The maximum weight of an asset.

    Returns:
        pd.Series: The weight per asset.
    """
    if method == "equal_weight":
        return pd.Series(1 / window.shape[1], index=window.columns)
    if method == "random":
        return optimization_model.get_random_weights(window.columns, generator)
    if method in ("minimum_mad", "minimum_semideviation"):
        return optimization_model.get_mean_risk_weights(
            window,
            risk_measure=method.removeprefix("minimum_"),
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
    if method == "minimum_cvar":
        return optimization_model.get_minimum_cvar_weights(
            window,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
    if method == "minimum_drawdown":
        return optimization_model.get_minimum_drawdown_weights(
            window,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
    risk_free_rate, mu, sigma = estimate(window)
    if method == "maximum_omega":
        return optimization_model.get_maximum_omega_weights(
            window,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
            expected_returns=mu / periods_per_year,
        )
    if method == "maximum_cvar_ratio":
        return optimization_model.get_maximum_cvar_ratio_weights(
            window,
            risk_free_rate=risk_free_rate / periods_per_year,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
            expected_returns=mu / periods_per_year,
        )
    if method == "hierarchical_equal_risk_contribution":
        return optimization_model.get_hierarchical_equal_risk_contribution_weights(
            sigma,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
    if method == "nested_clustered_optimization":
        return optimization_model.get_nested_clustered_optimization_weights(
            mu, sigma, risk_free_rate=risk_free_rate
        )
    if method == "inverse_volatility":
        return optimization_model.get_inverse_volatility_weights(sigma)
    if method == "hierarchical_risk_parity":
        return optimization_model.get_hierarchical_risk_parity_weights(
            sigma,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
    if method == "risk_parity":
        return optimization_model.get_risk_parity_weights(
            sigma,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
    if method == "minimum_variance":
        return optimization_model.get_minimum_variance_weights(
            sigma, minimum_weight, maximum_weight
        )
    if method == "maximum_diversification":
        return optimization_model.get_maximum_diversification_weights(
            sigma, minimum_weight, maximum_weight
        )
    if method == "maximum_sharpe":
        return optimization_model.get_maximum_sharpe_weights(
            mu, sigma, risk_free_rate, minimum_weight, maximum_weight
        )
    return optimization_model.get_mean_variance_weights(
        mu,
        sigma,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )
