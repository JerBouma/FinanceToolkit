"""Bet Sizing Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import optimize, stats

from financetoolkit.utilities.error_model import InvalidArgumentError

# The fits that agree to this many decimals (on the standardized values) are one root of
# the moment equations; the root found from the most starting points is the most likely one.
EF3M_ROOT_DECIMALS = 6


def get_probability_bet_size(
    probabilities: pd.Series,
    predictions: pd.Series | None = None,
    sides: pd.Series | None = None,
    classes: int = 2,
) -> pd.Series:
    """
    Calculate the bet size from the probability a classifier gives its prediction: how far the probability lies
    above that of a random guess, turned into a bet between -1 and 1.

    The formula is as follows:

    - z = (p - 1 / K) / SQRT(p x (1 - p))
    - m = prediction x side x (2 x N(z) - 1)

    Where p is the probability of the predicted outcome, K the number of classes and N the
    standard normal distribution function. A probability of 1 / K gives no bet and a certain
    prediction a full bet of 1. With meta-labels the prediction is 1 (act on the side) or 0 (do
    not), the side that of the primary model.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.3
      (Snippet 10.1).

    Referenced against:
    - RiskLabAI, probability_bet_size: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/bet_sizing.py#L33

    Args:
        probabilities (pd.Series): The probability of the predicted outcome of each bet.
        predictions (pd.Series | None, optional): The predicted outcome of each bet, such as 1 and
            -1, or 1 and 0 for meta-labels. Defaults to None, which sizes the bets without a sign.
        sides (pd.Series | None, optional): The side of each bet, 1 (long) or -1 (short), for
            meta-labels. Defaults to None.
        classes (int, optional): The number of classes the classifier chooses from. Defaults to 2.

    Returns:
        pd.Series: The bet size of each bet, between -1 and 1.
    """
    if classes < 2:  # noqa: PLR2004
        raise ValueError("classes needs to be 2 or more.")

    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = (probabilities - 1 / classes) / np.sqrt(
            probabilities * (1 - probabilities)
        )
    bet_sizes = pd.Series(
        2 * stats.norm.cdf(statistic.to_numpy(dtype=float)) - 1,
        index=probabilities.index,
    ).where(probabilities.notna())

    if predictions is not None:
        bet_sizes = bet_sizes * predictions
    if sides is not None:
        bet_sizes = bet_sizes * sides

    return bet_sizes


def _get_bet_spans(holding_periods: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """
    The positions of the bets and of the dates they close, from the holding periods.

    Args:
        holding_periods (pd.Series): The holding period of the bet placed at each date in
            periods, NaN for dates without a bet.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: The start and end positions of the bets and the number
        of dates. A bet is open from its start up to, but not including, its end.
    """
    count = len(holding_periods)
    periods = holding_periods.to_numpy(dtype=float)
    starts = np.flatnonzero(np.isfinite(periods) & (periods > 0))
    ends = np.minimum(starts + periods[starts].astype(int), count)

    return starts, ends, count


def _sum_over_open_bets(
    starts: np.ndarray, ends: np.ndarray, values: np.ndarray, count: int
) -> np.ndarray:
    """
    The sum of `values` over the bets open on each date, a bet being open from its start up to,
    but not including, its end.

    Args:
        starts (np.ndarray): The start positions of the bets.
        ends (np.ndarray): The end positions of the bets.
        values (np.ndarray): The value of each bet.
        count (int): The number of dates.

    Returns:
        np.ndarray: The sum per date.
    """
    changes = np.zeros(count + 1)
    np.add.at(changes, starts, values)
    np.add.at(changes, ends, -values)

    return np.cumsum(changes[:count])


def get_average_active_bet_size(
    bet_sizes: pd.Series, holding_periods: pd.Series
) -> pd.Series:
    """
    Calculate the average active bet size: the average of the bets still open on each date, so a new bet adds to the
    position of the bets before it instead of replacing it.

    The formula is as follows:

    - m_t = the average of m_i over the bets i with t0_i <= t < t1_i, 0 without an open bet

    Where t0_i is the date bet i is placed and t1_i the date it closes, `Holding Period` periods
    later.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.4
      (Snippet 10.2).

    Referenced against:
    - RiskLabAI, avg_active_signals: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/bet_sizing.py#L166

    Args:
        bet_sizes (pd.Series): The bet size of the bet placed at each date, NaN for dates without
            a bet.
        holding_periods (pd.Series): The holding period of the bet placed at each date in
            periods, on the same dates.

    Returns:
        pd.Series: The average bet size of the open bets per date.
    """
    sizes = bet_sizes.reindex(holding_periods.index).to_numpy(dtype=float)
    starts, ends, count = _get_bet_spans(holding_periods.where(np.isfinite(sizes)))

    totals = _sum_over_open_bets(starts, ends, sizes[starts], count)
    open_bets = _sum_over_open_bets(starts, ends, np.ones(len(starts)), count)
    with np.errstate(invalid="ignore", divide="ignore"):
        average = np.where(open_bets > 0, totals / open_bets, 0.0)

    return pd.Series(average, index=holding_periods.index)


def get_discrete_bet_size(bet_sizes: pd.Series, step_size: float) -> pd.Series:
    """
    Calculate the discrete bet size: the bet size rounded to a multiple of a step and capped at -1 and 1, so small
    changes of the prediction do not trade.

    The formula is as follows:

    - m* = max(-1, min(1, round(m / step_size) x step_size))

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.5
      (Snippet 10.3).

    Args:
        bet_sizes (pd.Series): The bet sizes.
        step_size (float): The step of the bet sizes, between 0 and 1.

    Returns:
        pd.Series: The rounded bet sizes.
    """
    if not 0 < step_size <= 1:
        raise ValueError("step_size needs to be above 0 and at most 1.")

    return (np.round(bet_sizes / step_size) * step_size).clip(-1, 1)


def get_hit_rate(
    labels: pd.Series, holding_periods: pd.Series, minimum_labels: int = 20
) -> pd.Series:
    """
    Calculate the hit rate of meta-labels known on each date: the share of the bets closed on or before the date
    whose meta-label is 1 (the bet made money).

    It uses only the bets that have closed, so it holds no information from after the date.

    The formula is as follows:

    - Hit Rate_t = SUM_i 1[t1_i <= t] x Label_i / SUM_i 1[t1_i <= t]

    Args:
        labels (pd.Series): The meta-label of the bet placed at each date, 1 or 0, NaN for dates
            without a bet.
        holding_periods (pd.Series): The holding period of the bet placed at each date in
            periods, on the same dates.
        minimum_labels (int, optional): The number of closed bets the hit rate needs. Defaults
            to 20.

    Returns:
        pd.Series: The hit rate per date, NaN before `minimum_labels` bets have closed.
    """
    outcomes = labels.reindex(holding_periods.index).to_numpy(dtype=float)
    starts, ends, count = _get_bet_spans(holding_periods.where(np.isfinite(outcomes)))
    closed = ends < count

    wins = np.zeros(count)
    np.add.at(wins, ends[closed], outcomes[starts][closed])
    bets = np.zeros(count)
    np.add.at(bets, ends[closed], 1)
    wins, bets = np.cumsum(wins), np.cumsum(bets)

    with np.errstate(invalid="ignore", divide="ignore"):
        hit_rate = np.where(bets >= max(minimum_labels, 1), wins / bets, np.nan)

    return pd.Series(hit_rate, index=holding_periods.index)


def get_concurrent_sides(holding_periods: pd.Series, sides: pd.Series) -> pd.DataFrame:
    """
    Calculate the concurrent sides of the bets: the number of long and short bets open on each date.

    The formula is as follows:

    - Long Bets_t = SUM_i 1[t0_i <= t < t1_i] x 1[side_i > 0]
    - Short Bets_t = SUM_i 1[t0_i <= t < t1_i] x 1[side_i < 0]

    Where t0_i is the date bet i is placed and t1_i the date it closes, `Holding Period` periods
    later: a bet counts from the date it is placed up to, but not including, the date it closes.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.2.

    Args:
        holding_periods (pd.Series): The holding period of the bet placed at each date in
            periods, NaN for dates without a bet.
        sides (pd.Series): The side of the bet placed at each date, 1 (long) or -1 (short).

    Returns:
        pd.DataFrame: The Long Bets and Short Bets per date.
    """
    side = sides.reindex(holding_periods.index).to_numpy(dtype=float)
    starts, ends, count = _get_bet_spans(holding_periods.where(np.isfinite(side)))

    return pd.DataFrame(
        {
            "Long Bets": _sum_over_open_bets(
                starts, ends, (side[starts] > 0).astype(float), count
            ),
            "Short Bets": _sum_over_open_bets(
                starts, ends, (side[starts] < 0).astype(float), count
            ),
        },
        index=holding_periods.index,
    )


def get_budget_bet_size(holding_periods: pd.Series, sides: pd.Series) -> pd.DataFrame:
    """
    Calculate the budget bet size: the open long bets as a share of the most long bets ever open, minus the same
    share of the short bets.

    It spreads a budget over the bets the strategy usually has open, without a model of how likely each bet is to
    win.

    The formula is as follows:

    - m_t = Long Bets_t / max(Long Bets) - Short Bets_t / max(Short Bets)

    Where the maximum is taken over the sample (a side without bets adds 0).

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.2.

    Args:
        holding_periods (pd.Series): The holding period of the bet placed at each date in
            periods, NaN for dates without a bet.
        sides (pd.Series): The side of the bet placed at each date, 1 (long) or -1 (short).

    Returns:
        pd.DataFrame: The Long Bets, Short Bets and Bet Size per date.
    """
    concurrent = get_concurrent_sides(holding_periods, sides)

    bet_size = pd.Series(0.0, index=concurrent.index)
    for column, sign in (("Long Bets", 1), ("Short Bets", -1)):
        most = concurrent[column].max()
        if most > 0:
            bet_size += sign * concurrent[column] / most
    concurrent["Bet Size"] = bet_size

    return concurrent


def _get_mixture_moments(parameters: np.ndarray) -> np.ndarray:
    """
    The first five raw moments of a mixture of two normal distributions.

    Args:
        parameters (np.ndarray): The parameters [mu_1, mu_2, sigma_1, sigma_2, p_1] in the last
            axis.

    Returns:
        np.ndarray: The raw moments 1 to 5 in the last axis.
    """
    mu_1, mu_2, sigma_1, sigma_2, p_1 = np.moveaxis(parameters, -1, 0)

    def normal_moments(mu: np.ndarray, sigma: np.ndarray) -> list[np.ndarray]:
        variance = sigma**2
        return [
            mu,
            mu**2 + variance,
            mu**3 + 3 * mu * variance,
            mu**4 + 6 * mu**2 * variance + 3 * variance**2,
            mu**5 + 10 * mu**3 * variance + 15 * mu * variance**2,
        ]

    return np.stack(
        [
            p_1 * first + (1 - p_1) * second
            for first, second in zip(
                normal_moments(mu_1, sigma_1), normal_moments(mu_2, sigma_2)
            )
        ],
        axis=-1,
    )


def _get_three_moment_fit(
    moments: np.ndarray, p_1: np.ndarray, mu_2: np.ndarray
) -> np.ndarray:
    """
    The exact fit of the first three moments of EF3M (López de Prado and Foreman, 2014): for a
    mixing probability p_1 and a mean mu_2 of the second distribution, the first moment gives
    mu_1 and the second and third moments sigma_2 and sigma_1.

    Args:
        moments (np.ndarray): The raw moments 1 to 5 of the values.
        p_1 (np.ndarray): The mixing probability of every starting point.
        mu_2 (np.ndarray): The mean of the second distribution of every starting point.

    Returns:
        np.ndarray: The parameters [mu_1, mu_2, sigma_1, sigma_2, p_1] per starting point, NaN
        where the first three moments give a negative variance.
    """
    m_1, m_2, m_3 = moments[:3]
    p_2 = 1 - p_1
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_1 = (m_1 - p_2 * mu_2) / p_1
        variance_2 = (
            m_3 - 3 * mu_1 * m_2 + 2 * p_1 * mu_1**3
            + 3 * mu_1 * p_2 * mu_2**2 - p_2 * mu_2**3
        ) / (3 * p_2 * (mu_2 - mu_1))  # fmt: skip
        variance_1 = (m_2 - p_2 * (mu_2**2 + variance_2)) / p_1 - mu_1**2
        valid = (variance_1 > 0) & (variance_2 > 0)
        parameters = np.stack(
            [mu_1, mu_2, np.sqrt(variance_1), np.sqrt(variance_2), p_1], axis=-1
        )
    parameters[~valid] = np.nan

    return parameters


def get_mixture_of_two_gaussians(
    values: pd.Series | np.ndarray, runs: int = 100, seed: int = 0
) -> pd.Series:
    """
    Fit a mixture of two normal distributions to values by matching their first five moments, with the EF3M
    algorithm of López de Prado and Foreman.

    EF3M (Exact Fit of the first 3 Moments) starts from many random points: each fit matches the first three moments
    exactly and adjusts the mixing probability and a mean until the fourth and fifth moments match too, and the
    solution most starting points reach is the most likely one.

    The formula is as follows:

    - m_k = p_1 x E[X_1^k] + (1 - p_1) x E[X_2^k], k = 1..5, X_j ~ N(mu_j, sigma_j²)
    - start: p_1 uniform on (0, 1), mu_2 uniform on (m_1, m_1 + 5 x sigma), with m_1 giving mu_1
      and m_2 and m_3 giving sigma_1 and sigma_2 (the exact fit of the first three moments)
    - fit: the five parameters that solve the five moment equations from each start

    EF3M solves the equations of the fourth and fifth moments by a fixed-point iteration (the
    fourth moment gives a new mu_2, the fifth a new p_1). The solution is often a repelling
    point of that iteration, which then moves away from it whatever the start and ends wherever
    the iterations stop, so a fit depends on the random starts. Here the same five equations are
    solved by Powell's hybrid method from the same starting points, which reaches the solution
    itself: the fits of different seeds agree to the precision of the computer. The values are
    standardized before the fit and the parameters scaled back, which leaves the solution
    unchanged (the moment equations follow a shift and scale of the values) but keeps the five
    moments of comparable size. The distribution of the higher mean comes first, so a solution
    and its mirror image are one.

    For more information, see the following paper and book:

    - López de Prado, M. and Foreman, M. (2014). "A mixture of two Gaussians approach to
      mathematical portfolio oversight: the EF3M algorithm." Quantitative Finance, 14(5),
      913-930.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.2.

    Args:
        values (pd.Series | np.ndarray): The values to fit.
        runs (int, optional): The number of random starting points. Defaults to 100.
        seed (int, optional): The seed of the random starting points, which makes the fit
            reproducible. Defaults to 0.

    Returns:
        pd.Series: The fitted Mu 1, Mu 2, Sigma 1, Sigma 2 and Probability 1 (of the first
        distribution) with the Error, the sum of the squared differences of the five
        standardized moments, and Runs, the share of the starting points that reached the
        solution. NaN when no starting point reaches a solution.
    """
    if runs < 1:
        raise ValueError("runs needs to be 1 or more.")

    names = ["Mu 1", "Mu 2", "Sigma 1", "Sigma 2", "Probability 1", "Error", "Runs"]
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 5 or not data.std() > 0:  # noqa: PLR2004
        return pd.Series(np.nan, index=names)

    mean, deviation = data.mean(), data.std()
    standardized = (data - mean) / deviation
    moments = np.array([np.mean(standardized**order) for order in range(1, 6)])

    def moment_errors(candidate: np.ndarray) -> np.ndarray:
        return _get_mixture_moments(candidate) - moments

    generator = np.random.default_rng(seed)
    starts = _get_three_moment_fit(
        moments, generator.uniform(0, 1, runs), generator.uniform(0, 5, runs)
    )

    solutions = []
    for start in starts[np.isfinite(starts).all(axis=1)]:
        with np.errstate(all="ignore"):
            fit = optimize.root(moment_errors, start, method="hybr")
        mu_1, mu_2, sigma_1, sigma_2, probability = fit.x
        sigma_1, sigma_2 = abs(sigma_1), abs(sigma_2)
        if not fit.success or not 0 < probability < 1 or min(sigma_1, sigma_2) == 0:
            continue
        if mu_2 > mu_1:
            mu_1, mu_2, sigma_1, sigma_2, probability = (
                mu_2, mu_1, sigma_2, sigma_1, 1 - probability
            )  # fmt: skip
        solutions.append([mu_1, mu_2, sigma_1, sigma_2, probability])

    if not solutions:
        return pd.Series(np.nan, index=names)

    solutions_array = np.array(solutions)
    errors = np.sum(moment_errors(solutions_array) ** 2, axis=1)
    keys = np.round(solutions_array, EF3M_ROOT_DECIMALS)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    # The solution most starting points reach; between solutions reached equally often the one
    # with the smallest error.
    best_key = max(
        range(len(counts)), key=lambda key: (counts[key], -errors[inverse == key].min())
    )
    members = np.flatnonzero(inverse == best_key)
    best = solutions_array[members[np.argmin(errors[members])]]

    return pd.Series(
        [
            mean + deviation * best[0],
            mean + deviation * best[1],
            deviation * best[2],
            deviation * best[3],
            best[4],
            errors[members].min(),
            counts[best_key] / runs,
        ],
        index=names,
    )


def get_mixture_cdf(values: np.ndarray | float, parameters: pd.Series) -> np.ndarray:
    """
    Calculate the distribution function of a mixture of two normal distributions: the probability that a value lies
    at or below each point.

    The formula is as follows:

    - F(x) = p_1 x N((x - mu_1) / sigma_1) + (1 - p_1) x N((x - mu_2) / sigma_2)

    Args:
        values (np.ndarray | float): The values to evaluate.
        parameters (pd.Series): The Mu 1, Mu 2, Sigma 1, Sigma 2 and Probability 1 of the
            mixture, such as those of `get_mixture_of_two_gaussians`.

    Returns:
        np.ndarray: The probability of a value at or below each value.
    """
    probability = parameters["Probability 1"]

    return probability * stats.norm.cdf(
        values, parameters["Mu 1"], parameters["Sigma 1"]
    ) + (1 - probability) * stats.norm.cdf(
        values, parameters["Mu 2"], parameters["Sigma 2"]
    )


def get_reserve_bet_size(
    holding_periods: pd.Series, sides: pd.Series, runs: int = 100, seed: int = 0
) -> pd.DataFrame:
    """
    Calculate the reserve bet size: the net number of open bets (long minus short) compared with how many the
    strategy usually has open, through a mixture of two normal distributions fitted to it with the EF3M algorithm.

    An unusually large net long count gives a bet close to 1 and a typical one a small bet, so part of the budget
    stays in reserve for stronger signals.

    The formula is as follows:

    - c_t = Long Bets_t - Short Bets_t
    - F = the mixture of two normal distributions fitted to c_t on the dates bets are placed
    - m_t = (F(c_t) - F(0)) / (1 - F(0)) when c_t >= 0, (F(c_t) - F(0)) / F(0) otherwise

    For more information, see the following book and paper:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.2.
    - López de Prado, M. and Foreman, M. (2014). "A mixture of two Gaussians approach to
      mathematical portfolio oversight: the EF3M algorithm." Quantitative Finance, 14(5),
      913-930.

    Args:
        holding_periods (pd.Series): The holding period of the bet placed at each date in
            periods, NaN for dates without a bet.
        sides (pd.Series): The side of the bet placed at each date, 1 (long) or -1 (short).
        runs (int, optional): The number of random starting points of the EF3M fit. Defaults
            to 100.
        seed (int, optional): The seed of the starting points. Defaults to 0.

    Returns:
        pd.DataFrame: The Long Bets, Short Bets and Bet Size per date.
    """
    concurrent = get_concurrent_sides(holding_periods, sides)
    net = concurrent["Long Bets"] - concurrent["Short Bets"]

    side = sides.reindex(holding_periods.index).to_numpy(dtype=float)
    starts, _, _ = _get_bet_spans(holding_periods.where(np.isfinite(side)))
    parameters = get_mixture_of_two_gaussians(net.iloc[starts], runs=runs, seed=seed)

    if parameters.isna().any():
        concurrent["Bet Size"] = np.nan
        return concurrent

    zero = get_mixture_cdf(0.0, parameters)
    distribution = get_mixture_cdf(net.to_numpy(dtype=float), parameters)
    with np.errstate(divide="ignore", invalid="ignore"):
        concurrent["Bet Size"] = np.where(
            net >= 0, (distribution - zero) / (1 - zero), (distribution - zero) / zero
        )

    return concurrent


def get_sigmoid_width(divergence: float, bet_size: float) -> float:
    """
    Calculate the width of the sigmoid bet size function: the width that gives a chosen bet size at a chosen
    divergence of the forecast from the market price.

    The formula is as follows:

    - w = x² x (1 / m² - 1)

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.6
      (Snippet 10.4).

    Args:
        divergence (float): The divergence x of the forecast from the market price.
        bet_size (float): The bet size m at that divergence, between 0 and 1.

    Returns:
        float: The width w.
    """
    if not 0 < abs(bet_size) < 1:
        raise ValueError("bet_size needs to lie between 0 and 1.")

    return divergence**2 * (bet_size**-2 - 1)


def get_power_width(divergence: float, bet_size: float) -> float:
    """
    Calculate the exponent of the power bet size function: the exponent that gives a chosen bet size at a chosen
    divergence of the forecast from the market price.

    The formula is as follows:

    - w = ln(|m|) / ln(|x|)

    Args:
        divergence (float): The divergence x, between -1 and 1 (not 0).
        bet_size (float): The bet size m at that divergence, between -1 and 1 (not 0).

    Returns:
        float: The exponent w.
    """
    if not 0 < abs(divergence) < 1 or not 0 < abs(bet_size) <= 1:
        raise ValueError(
            "The power function needs a divergence and a bet size between -1 and 1 (not 0)."
        )

    return np.log(abs(bet_size)) / np.log(abs(divergence))


def get_dynamic_bet_size(
    divergence: np.ndarray | pd.Series | float,
    width: float,
    method: str = "sigmoid",
) -> np.ndarray | pd.Series | float:
    """
    Calculate the dynamic bet size: how large a bet the divergence of a forecast from the market price warrants,
    with a sigmoid function (López de Prado) or a power function.

    The formula is as follows:

    - sigmoid: m = x / SQRT(w + x²)
    - power: m = sign(x) x |x|^w, for x between -1 and 1

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.6
      (Snippet 10.4) for the sigmoid function.

    Args:
        divergence (np.ndarray | pd.Series | float): The forecast minus the market price, or for
            the power function a divergence scaled to -1 to 1.
        width (float): The width w of `get_sigmoid_width` or the exponent of `get_power_width`.
        method (str, optional): The function from the divergence to the bet size. Options are:
            "sigmoid" (sigmoid function), "power" (power function). Defaults to "sigmoid".

    Returns:
        np.ndarray | pd.Series | float: The bet size, between -1 and 1.
    """
    if method == "sigmoid":
        return divergence / np.sqrt(width + divergence**2)
    if method == "power":
        if np.any(np.abs(divergence) > 1):
            raise ValueError("The power function needs a divergence between -1 and 1.")
        return np.sign(divergence) * np.abs(divergence) ** width

    raise ValueError("method needs to be 'sigmoid' or 'power'.")


def get_inverse_price(
    forecast_price: float, width: float, bet_size: float, method: str = "sigmoid"
) -> float:
    """
    Calculate the inverse price: the market price at which the bet size function gives a given bet size.

    The formula is as follows:

    - sigmoid: L = f - m x SQRT(w / (1 - m²))
    - power: L = f - sign(m) x |m|^(1 / w)

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.6
      (Snippet 10.4).

    Args:
        forecast_price (float): The forecast price f.
        width (float): The width (sigmoid) or exponent (power) w.
        bet_size (float): The bet size m, between -1 and 1.
        method (str, optional): The function from the divergence to the bet size. Options are:
            "sigmoid" (sigmoid function), "power" (power function). Defaults to "sigmoid".

    Returns:
        float: The market price L.
    """
    if method == "sigmoid":
        with np.errstate(divide="ignore"):
            return forecast_price - bet_size * np.sqrt(width / (1 - bet_size**2))
    if method == "power":
        return forecast_price - np.sign(bet_size) * abs(bet_size) ** (1 / width)

    raise ValueError("method needs to be 'sigmoid' or 'power'.")


def get_limit_price(
    target_position: int,
    current_position: int,
    forecast_price: float,
    width: float,
    maximum_position: int,
    method: str = "sigmoid",
) -> float:
    """
    Calculate the breakeven limit price of an order that moves the position from `current_position` to
    `target_position`.

    It is the average of the market prices at which each unit of the change would be the bet size, so the order does
    not trade at a loss against the forecast.

    The formula is as follows:

    - L = 1 / |q* - q| x SUM_{j = q + sign(q* - q)}^{q*} L(f, w, j / Q)

    Where q is the current position, q* the target position, Q the maximum position and L(f, w,
    m) the inverse price of `get_inverse_price`.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.6
      (Snippet 10.4).

    Args:
        target_position (int): The target position q*.
        current_position (int): The current position q.
        forecast_price (float): The forecast price f.
        width (float): The width (sigmoid) or exponent (power) w.
        maximum_position (int): The maximum position Q.
        method (str, optional): The function from the divergence to the bet size. Options are:
            "sigmoid" (sigmoid function), "power" (power function). Defaults to "sigmoid".

    Returns:
        float: The limit price, NaN when the position does not change.
    """
    if target_position == current_position:
        return np.nan

    sign = 1 if target_position > current_position else -1
    units = np.arange(current_position + sign, target_position + sign, sign)

    return float(
        np.mean(
            [
                get_inverse_price(
                    forecast_price, width, unit / maximum_position, method
                )
                for unit in units
            ]
        )
    )


def get_target_bet_sizes(
    market_price: pd.Series | float,
    forecast_price: pd.Series | float,
    current_position: pd.Series | int = 0,
    maximum_position: int = 100,
    calibration_divergence: float = 10.0,
    calibration_bet_size: float = 0.95,
    method: str = "sigmoid",
) -> pd.DataFrame:
    """
    Calculate the dynamic bet size of López de Prado: the bet size from the divergence of a forecast price from the
    market price, the target position it gives and the limit price of the order that reaches it.

    The function is calibrated so that a divergence of `calibration_divergence` gives a bet size of
    `calibration_bet_size`, and the position is updated as the market price moves towards or away from the forecast.

    The formula is as follows:

    - x = f - p, the forecast minus the market price
    - sigmoid: m = x / SQRT(w + x²) with w = x̄² x (1 / m̄² - 1)
    - power: m = sign(x) x |x|^w with w = ln(m̄) / ln(x̄), for x between -1 and 1
    - q* = int(m x Q), the target position (rounded towards zero)
    - limit price = the average of L(f, w, j / Q) for j from q + sign(q* - q) to q*

    Where x̄ and m̄ are the calibration divergence and bet size, q the current position and Q
    the maximum position.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 10.6
      (Snippet 10.4).

    Referenced against:
    - RiskLabAI, bet_size_sigmoid: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/bet_sizing.py#L347
    - RiskLabAI, target_position: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/bet_sizing.py#L369
    - RiskLabAI, limit_price: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/bet_sizing.py#L423

    Args:
        market_price (pd.Series | float): The market price p.
        forecast_price (pd.Series | float): The forecast price f.
        current_position (pd.Series | int, optional): The current position q. Defaults to 0.
        maximum_position (int, optional): The maximum position Q. Defaults to 100.
        calibration_divergence (float, optional): The divergence x̄ of the calibration.
            Defaults to 10.0.
        calibration_bet_size (float, optional): The bet size m̄ at that divergence. Defaults to
            0.95.
        method (str, optional): The function from the divergence to the bet size. Options are:
            "sigmoid" (sigmoid function), "power" (power function). Defaults to "sigmoid".

    Returns:
        pd.DataFrame: The Bet Size, Target Position and Limit Price per row.
    """
    if method == "sigmoid":
        width = get_sigmoid_width(calibration_divergence, calibration_bet_size)
    elif method == "power":
        width = get_power_width(calibration_divergence, calibration_bet_size)
    else:
        raise InvalidArgumentError("method needs to be 'sigmoid' or 'power'.")

    frame = pd.DataFrame(
        {
            "Market Price": market_price,
            "Forecast Price": forecast_price,
            "Position": current_position,
        },
        index=next(
            (
                value.index
                for value in (market_price, forecast_price, current_position)
                if isinstance(value, pd.Series)
            ),
            [0],
        ),
    )

    bet_size = get_dynamic_bet_size(
        (frame["Forecast Price"] - frame["Market Price"]).to_numpy(dtype=float),
        width,
        method,
    )
    target = np.trunc(bet_size * maximum_position).astype(int)
    limit = [
        get_limit_price(
            int(target_position),
            int(position),
            forecast,
            width,
            maximum_position,
            method,
        )
        for target_position, position, forecast in zip(
            target, frame["Position"], frame["Forecast Price"]
        )
    ]

    return pd.DataFrame(
        {"Bet Size": bet_size, "Target Position": target, "Limit Price": limit},
        index=frame.index,
    )


def get_bet_sizes(
    prices: pd.Series,
    labels: pd.DataFrame,
    method: str = "budget",
    step_size: float | None = None,
    minimum_labels: int = 20,
    runs: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Calculate the bet sizes of one ticker from its meta-labels: the budget, reserve or
    probability bet size, optionally rounded to a step size.

    Args:
        prices (pd.Series): The closes of the ticker, named after the ticker.
        labels (pd.DataFrame): Meta-labels with a Side, Holding Period and (for "probability")
            Label per date and ticker.
        method (str, optional): "budget", "reserve" or "probability". Defaults to "budget".
        step_size (float | None, optional): The step the bet sizes are rounded to. Defaults to
            None, unrounded.
        minimum_labels (int, optional): The number of closed bets the hit rate of
            "probability" needs before it sizes a bet. Defaults to 20.
        runs (int, optional): The number of random starting points of the EF3M fit of
            "reserve". Defaults to 100.
        seed (int, optional): The seed of the starting points of the EF3M fit. Defaults to 0.

    Returns:
        pd.DataFrame: The bet sizes of the ticker per date.
    """
    ticker = prices.name
    holding_periods = (
        labels["Holding Period"]
        .get(ticker, pd.Series(dtype=float))
        .reindex(prices.index)
    )
    sides = labels["Side"].get(ticker, pd.Series(dtype=float)).reindex(prices.index)
    if method == "budget":
        sizes = get_budget_bet_size(holding_periods, sides)
    elif method == "reserve":
        sizes = get_reserve_bet_size(holding_periods, sides, runs=runs, seed=seed)
    else:
        outcomes = labels["Label"].get(ticker, pd.Series(dtype=float))
        hit_rate = get_hit_rate(
            outcomes.reindex(prices.index), holding_periods, minimum_labels
        )
        # The bets placed while the hit rate is unknown are left out of the average.
        placed = sides.notna() & holding_periods.notna() & hit_rate.notna()
        bet_sizes = get_probability_bet_size(
            hit_rate.where(hit_rate > 0.5, 1 - hit_rate),  # noqa: PLR2004
            predictions=(hit_rate > 0.5).astype(float),  # noqa: PLR2004
            sides=sides,
        ).where(placed)
        sizes = get_concurrent_sides(holding_periods.where(placed), sides)
        sizes["Hit Rate"] = hit_rate
        sizes["Bet Size"] = get_average_active_bet_size(
            bet_sizes, holding_periods.where(placed)
        )
    if step_size is not None:
        sizes["Bet Size"] = get_discrete_bet_size(sizes["Bet Size"], step_size)
    return sizes
