"""Pairs Trading Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.econometrics.stationarity import cointegration_model

NO_REVERSION = 1e-10


def get_half_life(spread: pd.Series) -> float:
    """
    Calculate the half-life of a mean-reverting spread: the time it takes to close half the
    gap to its mean.

    The half-life follows from the AR(1) regression Δs_t = c + λ s_{t-1} + ε_t (the discretized
    Ornstein-Uhlenbeck process), as -ln(2) / ln(1 + λ).

    Args:
        spread (pd.Series): The spread.

    Returns:
        float: The half-life in periods, infinite when the spread does not revert.
    """
    values = spread.dropna().to_numpy(dtype=float)
    lagged, change = values[:-1], np.diff(values)
    design = np.column_stack([np.ones_like(lagged), lagged])
    (_, slope), *_ = np.linalg.lstsq(design, change, rcond=None)
    # A slope of (nearly) zero or above means the spread does not revert.
    if slope > -NO_REVERSION:
        return np.inf
    if slope <= -1:
        return 0.0
    return float(-np.log(2) / np.log(1 + slope))


def get_signals(
    z_score: pd.Series, entry: float = 2.0, exit_: float = 0.5
) -> pd.Series:
    """
    Calculate the position in the spread that z-score rules give: short, long or flat.

    The rules go short the spread (-1) once the
    z-score rises above the entry threshold, long (+1) once it falls below minus the
    threshold, and flat again once it crosses the exit threshold on its way back: a short
    position closes once the z-score falls below the exit threshold, a long one once it
    rises above minus the exit threshold. A z-score that jumps past the band in one step
    still closes the position, and when it lands beyond the entry threshold on the other
    side the position reverses at once.

    Args:
        z_score (pd.Series): The z-score of the spread.
        entry (float, optional): The entry threshold. Defaults to 2.0.
        exit_ (float, optional): The exit threshold. Defaults to 0.5.

    Returns:
        pd.Series: The position per date: -1, 0 or 1.
    """
    if not exit_ < entry:
        raise ValueError(
            f"The exit threshold ({exit_}) must lie below the entry threshold ({entry}), "
            "or a position would close on the date it opens."
        )
    position = 0
    positions = []
    for value in z_score.to_numpy(dtype=float):
        if np.isnan(value):
            positions.append(position)
            continue
        # Exiting on a crossing rather than on |z| < exit, so a z-score that jumps from
        # above the band to below it (2.5 to -0.7) is not held as a short indefinitely.
        if (position == -1 and value < exit_) or (position == 1 and value > -exit_):
            position = 0
        # A flat position, including one just closed, opens on the entry rule, so the
        # position on every date is the one the rules give for that date's z-score.
        if position == 0:
            if value > entry:
                position = -1
            elif value < -entry:
                position = 1
        positions.append(position)
    return pd.Series(positions, index=z_score.index, name="Signal")


def get_pairs_trading(
    dependent: pd.Series,
    independent: pd.Series,
    window: int | None = None,
    entry: float = 2.0,
    exit_: float = 0.5,
    log_prices: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Calculate the statistics of trading the spread between two prices: how far the spread is
    from its mean and the position that gives.

    The statistics are the hedge ratio from the regression of one (log) price on the other,
    the spread that remains, its z-score
    (against its full-sample mean and standard deviation, or a rolling window), the
    position the z-score rules give, the half-life of the spread and the Engle-Granger
    test of whether the two prices are cointegrated at all.

    The hedge ratio and intercept are estimated on the whole sample, so the spread, and
    with it every z-score and signal, uses information from after its date. A rolling
    window only makes the mean and standard deviation of the z-score backward-looking; a
    backtest without look-ahead estimates the hedge ratio on a formation period and trades
    in the period after it.

    For more information, see the following papers:

    - Gatev, E., Goetzmann, W.N. and Rouwenhorst, K.G. (2006). "Pairs Trading: Performance
      of a Relative-Value Arbitrage Rule." The Review of Financial Studies, 19(3), 797-827.
    - Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error Correction:
      Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

    Referenced against:
    - statsmodels, coint: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html

    Args:
        dependent (pd.Series): The price of the first asset.
        independent (pd.Series): The price of the second asset.
        window (int | None, optional): The rolling window of the z-score. Defaults to None,
            which uses the full sample.
        entry (float, optional): The z-score at which a position is opened. Defaults to 2.0.
        exit_ (float, optional): The z-score whose crossing on the way back closes a
            position (0 closes when the spread is back at its mean). Defaults to 0.5.
        log_prices (bool, optional): Whether to use log prices, so the hedge ratio is an
            elasticity. Defaults to True.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: Per date the Spread, Z-Score and Signal (NaN
        where the Z-Score is); and the Hedge Ratio, Intercept, Half-Life, Cointegration P-Value,
        the latest Z-Score and Signal and the Number of Trades (NaN without any Z-Score).
    """
    data = pd.concat([dependent, independent], axis=1).dropna()
    if log_prices:
        if (data <= 0).any().any():
            raise ValueError("Log prices need prices above zero; set log_prices=False.")
        data = np.log(data)
    if len(data) < 30:  # noqa: PLR2004
        raise ValueError("At least 30 observations are needed for a pairs trade.")

    y, x = data.iloc[:, 0], data.iloc[:, 1]
    design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    (intercept, hedge_ratio), *_ = np.linalg.lstsq(
        design, y.to_numpy(dtype=float), rcond=None
    )
    spread = y - hedge_ratio * x - intercept

    if window:
        mean = spread.rolling(window).mean()
        deviation = spread.rolling(window).std()
    else:
        mean, deviation = spread.mean(), spread.std()
    z_score = (spread - mean) / deviation
    positions = get_signals(z_score, entry, exit_)
    # A date without a z-score (the warm-up of the rolling window) has no signal rather than a flat one.
    signals = positions.where(z_score.notna())

    cointegration = cointegration_model.get_engle_granger_cointegration(y, x)

    series = pd.DataFrame({"Spread": spread, "Z-Score": z_score, "Signal": signals})
    summary = {
        "Hedge Ratio": float(hedge_ratio),
        "Intercept": float(intercept),
        "Half-Life": get_half_life(spread),
        "Cointegration P-Value": float(cointegration["P-Value"]),
        "Z-Score": float(z_score.iloc[-1]),
        "Signal": float(signals.iloc[-1]),
        # Every new position counts, including a reversal from short to long at once; without
        # any z-score there is nothing to count.
        "Number of Trades": (
            float(
                ((positions != 0) & (positions != positions.shift(fill_value=0))).sum()
            )
            if z_score.notna().any()
            else np.nan
        ),
    }
    return series, summary


# The fewest observations for an AR(1) fit of the Ornstein-Uhlenbeck process.
MINIMUM_ORNSTEIN_UHLENBECK_OBSERVATIONS = 30


def get_ornstein_uhlenbeck_fit(levels: pd.Series) -> dict[str, float]:
    """
    Fit the discrete Ornstein-Uhlenbeck process: how strongly a series reverts to its
    long-run level.

    The process p_t = c + φ·p_(t-1) + σ·ε_t is fitted by OLS, as López de Prado (2018,
    section 13.5) estimates it: φ the persistence, E[p] = c / (1 - φ)
    the level the process reverts to and σ the standard deviation of the residuals
    (n - 2 degrees of freedom).

    Shares similar logic:
    - RiskLabAI, fit_ornstein_uhlenbeck: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/ou_trading_rules.py#L269

    Args:
        levels (pd.Series): The levels (prices, log prices or a spread).

    Returns:
        dict[str, float]: The Phi, Intercept, Sigma, Long-Run Mean (NaN when φ ≥ 1, as the
        process does not revert), Half-Life (-ln(2) / ln(φ), see `get_half_life`) and the
        Last observation.
    """
    values = levels.dropna().to_numpy(dtype=float)
    if len(values) < MINIMUM_ORNSTEIN_UHLENBECK_OBSERVATIONS:
        raise ValueError(
            f"At least {MINIMUM_ORNSTEIN_UHLENBECK_OBSERVATIONS} observations are needed to "
            "fit an Ornstein-Uhlenbeck process."
        )

    # A constant series makes the regressors collinear, and the residuals of the fit are
    # exactly zero or rounding noise depending on the linear algebra library.
    if np.ptp(values) == 0:
        raise ValueError(
            "The series is constant, so it has no Ornstein-Uhlenbeck process."
        )

    design = np.column_stack([np.ones(len(values) - 1), values[:-1]])
    (intercept, phi), *_ = np.linalg.lstsq(design, values[1:], rcond=None)
    residuals = values[1:] - design @ np.array([intercept, phi])
    sigma = float(np.sqrt(residuals @ residuals / (len(residuals) - 2)))

    if sigma == 0:
        raise ValueError(
            "The series follows its AR(1) fit exactly, so it has no noise."
        )

    return {
        "Phi": float(phi),
        "Intercept": float(intercept),
        "Sigma": sigma,
        "Long-Run Mean": float(intercept / (1 - phi)) if phi < 1 else np.nan,
        "Half-Life": get_half_life(levels),
        "Last": float(values[-1]),
    }


def get_trading_rule_mesh(
    phi: float,
    drift: float,
    maximum_threshold: float = 10.0,
    grid_size: int = 20,
    maximum_holding_period: int = 100,
    simulations: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate the gains of a long position in a discrete Ornstein-Uhlenbeck process: the Sharpe
    ratio of every combination of profit-taking and stop-loss thresholds on a mesh.

    The simulation follows López de Prado (2018, Snippet 13.2), in units of the standard
    deviation σ of the shocks:

    - y_t = drift + φ·y_(t-1) + ε_t, y_0 = 0, ε_t ~ N(0, 1), for `simulations` paths of at most
      `maximum_holding_period` steps; with E[p] the long-run mean and p_0 the level at entry,
      drift = (1 - φ)·(E[p] - p_0) / σ
    - For every profit-taking π̄ and stop-loss π̲ on the mesh {h, 2h, ..., `maximum_threshold`} with
      h = `maximum_threshold` / `grid_size`: the exit is the first t with y_t > π̄ or y_t < -π̲, or
      t = `maximum_holding_period`, with the gain y_exit
    - Sharpe Ratio = mean / standard deviation (ddof=1) of the gains over the paths

    Every mesh point uses the same simulated paths (common random numbers), so the
    differences between neighbouring points are not simulation noise, and the paths come from
    `numpy.random.default_rng(seed)`, so the result is the same on every run.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 13
    (Snippets 13.1 and 13.2).

    Shares similar logic:
    - RiskLabAI, optimal_ou_trading_rule: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/ou_trading_rules.py#L198

    Args:
        phi (float): The persistence φ.
        drift (float): The constant of the process in units of σ.
        maximum_threshold (float, optional): The largest profit-taking and stop-loss in units
            of σ. Defaults to 10.0.
        grid_size (int, optional): The number of thresholds of each side of the mesh.
            Defaults to 20.
        maximum_holding_period (int, optional): The most steps a position is held. Defaults
            to 100.
        simulations (int, optional): The number of simulated paths. Defaults to 10000.
        seed (int, optional): The seed of the simulation. Defaults to 0.

    Returns:
        pd.DataFrame: Per mesh point (Profit-Taking, Stop-Loss) the Mean, Standard Deviation
        and Sharpe Ratio of the gains (in σ) and the Average Holding Period.

    Notes:
        Unlike López de Prado, who draws new paths for every mesh point, holds a position
        for up to one step more than the maximum (his loop exits when the holding period
        exceeds it) and takes the standard deviation with ddof=0, the mesh here uses common
        paths, the maximum holding period and ddof=1.
    """
    if grid_size < 1 or maximum_threshold <= 0:
        raise ValueError(
            "The mesh needs a grid_size of at least 1 and a positive maximum."
        )
    if maximum_holding_period < 1 or simulations < 2:  # noqa: PLR2004
        raise ValueError("The simulation needs at least one step and two paths.")

    # The shocks are turned into the path in place, step by step, to hold one array.
    gains = np.random.default_rng(seed).standard_normal(
        (simulations, maximum_holding_period)
    )
    gains[:, 0] += drift
    for step in range(1, maximum_holding_period):
        gains[:, step] += drift + phi * gains[:, step - 1]

    thresholds = np.linspace(
        maximum_threshold / grid_size, maximum_threshold, grid_size
    )
    last = maximum_holding_period - 1

    def first_crossings(crossed: np.ndarray) -> np.ndarray:
        """The first step of every path that crossed, or the last step when none did."""
        return np.where(crossed.any(axis=1), crossed.argmax(axis=1), last)

    profit_steps = np.stack([first_crossings(gains > level) for level in thresholds])
    loss_steps = np.stack([first_crossings(gains < -level) for level in thresholds])
    rows = np.arange(simulations)

    mesh = []
    for profit_taking, profit_step in zip(thresholds, profit_steps, strict=True):
        exits = np.minimum(profit_step[None, :], loss_steps)
        outcomes = gains[rows[None, :], exits]
        means = outcomes.mean(axis=1)
        deviations = outcomes.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe_ratios = np.where(deviations > 0, means / deviations, np.nan)
        for stop_loss, mean, deviation, sharpe_ratio, holding in zip(
            thresholds,
            means,
            deviations,
            sharpe_ratios,
            exits.mean(axis=1) + 1,
            strict=True,
        ):
            mesh.append(
                {
                    "Profit-Taking": float(profit_taking),
                    "Stop-Loss": float(stop_loss),
                    "Mean": float(mean),
                    "Standard Deviation": float(deviation),
                    "Sharpe Ratio": float(sharpe_ratio),
                    "Average Holding Period": float(holding),
                }
            )

    return pd.DataFrame(mesh).set_index(["Profit-Taking", "Stop-Loss"])


def get_optimal_trading_rule(
    levels: pd.Series,
    maximum_threshold: float = 10.0,
    grid_size: int = 20,
    maximum_holding_period: int = 100,
    simulations: int = 10000,
    seed: int = 0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Find the optimal trading rule of a mean-reverting series: the profit-taking and stop-loss
    thresholds with the highest Sharpe ratio per trade for a long position.

    The series follows a discrete Ornstein-Uhlenbeck process whose paths are simulated from
    the last observation, as López de Prado (2018, chapter 13) sets out: the process
    p_t = c + φ·p_(t-1) + σ·ε_t is fitted to the levels by OLS (see
    `get_ornstein_uhlenbeck_fit`) and `get_trading_rule_mesh` simulates it from the last
    level p_0 with drift = (c + (φ - 1)·p_0) / σ, which is (1 - φ)·(E[p] - p_0) / σ when the
    process reverts. The optimal rule is the first mesh point with the highest Sharpe ratio.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 13
    (Snippets 13.1 and 13.2).

    Referenced against:
    - RiskLabAI, ou_rule_metrics: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/ou_trading_rules.py#L148

    Shares similar logic:
    - RiskLabAI, fit_ornstein_uhlenbeck: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/ou_trading_rules.py#L269
    - RiskLabAI, optimal_ou_trading_rule: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/ou_trading_rules.py#L198

    Args:
        levels (pd.Series): The levels (log prices or a spread).
        maximum_threshold (float, optional): The largest profit-taking and stop-loss in units
            of σ. Defaults to 10.0.
        grid_size (int, optional): The number of thresholds of each side of the mesh.
            Defaults to 20.
        maximum_holding_period (int, optional): The most steps a position is held. Defaults
            to 100.
        simulations (int, optional): The number of simulated paths. Defaults to 10000.
        seed (int, optional): The seed of the simulation. Defaults to 0.

    Returns:
        tuple[dict[str, float], pd.DataFrame]: The Phi, Half-Life, Sigma and Long-Run Mean of
        the fit, the Forecast (E[p] - p_0) / σ, the optimal Profit-Taking and Stop-Loss (in σ)
        with the Mean and Standard Deviation of the gain (in σ), its Sharpe Ratio and the
        Average Holding Period; and the mesh of `get_trading_rule_mesh`.

    Notes:
        RiskLabAI's ou_rule_metrics gives the closed-form expected gain and exit time of the
        continuous process: with the thresholds moved out by 0.5826σ for the discrete monitoring (Broadie, Glasserman and
        Kou, 1997) and a long maximum holding period, the simulated mean gain agrees within
        0.1σ and the holding period within 6%.
    """
    fit = get_ornstein_uhlenbeck_fit(levels)
    sigma = fit["Sigma"]

    grid = get_trading_rule_mesh(
        fit["Phi"],
        (fit["Intercept"] + (fit["Phi"] - 1) * fit["Last"]) / sigma,
        maximum_threshold=maximum_threshold,
        grid_size=grid_size,
        maximum_holding_period=maximum_holding_period,
        simulations=simulations,
        seed=seed,
    )
    if grid["Sharpe Ratio"].isna().all():
        raise ValueError(
            "No mesh point has gains that vary, so no Sharpe ratio exists."
        )

    # The first mesh point with the highest Sharpe ratio, so a tie always resolves the same way.
    best = int(np.nanargmax(grid["Sharpe Ratio"].to_numpy()))
    (profit_taking, stop_loss), optimum = grid.index[best], grid.iloc[best]

    summary = {
        "Phi": fit["Phi"],
        "Half-Life": fit["Half-Life"],
        "Sigma": sigma,
        "Long-Run Mean": fit["Long-Run Mean"],
        "Forecast": (fit["Long-Run Mean"] - fit["Last"]) / sigma,
        "Profit-Taking": float(profit_taking),
        "Stop-Loss": float(stop_loss),
        "Mean": float(optimum["Mean"]),
        "Standard Deviation": float(optimum["Standard Deviation"]),
        "Sharpe Ratio": float(optimum["Sharpe Ratio"]),
        "Average Holding Period": float(optimum["Average Holding Period"]),
    }

    return summary, grid
