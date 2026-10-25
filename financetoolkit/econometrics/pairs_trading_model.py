"""Pairs Trading Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.econometrics import cointegration_model

NO_REVERSION = 1e-10


def get_half_life(spread: pd.Series) -> float:
    """
    The half-life of a mean-reverting spread: the time it takes to close half the gap to
    its mean, from the AR(1) regression Δs_t = c + λ s_{t-1} + ε_t (the discretized
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
    The position in the spread that z-score rules give: short the spread (-1) once the
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
    The statistics of trading the spread between two prices: the hedge ratio from the
    regression of one (log) price on the other, the spread that remains, its z-score
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
        tuple[pd.DataFrame, dict[str, float]]: Per date the Spread, Z-Score and Signal; and
        the Hedge Ratio, Intercept, Half-Life, Cointegration P-Value, the latest Z-Score and
        Signal and the Number of Trades.
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
    signals = get_signals(z_score, entry, exit_)

    cointegration = cointegration_model.get_engle_granger_cointegration(y, x)

    series = pd.DataFrame({"Spread": spread, "Z-Score": z_score, "Signal": signals})
    summary = {
        "Hedge Ratio": float(hedge_ratio),
        "Intercept": float(intercept),
        "Half-Life": get_half_life(spread),
        "Cointegration P-Value": float(cointegration["P-Value"]),
        "Z-Score": float(z_score.iloc[-1]),
        "Signal": float(signals.iloc[-1]),
        # Every new position counts, including a reversal from short to long at once.
        "Number of Trades": float(
            ((signals != 0) & (signals != signals.shift(fill_value=0))).sum()
        ),
    }
    return series, summary
