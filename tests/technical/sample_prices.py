"""Sample prices for the technical model tests, long enough for every window to fill."""

import numpy as np
import pandas as pd

LENGTH = 60


def sample_prices() -> pd.DataFrame:
    """A deterministic series of 60 daily bars that trends up with swings in both directions."""
    steps = np.arange(LENGTH)
    close = 100 + 0.3 * steps + 4 * np.sin(steps / 3)
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": close + 1 + 0.5 * np.cos(steps / 2),
            "Low": close - 1 - 0.5 * np.sin(steps / 2) ** 2,
            "Close": close,
            "Volume": 1_000 + 100 * (steps % 7),
        },
        index=pd.period_range("2024-01-01", periods=LENGTH, freq="D"),
    )
