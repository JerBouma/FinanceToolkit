"""Peer Comparison Model"""

__docformat__ = "google"

from collections.abc import Sequence

import numpy as np
import pandas as pd

# The metrics of the peer comparison that are valuation multiples, which are averaged with the
# harmonic mean of the positive values rather than the arithmetic mean.
VALUATION_MULTIPLES = (
    "Price-to-Earnings",
    "EV-to-EBITDA",
    "Price-to-Book",
    "Price-to-Sales",
)


def get_latest_values(frame: pd.DataFrame | pd.Series) -> pd.Series:
    """
    The latest reported value per ticker of a ratio (tickers by periods).

    Args:
        frame (pd.DataFrame | pd.Series): The ratio per ticker (rows) and period (columns).

    Returns:
        pd.Series: The last non-missing value per ticker.
    """
    if isinstance(frame, pd.Series):
        frame = frame.to_frame().T
    # A ratio that could not be calculated arrives without periods.
    if frame.columns.empty:
        return pd.Series(np.nan, index=frame.index)
    return frame.ffill(axis=1).iloc[:, -1]


def get_harmonic_mean(values: pd.Series) -> float:
    """
    The harmonic mean of the positive values, the average of a valuation multiple. A
    multiple divides a price by a fundamental, so its arithmetic mean is dominated by the
    companies with the smallest earnings (or book value, or sales), such as a
    Price-to-Earnings ratio of 200 on earnings close to zero. The harmonic mean is the
    inverse of the average yield (the fundamental divided by the price), where these
    companies weigh no more than others, and both papers below prefer it over the arithmetic
    mean. Negative multiples, from losses or a negative book value, say nothing about the
    price paid and are left out.

    For more information, see the following papers:

    - Baker, M. and Ruback, R.S. (1999). "Estimating Industry Multiples." Working paper,
      Harvard University.
    - Liu, J., Nissim, D. and Thomas, J. (2002). "Equity Valuation Using Multiples."
      Journal of Accounting Research, 40(1), 135-172.

    Args:
        values (pd.Series): The multiples.

    Returns:
        float: The harmonic mean of the positive values, or NaN when there are none.
    """
    positive = values[values > 0]
    return float(len(positive) / (1 / positive).sum()) if not positive.empty else np.nan


def get_peer_comparison(
    metrics: pd.DataFrame,
    company: str,
    peers: list[str],
    multiples: Sequence[str] = VALUATION_MULTIPLES,
) -> pd.DataFrame:
    """
    Compare a company with its peers on every metric: its value, the peers' median and
    mean, and its percentile among them (the share of peers with a lower value).

    The Peer Mean of a valuation multiple is the harmonic mean of the positive values (see
    `get_harmonic_mean`), the standard way to average multiples; for the other metrics it is
    the arithmetic mean. The Peer Median and Percentile use all values.

    Args:
        metrics (pd.DataFrame): The latest value per ticker (rows) and metric (columns).
        company (str): The company to compare.
        peers (list[str]): Its peers.
        multiples (Sequence[str], optional): The metrics that are valuation multiples (a
            price divided by a fundamental). Defaults to the Price-to-Earnings, EV-to-EBITDA,
            Price-to-Book and Price-to-Sales ratios.

    Returns:
        pd.DataFrame: Per metric the Company value, Peer Median, Peer Mean, Percentile and the
        number of Peers with a value.
    """
    peer_values = metrics.reindex([peer for peer in peers if peer != company])
    company_values = (
        metrics.loc[company]
        if company in metrics.index
        else pd.Series(float("nan"), index=metrics.columns)
    )

    rows = {}
    for metric in metrics.columns:
        values = pd.to_numeric(peer_values[metric], errors="coerce").dropna()
        value = pd.to_numeric(company_values[metric], errors="coerce")
        # The median and means of no values are NaN, as is the percentile of no company value.
        rows[metric] = {
            "Company": value,
            "Peer Median": values.median(),
            "Peer Mean": (
                get_harmonic_mean(values) if metric in multiples else values.mean()
            ),
            "Percentile": (
                float((values < value).mean()) if pd.notna(value) else float("nan")
            ),
            "Peers": float(len(values)),
        }

    return pd.DataFrame(rows).T
