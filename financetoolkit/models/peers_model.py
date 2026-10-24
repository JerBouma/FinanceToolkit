"""Peer Comparison Model"""

__docformat__ = "google"

import pandas as pd


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
    return frame.apply(
        lambda row: row.dropna().iloc[-1] if row.notna().any() else float("nan"),
        axis=1,
    )


def get_peer_comparison(
    metrics: pd.DataFrame, company: str, peers: list[str]
) -> pd.DataFrame:
    """
    Compare a company with its peers on every metric: its value, the peers' median and
    mean, and its percentile among them (the share of peers with a lower value).

    Args:
        metrics (pd.DataFrame): The latest value per ticker (rows) and metric (columns).
        company (str): The company to compare.
        peers (list[str]): Its peers.

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
        rows[metric] = {
            "Company": value,
            "Peer Median": values.median() if not values.empty else float("nan"),
            "Peer Mean": values.mean() if not values.empty else float("nan"),
            "Percentile": (
                float((values < value).mean())
                if not values.empty and pd.notna(value)
                else float("nan")
            ),
            "Peers": float(len(values)),
        }

    return pd.DataFrame(rows).T
