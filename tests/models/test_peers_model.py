"""Peers Model Tests"""

import pandas as pd
import pytest

from financetoolkit.models import peers_model

# pylint: disable=missing-function-docstring


def test_peer_comparison():
    metrics = pd.DataFrame(
        {"P/E": [30.0, 10.0, 20.0, 40.0], "Margin": [0.3, 0.1, 0.2, None]},
        index=["A", "B", "C", "D"],
    )
    result = peers_model.get_peer_comparison(metrics, "A", ["B", "C", "D"])

    assert result.loc["P/E", "Peer Median"] == 20
    assert result.loc["P/E", "Percentile"] == pytest.approx(2 / 3)
    assert result.loc["Margin", "Peers"] == 2
    assert result.loc["Margin", "Percentile"] == 1


def test_latest_values():
    frame = pd.DataFrame({"2022": [1.0, 2.0], "2023": [3.0, None]}, index=["A", "B"])
    assert peers_model.get_latest_values(frame).tolist() == [3.0, 2.0]


def test_peer_mean_of_multiples_is_the_harmonic_mean_of_positive_values():
    metrics = pd.DataFrame(
        {
            "Price-to-Earnings": [15.0, 10.0, 40.0, -20.0],
            "Gross Margin": [0.3, 0.1, 0.4, -0.2],
        },
        index=["A", "B", "C", "D"],
    )
    result = peers_model.get_peer_comparison(metrics, "A", ["B", "C", "D"])

    # The loss maker's negative multiple is left out and the high multiple weighs less.
    assert result.loc["Price-to-Earnings", "Peer Mean"] == pytest.approx(
        2 / (1 / 10 + 1 / 40)
    )
    assert result.loc["Price-to-Earnings", "Peer Median"] == 10
    assert result.loc["Gross Margin", "Peer Mean"] == pytest.approx(0.1)
    assert pd.isna(peers_model.get_harmonic_mean(pd.Series([-1.0, -2.0])))


def test_a_ratio_without_periods_is_missing():
    latest = peers_model.get_latest_values(pd.Series(dtype=object))

    assert latest.isna().all()
