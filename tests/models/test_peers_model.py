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
