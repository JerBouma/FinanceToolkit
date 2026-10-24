"""Portfolio Definition Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.portfolio import definition_model
from financetoolkit.portfolio.definition_model import Holding, PortfolioDefinition

# pylint: disable=missing-function-docstring


def test_every_preset_is_valid_and_sourced():
    presets = definition_model.load_presets()

    assert len(presets) == 13  # noqa: PLR2004
    for identifier, definition in presets.items():
        assert definition.identifier == identifier
        assert sum(definition.weights.values()) == pytest.approx(1, abs=1e-3)
        assert definition.description
        assert "text" in definition.source
        for holding in definition.holdings:
            assert holding.asset_class in {
                "Equity",
                "Fixed Income",
                "Commodity",
                "Cash",
            }
            assert len(holding.currency) == 3  # noqa: PLR2004


def test_presets_are_found_by_alias_and_name():
    assert definition_model.find_preset("60/40").identifier == "balanced_60_40"
    assert definition_model.find_preset("All Weather").identifier == "all_weather"
    assert (
        definition_model.find_preset("three-fund").identifier == "bogleheads_three_fund"
    )
    with pytest.raises(ValueError, match="No classic portfolio"):
        definition_model.find_preset("Moon Portfolio")


def test_definition_round_trips_through_yaml():
    definition = definition_model.find_preset("Permanent Portfolio")
    again = PortfolioDefinition.from_yaml(definition.to_yaml())

    assert again.weights == definition.weights
    assert again.holdings[1].duration == definition.holdings[1].duration
    assert again.source == definition.source


def test_glidepath_weights_are_interpolated():
    glidepath = definition_model.find_preset("target date")

    assert glidepath.weights_at(0) == {"VTI": 0.9, "BND": 0.1}
    assert glidepath.weights_at(17.5)["VTI"] == pytest.approx(0.9 - 0.4 * 12.5 / 25)
    assert glidepath.weights_at(50) == {"VTI": 0.3, "BND": 0.7}


def test_weights_are_parsed_in_every_form():
    expected = {"VTI": 0.6, "BND": 0.4}
    assert definition_model.parse_weights("VTI:0.6, BND:0.4") == pytest.approx(expected)
    assert definition_model.parse_weights("VTI:60%; BND:40%") == pytest.approx(expected)
    assert definition_model.parse_weights({"VTI": 3, "BND": 2}) == pytest.approx(
        expected
    )
    table = pd.DataFrame({"Weight": [0.6, 0.4, 1.0]}, index=["VTI", "BND", "Portfolio"])
    assert definition_model.parse_weights(table) == pytest.approx(expected)


def test_invalid_definitions_are_refused():
    with pytest.raises(ValueError, match="sum to"):
        PortfolioDefinition(name="Bad", holdings=[Holding("A", 0.5)])
    with pytest.raises(ValueError, match="rebalancing"):
        PortfolioDefinition(
            name="Bad", holdings=[Holding("A", 1.0)], rebalancing="daily"
        )


def test_build_transactions_rebalances_to_the_targets():
    days = pd.period_range("2020-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(days))),
            "B": 50 * np.cumprod(1 + rng.normal(0.0, 0.005, len(days))),
        },
        index=days,
    )
    definition = PortfolioDefinition.from_weights({"A": 0.6, "B": 0.4})

    transactions = definition_model.build_transactions(
        definition, prices, initial_value=1000
    )

    # The purchase and three quarterly rebalances (the running quarter is not rebalanced).
    assert sorted(transactions["Date"].unique()) == [
        "2020-01-01",
        "2020-03-31",
        "2020-06-30",
        "2020-09-30",
    ]
    first = transactions[transactions["Date"] == "2020-01-01"]
    assert (first["Price"] * first["Volume"]).sum() == pytest.approx(1000)

    shares = transactions.groupby("Identifier")["Volume"].sum()
    value = shares * prices.loc[pd.Period("2020-09-30", "D")]
    assert (value / value.sum())["A"] == pytest.approx(0.6)
