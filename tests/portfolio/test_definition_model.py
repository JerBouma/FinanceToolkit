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


def _two_currency_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A US fund at $100 and a UK fund at £10, with the pound rising from $1.27."""
    days = pd.period_range("2021-01-01", "2021-12-31", freq="D")
    prices = pd.DataFrame({"US": 100.0, "UK": 10.0}, index=days)
    pounds_per_dollar = pd.DataFrame(
        {"USDGBP=X": 1 / np.linspace(1.27, 1.40, len(days))}, index=days
    )
    return prices, pounds_per_dollar


def test_build_transactions_converts_every_price_to_the_portfolio_currency():
    prices, rates = _two_currency_prices()
    definition = PortfolioDefinition(
        name="Two Currencies",
        holdings=[Holding("US", 0.5, currency="USD"), Holding("UK", 0.5)],
    )

    transactions = definition_model.build_transactions(
        definition,
        prices,
        currencies={"UK": "GBP"},
        initial_value=10000,
        currency="USD",
        exchange_rates=rates,
    )

    first = transactions[transactions["Date"] == "2021-01-01"].set_index("Identifier")
    assert set(transactions["Currency"]) == {"USD"}
    assert first.loc["UK", "Price"] == pytest.approx(12.7)
    value = first["Price"] * first["Volume"]
    assert (value / value.sum()).to_dict() == pytest.approx({"US": 0.5, "UK": 0.5})

    # The pound rose, so the rebalancing sells part of the UK fund at the dollar price of
    # that day and the weights are back at their targets in dollars.
    shares = transactions.groupby("Identifier")["Volume"].sum()
    rebalanced = pd.Period("2021-09-30", "D")
    dollars = pd.Series(
        {
            "US": 100.0,
            "UK": 10.0 / float(rates.loc[rebalanced, "USDGBP=X"]),
        }
    )
    weights = shares * dollars / (shares * dollars).sum()
    assert weights.to_dict() == pytest.approx({"US": 0.5, "UK": 0.5})


def test_build_transactions_converts_pence_to_pounds():
    prices, rates = _two_currency_prices()
    prices["UK"] = 1000.0
    definition = PortfolioDefinition.from_weights({"US": 0.5, "UK": 0.5})

    for pence in ("GBp", "GBX"):
        transactions = definition_model.build_transactions(
            definition,
            prices,
            currencies={"US": "USD", "UK": pence},
            initial_value=10000,
            exchange_rates=rates,
        )
        first = transactions[transactions["Date"] == "2021-01-01"].set_index(
            "Identifier"
        )
        assert first.loc["UK", "Price"] == pytest.approx(12.7)
        value = first["Price"] * first["Volume"]
        assert value["UK"] == pytest.approx(5000)


def test_build_transactions_in_another_currency_needs_its_exchange_rate():
    prices, _ = _two_currency_prices()
    definition = PortfolioDefinition.from_weights({"US": 0.5, "UK": 0.5})

    with pytest.raises(ValueError, match="USDGBP=X"):
        definition_model.build_transactions(
            definition, prices, currencies={"US": "USD", "UK": "GBP"}
        )

    # Valued in pounds instead: the dollar fund is converted with GBPUSD=X.
    dollars_per_pound = pd.DataFrame({"GBPUSD=X": 1.27}, index=prices.index)
    transactions = definition_model.build_transactions(
        definition,
        prices,
        currencies={"US": "USD", "UK": "GBP"},
        currency="GBP",
        exchange_rates=dollars_per_pound,
    )
    first = transactions.set_index("Identifier")
    assert set(transactions["Currency"]) == {"GBP"}
    assert first.loc["US", "Price"] == pytest.approx(100 / 1.27)


def test_the_presets_have_prices_in_their_own_currency():
    # Every classic portfolio is valued in dollars; the euro and pound bond funds are
    # converted, and their listings are ones the data providers carry.
    for identifier in ("conservative_income", "global_60_40"):
        definition = definition_model.find_preset(identifier)
        currencies = {
            holding.ticker: holding.currency for holding in definition.holdings
        }
        assert currencies["IEGA.AS"] == "EUR"
        assert currencies["IGLT.L"] == "GBP"


def test_build_transactions_reinvests_the_dividends_on_rebalancing():
    """Flat prices and one dividend of 5 per share: the yearly rebalancing invests the cash."""
    dates = pd.bdate_range("2022-01-03", "2023-12-29")
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    dividends = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    dividends.loc[pd.Timestamp("2022-06-01"), "A"] = 5.0
    definition = PortfolioDefinition.from_weights(
        {"A": 0.5, "B": 0.5}, rebalancing="yearly"
    )

    transactions = definition_model.build_transactions(
        definition, prices, initial_value=10000.0, currency="USD", dividends=dividends
    )

    # 50 shares of A received 250 of cash, reinvested half in A and half in B.
    rebalanced = transactions[transactions["Date"] == "2022-12-30"]
    assert rebalanced["Volume"].tolist() == pytest.approx([1.25, 1.25])
    assert (rebalanced["Price"] * rebalanced["Volume"]).sum() == pytest.approx(250.0)


def test_build_transactions_without_dividends_keeps_the_value():
    dates = pd.bdate_range("2022-01-03", "2023-12-29")
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    definition = PortfolioDefinition.from_weights(
        {"A": 0.5, "B": 0.5}, rebalancing="yearly"
    )

    transactions = definition_model.build_transactions(
        definition, prices, initial_value=10000.0, currency="USD"
    )

    assert len(transactions) == 2
