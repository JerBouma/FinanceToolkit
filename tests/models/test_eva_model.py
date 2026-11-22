"""EVA Model Tests""" ""
import pandas as pd
import pytest

from financetoolkit.models.valuation import eva_model


def test_get_net_operating_profit_after_taxes(recorder):
    recorder.capture(
        eva_model.get_net_operating_profit_after_taxes(
            ebit=pd.Series([100, 110, 120, 130, 80]),
            effective_tax_rate=pd.Series([0.2, 0.21, 0.19, 0.22, 0.18]),
        )
    )


def test_get_invested_capital(recorder):
    recorder.capture(
        eva_model.get_invested_capital(
            total_equity=pd.Series([500, 520, 540, 560, 580]),
            total_debt=pd.Series([200, 210, 220, 230, 240]),
        )
    )


def test_get_economic_value_added(recorder):
    recorder.capture(
        eva_model.get_economic_value_added(
            net_operating_profit_after_taxes=pd.Series([80, 87, 97, 101, 66]),
            weighted_average_cost_of_capital=pd.Series([0.08, 0.09, 0.085, 0.095, 0.1]),
            invested_capital=pd.Series([700, 730, 760, 790, 820]),
        )
    )


def test_get_market_value_added(recorder):
    recorder.capture(
        eva_model.get_market_value_added(
            market_value_of_equity=pd.Series([5000, 5200, 4800, 5300, 5600]),
            market_value_of_debt=pd.Series([1000, 1050, 1100, 1150, 1200]),
            invested_capital=pd.Series([3000, 3100, 3200, 3300, 3400]),
        )
    )
    # Synthetic ground-truth: MVA = (5000 + 1000) - 3000 = 3000
    assert (
        eva_model.get_market_value_added(
            market_value_of_equity=5000,
            market_value_of_debt=1000,
            invested_capital=3000,
        )
        == 3000
    )


def test_get_market_value_added_type_error():
    with pytest.raises(TypeError):
        eva_model.get_market_value_added(
            market_value_of_equity="not a number",
            market_value_of_debt=1000,
            invested_capital=3000,
        )


def test_get_economic_value_added_spread_times_capital_is_the_eva():
    spread = eva_model.get_economic_value_added_spread(
        return_on_invested_capital=0.15, weighted_average_cost_of_capital=0.09
    )
    assert spread == pytest.approx(0.06)
    assert spread * 1000 == pytest.approx(
        eva_model.get_economic_value_added(150, 0.09, 1000)
    )


def test_get_return_on_incremental_invested_capital_needs_growing_capital():
    result = eva_model.get_return_on_incremental_invested_capital(
        change_in_net_operating_profit_after_taxes=pd.Series([20.0, -5.0, 10.0]),
        change_in_invested_capital=pd.Series([100.0, -50.0, 0.0]),
    )
    assert result[0] == pytest.approx(0.2)
    # Shrinking or unchanged capital gives no return, not a positive or infinite one.
    assert result[1:].isna().all()


def test_get_research_and_development_asset_follows_damodarans_converter():
    # Three years of life: the asset holds the current year in full, last year at 2/3 and
    # the year before at 1/3; this year's amortization is a third of each of the three
    # years before.
    expenses = pd.DataFrame([[90.0, 120.0, 150.0, 180.0]], columns=range(4))
    asset, amortization = eva_model.get_research_and_development_asset(expenses, 3)
    assert asset.iloc[0, 3] == pytest.approx(180 + 150 * 2 / 3 + 120 / 3)
    assert amortization.iloc[0, 3] == pytest.approx((150 + 120 + 90) / 3)
    assert asset.iloc[0, :3].isna().all() and amortization.iloc[0, :3].isna().all()

    # Quarterly columns of trailing yearly expenses take every fourth column.
    quarterly = pd.DataFrame([[float(value) for value in range(1, 10)]])
    asset, amortization = eva_model.get_research_and_development_asset(
        quarterly, 2, periods_per_year=4
    )
    assert asset.iloc[0, 8] == pytest.approx(9 + 5 / 2)
    assert amortization.iloc[0, 8] == pytest.approx((5 + 1) / 2)

    with pytest.raises(ValueError):
        eva_model.get_research_and_development_asset(expenses, 0)
