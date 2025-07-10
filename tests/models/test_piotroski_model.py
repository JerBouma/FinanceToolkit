"""Piotroski Model Tests"""

# ruff: noqa: PLR2004

import pandas as pd
import pytest

from financetoolkit.models import piotroski_model


@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing"""
    return {
        "net_income": pd.Series(
            [100, 110, 120, 130, 140], index=["2019", "2020", "2021", "2022", "2023"]
        ),
        "total_assets": pd.Series(
            [1000, 1100, 1200, 1300, 1400],
            index=["2019", "2020", "2021", "2022", "2023"],
        ),
        "operating_cashflow": pd.Series(
            [120, 130, 140, 150, 160], index=["2019", "2020", "2021", "2022", "2023"]
        ),
        "total_debt": pd.Series(
            [400, 420, 440, 460, 480], index=["2019", "2020", "2021", "2022", "2023"]
        ),
        "current_assets": pd.Series(
            [300, 320, 340, 360, 380], index=["2019", "2020", "2021", "2022", "2023"]
        ),
        "current_liabilities": pd.Series(
            [200, 210, 220, 230, 240], index=["2019", "2020", "2021", "2022", "2023"]
        ),
        "common_stock_issued": pd.Series(
            [10, 5, 0, -5, -10], index=["2019", "2020", "2021", "2022", "2023"]
        ),
        "revenue": pd.Series(
            [800, 850, 900, 950, 1000], index=["2019", "2020", "2021", "2022", "2023"]
        ),
        "cost_of_goods_sold": pd.Series(
            [500, 520, 540, 560, 580], index=["2019", "2020", "2021", "2022", "2023"]
        ),
    }


def test_get_return_on_assets_criteria(sample_financial_data):
    """Test return on assets criteria calculation"""
    result = piotroski_model.get_return_on_assets_criteria(
        net_income=sample_financial_data["net_income"],
        average_total_assets=sample_financial_data["total_assets"],
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    # ROA should be positive for all years, so all criteria should be 1
    assert all(result == 1)


def test_get_return_on_assets_criteria_negative_income():
    """Test return on assets criteria with negative income"""
    net_income = pd.Series(
        [-100, 110, -120, 130, -140], index=["2019", "2020", "2021", "2022", "2023"]
    )
    total_assets = pd.Series(
        [1000, 1100, 1200, 1300, 1400], index=["2019", "2020", "2021", "2022", "2023"]
    )

    result = piotroski_model.get_return_on_assets_criteria(
        net_income=net_income, average_total_assets=total_assets
    )

    # Should be 0 for negative income years, 1 for positive
    expected = pd.Series(
        [False, True, False, True, False],
        index=["2019", "2020", "2021", "2022", "2023"],
    )
    pd.testing.assert_series_equal(result, expected)


def test_get_operating_cashflow_criteria(sample_financial_data):
    """Test operating cashflow criteria calculation"""
    result = piotroski_model.get_operating_cashflow_criteria(
        operating_cashflow=sample_financial_data["operating_cashflow"]
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    # Operating cashflow is positive for all years, so all criteria should be 1
    assert all(result == 1)


def test_get_operating_cashflow_criteria_negative():
    """Test operating cashflow criteria with negative cashflow"""
    operating_cashflow = pd.Series(
        [-120, 130, -140, 150, -160], index=["2019", "2020", "2021", "2022", "2023"]
    )

    result = piotroski_model.get_operating_cashflow_criteria(
        operating_cashflow=operating_cashflow
    )

    # Should be 0 for negative cashflow years, 1 for positive
    expected = pd.Series(
        [False, True, False, True, False],
        index=["2019", "2020", "2021", "2022", "2023"],
    )
    pd.testing.assert_series_equal(result, expected)


def test_get_accruals_criteria(sample_financial_data):
    """Test accruals criteria calculation"""
    result = piotroski_model.get_accruals_criteria(
        net_income=sample_financial_data["net_income"],
        average_total_assets=sample_financial_data["total_assets"],
        operating_cashflow=sample_financial_data["operating_cashflow"],
        total_assets=sample_financial_data["total_assets"],
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    # Since operating cashflow > net income, accruals should be positive, criteria should be 1
    assert all(result == 1)


def test_get_accruals_criteria_negative():
    """Test accruals criteria with negative accruals"""
    net_income = pd.Series(
        [150, 160, 170, 180, 190], index=["2019", "2020", "2021", "2022", "2023"]
    )
    operating_cashflow = pd.Series(
        [120, 130, 140, 150, 160], index=["2019", "2020", "2021", "2022", "2023"]
    )
    total_assets = pd.Series(
        [1000, 1100, 1200, 1300, 1400], index=["2019", "2020", "2021", "2022", "2023"]
    )

    result = piotroski_model.get_accruals_criteria(
        net_income=net_income,
        average_total_assets=total_assets,
        operating_cashflow=operating_cashflow,
        total_assets=total_assets,
    )

    # Since operating cashflow < net income, accruals should be negative, criteria should be 0
    assert all(result == 0)


def test_get_number_of_shares_criteria(sample_financial_data):
    """Test number of shares criteria calculation"""
    result = piotroski_model.get_number_of_shares_criteria(
        common_stock_issued=sample_financial_data["common_stock_issued"]
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    # Shares were issued in early years (positive), bought back in later years (negative)
    # Criteria should be 0 for positive values, 1 for negative values (buybacks)
    expected = pd.Series(
        [False, False, True, False, False],
        index=["2019", "2020", "2021", "2022", "2023"],
    )
    pd.testing.assert_series_equal(result, expected)


def test_get_number_of_shares_criteria_all_positive():
    """Test number of shares criteria with all positive values"""
    common_stock_issued = pd.Series(
        [10, 15, 20, 25, 30], index=["2019", "2020", "2021", "2022", "2023"]
    )

    result = piotroski_model.get_number_of_shares_criteria(
        common_stock_issued=common_stock_issued
    )

    # All positive (shares issued), so all criteria should be 0
    assert all(result == 0)


def test_get_piotroski_score():
    """Test Piotroski score calculation"""
    # Create test data where all criteria are 1
    criteria_data = pd.Series(
        [1, 1, 1, 1, 1], index=["2019", "2020", "2021", "2022", "2023"]
    )

    result = piotroski_model.get_piotroski_score(
        return_on_assets_criteria=criteria_data,
        operating_cashflow_criteria=criteria_data,
        change_in_return_on_asset_criteria=criteria_data,
        accruals_criteria=criteria_data,
        change_in_leverage_criteria=criteria_data,
        change_in_current_ratio_criteria=criteria_data,
        number_of_shares_criteria=criteria_data,
        gross_margin_criteria=criteria_data,
        asset_turnover_ratio_criteria=criteria_data,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    # All criteria are 1, so score should be 9
    assert all(result == 9)


def test_get_piotroski_score_mixed():
    """Test Piotroski score calculation with mixed criteria"""
    # Create test data with mixed criteria
    good_criteria = pd.Series(
        [1, 1, 1, 1, 1], index=["2019", "2020", "2021", "2022", "2023"]
    )
    bad_criteria = pd.Series(
        [0, 0, 0, 0, 0], index=["2019", "2020", "2021", "2022", "2023"]
    )

    result = piotroski_model.get_piotroski_score(
        return_on_assets_criteria=good_criteria,
        operating_cashflow_criteria=good_criteria,
        change_in_return_on_asset_criteria=good_criteria,
        accruals_criteria=good_criteria,
        change_in_leverage_criteria=good_criteria,
        change_in_current_ratio_criteria=bad_criteria,
        number_of_shares_criteria=bad_criteria,
        gross_margin_criteria=bad_criteria,
        asset_turnover_ratio_criteria=bad_criteria,
    )

    # 5 criteria are 1, 4 criteria are 0, so score should be 5
    assert all(result == 5)


def test_edge_cases_zero_values():
    """Test edge cases with zero values"""
    # Test with zero net income
    zero_net_income = pd.Series(
        [0, 0, 0, 0, 0], index=["2019", "2020", "2021", "2022", "2023"]
    )
    total_assets = pd.Series(
        [1000, 1100, 1200, 1300, 1400], index=["2019", "2020", "2021", "2022", "2023"]
    )

    result = piotroski_model.get_return_on_assets_criteria(
        net_income=zero_net_income, average_total_assets=total_assets
    )

    # Zero net income should result in 0 criteria
    assert all(result == 0)


def test_edge_cases_empty_series():
    """Test edge cases with empty series"""
    empty_series = pd.Series([], dtype=float)

    result = piotroski_model.get_return_on_assets_criteria(
        net_income=empty_series, average_total_assets=empty_series
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 0
