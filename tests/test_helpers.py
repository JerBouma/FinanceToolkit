# ruff: noqa

"""Helpers Tests"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import requests

from financetoolkit import helpers


def test_calculate_growth_basic():
    """Test basic growth calculation with default parameters."""
    data = pd.DataFrame(
        {"2020": [100, 200], "2021": [110, 220], "2022": [121, 242]},
        index=["Revenue", "Expenses"],
    )

    result = helpers.calculate_growth(data)

    expected_values = np.array([[np.nan, 0.1, 0.1], [np.nan, 0.1, 0.1]])

    # Check shape
    assert result.shape == data.shape

    # Check that first column is NaN (no previous period)
    assert result.iloc[:, 0].isna().all()

    # Check growth values
    np.testing.assert_array_almost_equal(
        result.iloc[:, 1:].values, expected_values[:, 1:]
    )


def test_calculate_growth_with_lag():
    """Test growth calculation with custom lag."""
    data = pd.DataFrame(
        {"2020": [100], "2021": [110], "2022": [121], "2023": [133]}, index=["Revenue"]
    )

    result = helpers.calculate_growth(data, lag=2)

    # First two values should be NaN
    assert result.iloc[0, 0:2].isna().all()

    # Third value should be (121-100)/100 = 0.21
    assert abs(result.iloc[0, 2] - 0.21) < 0.01


def test_calculate_growth_with_list_lag():
    """Test growth calculation with list of lags."""
    data = pd.DataFrame(
        {"2020": [100], "2021": [110], "2022": [121], "2023": [133]}, index=["Revenue"]
    )

    result = helpers.calculate_growth(data, lag=[1, 2])

    # Check that result has MultiIndex
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == [None, None]

    # Check lag structure
    assert ("Revenue", "Lag 1") in result.index
    assert ("Revenue", "Lag 2") in result.index


def test_calculate_growth_axis_parameter():
    """Test growth calculation with axis parameter."""
    data = pd.DataFrame(
        {"2020": [100, 200], "2021": [110, 220], "2022": [121, 242]},
        index=["Revenue", "Expenses"],
    )

    result = helpers.calculate_growth(data, axis="index")

    # Check shape
    assert result.shape == data.shape

    # When axis="index", growth is calculated along rows
    assert result.columns.equals(data.columns)


def test_calculate_growth_with_rounding():
    """Test growth calculation with custom rounding."""
    data = pd.DataFrame({"2020": [100], "2021": [110.12345]}, index=["Revenue"])

    result = helpers.calculate_growth(data, rounding=2)

    # Check rounding
    assert result.iloc[0, 1] == 0.10


def test_calculate_growth_with_missing_data():
    """Test growth calculation with missing data."""
    data = pd.DataFrame(
        {"2020": [100, np.nan], "2021": [110, 220], "2022": [121, 242]},
        index=["Revenue", "Expenses"],
    )

    result = helpers.calculate_growth(data)

    # Should handle NaN values properly
    assert not result.isna().all().all()


def test_calculate_growth_warnings_suppression():
    """Test that FutureWarning is suppressed."""
    data = pd.DataFrame({"2020": [100], "2021": [110]}, index=["Revenue"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = helpers.calculate_growth(data)

        # Check that no FutureWarnings were raised
        future_warnings = [
            warning for warning in w if issubclass(warning.category, FutureWarning)
        ]
        assert len(future_warnings) == 0


def test_combine_dataframes():
    """Test combining dataframes from different companies."""
    df1 = pd.DataFrame(
        {"2020": [100, 50], "2021": [110, 55]}, index=["Revenue", "Profit"]
    )

    df2 = pd.DataFrame(
        {"2020": [200, 80], "2021": [220, 88]}, index=["Revenue", "Profit"]
    )

    dataset_dict = {"AAPL": df1, "MSFT": df2}

    result = helpers.combine_dataframes(dataset_dict)

    # Check structure
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.levels[0].tolist() == ["AAPL", "MSFT"]
    assert result.index.levels[1].tolist() == ["Profit", "Revenue"]

    # Check values
    assert result.loc[("AAPL", "Revenue"), "2020"] == 100
    assert result.loc[("MSFT", "Revenue"), "2020"] == 200


def test_combine_dataframes_single_ticker():
    """Test combining single ticker."""
    df1 = pd.DataFrame({"2020": [100], "2021": [110]}, index=["Revenue"])

    dataset_dict = {"AAPL": df1}

    result = helpers.combine_dataframes(dataset_dict)

    assert isinstance(result.index, pd.MultiIndex)
    assert result.loc[("AAPL", "Revenue"), "2020"] == 100


def test_equal_length_first_dataset_starts_later():
    """Test equal_length when first dataset starts later."""
    # Create mock dataframes with different starting years
    df1 = pd.DataFrame([[100]], columns=["2022"])
    df2 = pd.DataFrame([[200]], columns=["2020"])

    # Mock the insert method
    df1.insert = MagicMock()
    df1.sort_index = MagicMock(return_value=df1)

    result1, result2 = helpers.equal_length(df1, df2)

    # Should have called insert to add earlier years
    assert df1.insert.call_count == 2  # 2021 and 2020
    df1.sort_index.assert_called_once()


def test_equal_length_second_dataset_starts_later():
    """Test equal_length when second dataset starts later."""
    df1 = pd.DataFrame([[100]], columns=["2020"])
    df2 = pd.DataFrame([[200]], columns=["2022"])

    # Mock the insert method
    df2.insert = MagicMock()
    df2.sort_index = MagicMock(return_value=df2)

    result1, result2 = helpers.equal_length(df1, df2)

    # Should have called insert to add earlier years
    assert df2.insert.call_count == 2  # 2021 and 2020
    df2.sort_index.assert_called_once()


def test_equal_length_same_start():
    """Test equal_length when both datasets start at same time."""
    df1 = pd.DataFrame([[100]], columns=["2020"])
    df2 = pd.DataFrame([[200]], columns=["2020"])

    result1, result2 = helpers.equal_length(df1, df2)

    # Should return original dataframes
    assert result1.equals(df1)
    assert result2.equals(df2)


def test_convert_isin_to_ticker_valid_isin():
    """Test converting valid ISIN to ticker."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"quotes": [{"symbol": "AAPL"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        with patch("financetoolkit.helpers.logger") as mock_logger:
            result = helpers.convert_isin_to_ticker("US0378331005")

            assert result == "AAPL"
            mock_logger.info.assert_called_once()


def test_convert_isin_to_ticker_invalid_format():
    """Test converting invalid ISIN format."""
    result = helpers.convert_isin_to_ticker("INVALID")

    assert result == "INVALID"


def test_convert_isin_to_ticker_no_quotes():
    """Test converting ISIN when no quotes found."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"quotes": []}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        with patch("financetoolkit.helpers.logger") as mock_logger:
            result = helpers.convert_isin_to_ticker("US0378331005")

            assert result == "US0378331005"
            mock_logger.warning.assert_called_once()


def test_convert_isin_to_ticker_request_exception():
    """Test converting ISIN when request fails."""
    with patch(
        "requests.get",
        side_effect=requests.exceptions.RequestException("Network error"),
    ):
        with patch("financetoolkit.helpers.logger") as mock_logger:
            result = helpers.convert_isin_to_ticker("US0378331005")

            assert result == "US0378331005"
            mock_logger.warning.assert_called_once()


def test_convert_isin_to_ticker_json_parse_error():
    """Test converting ISIN when JSON parsing fails."""
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        with patch("financetoolkit.helpers.logger") as mock_logger:
            result = helpers.convert_isin_to_ticker("US0378331005")

            assert result == "US0378331005"
            mock_logger.warning.assert_called_once()


def test_enrich_historical_data_basic():
    """Test basic enrichment of historical data."""
    data = pd.DataFrame(
        {"Adj Close": [100, 110, 121, 133]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = helpers.enrich_historical_data(data)

    # Check new columns
    assert "Return" in result.columns
    assert "Volatility" in result.columns
    assert "Cumulative Return" in result.columns

    # Check return calculation
    assert pd.isna(result["Return"].iloc[0])
    assert abs(result["Return"].iloc[1] - 0.1) < 0.01


def test_enrich_historical_data_with_risk_free_rate():
    """Test enrichment with risk-free rate."""
    data = pd.DataFrame(
        {"Adj Close": [100, 110, 121, 133]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    risk_free_rate = pd.DataFrame(
        {"Adj Close": [0.01, 0.01, 0.01, 0.01]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = helpers.enrich_historical_data(data, risk_free_rate=risk_free_rate)

    # Check excess return columns
    assert "Excess Return" in result.columns
    assert "Excess Volatility" in result.columns


def test_enrich_historical_data_with_date_range():
    """Test enrichment with start and end dates."""
    data = pd.DataFrame(
        {"Adj Close": [100, 110, 121, 133, 146]},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )

    result = helpers.enrich_historical_data(data, start="2020-01-02", end="2020-01-04")

    # Should still have all rows but calculations based on subset
    assert len(result) == 5
    assert "Return" in result.columns


def test_enrich_historical_data_custom_return_column():
    """Test enrichment with custom return column."""
    data = pd.DataFrame(
        {"Close": [100, 110, 121, 133], "Adj Close": [90, 99, 109, 120]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = helpers.enrich_historical_data(data, return_column="Close")

    # Should use Close for calculations
    assert abs(result["Return"].iloc[1] - 0.1) < 0.01


def test_handle_portfolio_decorator_basic():
    """Test handle_portfolio decorator basic functionality."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = False
            self._portfolio_weights = {
                "yearly": pd.DataFrame(
                    {"AAPL": [0.6, 0.6, 0.6], "MSFT": [0.4, 0.4, 0.4]},
                    index=pd.PeriodIndex(["2020", "2021", "2022"], freq="Y"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self):
        return pd.DataFrame(
            {"AAPL": [10, 20, 30], "MSFT": [15, 25, 35]},
            index=pd.PeriodIndex(["2020", "2021", "2022"], freq="Y"),
        )

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should have Portfolio column
    assert "Portfolio" in result.columns

    # Check weighted average calculation
    expected_portfolio = 0.6 * 10 + 0.4 * 15  # First row: 0.6*10 + 0.4*15 = 12
    assert abs(result["Portfolio"].iloc[0] - expected_portfolio) < 0.01


def test_handle_portfolio_decorator_with_benchmark():
    """Test handle_portfolio decorator excludes benchmark."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = False
            self._portfolio_weights = {
                "yearly": pd.DataFrame(
                    {"AAPL": [0.6, 0.6], "MSFT": [0.4, 0.4]},
                    index=pd.PeriodIndex(["2020", "2021"], freq="Y"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self):
        return pd.DataFrame(
            {"AAPL": [10, 20], "MSFT": [15, 25], "Benchmark": [12, 22]},
            index=pd.PeriodIndex(["2020", "2021"], freq="Y"),
        )

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should have Portfolio column
    assert "Portfolio" in result.columns
    # Should still have Benchmark column
    assert "Benchmark" in result.columns

    # Portfolio calculation should exclude benchmark
    expected_portfolio = 0.6 * 10 + 0.4 * 15  # Should be 12
    assert abs(result["Portfolio"].iloc[0] - expected_portfolio) < 0.01


def test_handle_portfolio_decorator_no_portfolio():
    """Test handle_portfolio decorator when Portfolio not in tickers."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT"]
            self._rounding = 4

    @helpers.handle_portfolio
    def test_function(self):
        return pd.DataFrame({"AAPL": [10, 20], "MSFT": [15, 25]})

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should not have Portfolio column
    assert "Portfolio" not in result.columns
    assert len(result.columns) == 2


def test_handle_portfolio_decorator_quarterly():
    """Test handle_portfolio decorator with quarterly data."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = True
            self._portfolio_weights = {
                "quarterly": pd.DataFrame(
                    {"AAPL": [0.6, 0.6], "MSFT": [0.4, 0.4]},
                    index=pd.PeriodIndex(["2020Q1", "2020Q2"], freq="Q"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self):
        return pd.DataFrame(
            {"AAPL": [10, 20], "MSFT": [15, 25]},
            index=pd.PeriodIndex(["2020Q1", "2020Q2"], freq="Q"),
        )

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should have Portfolio column
    assert "Portfolio" in result.columns


def test_handle_portfolio_decorator_with_growth_warning():
    """Test handle_portfolio decorator shows warning for growth with multiple lags."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = False
            self._portfolio_weights = {
                "yearly": pd.DataFrame(
                    {"AAPL": [0.6], "MSFT": [0.4]},
                    index=pd.PeriodIndex(["2020"], freq="Y"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self, growth=False, lag=1):
        return pd.DataFrame(
            {"AAPL": [10], "MSFT": [15]}, index=pd.PeriodIndex(["2020"], freq="Y")
        )

    mock_self = MockSelf()

    with patch("financetoolkit.helpers.logger") as mock_logger:
        result = test_function(mock_self, growth=True, lag=[1, 2])

        mock_logger.warning.assert_called_once()
        assert "multiple lags" in mock_logger.warning.call_args[0][0]


def test_handle_portfolio_decorator_non_dataframe_result():
    """Test handle_portfolio decorator with non-DataFrame result."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4

    @helpers.handle_portfolio
    def test_function(self):
        return "Not a DataFrame"

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should return original result unchanged
    assert result == "Not a DataFrame"


def test_handle_portfolio_decorator_multiindex_columns():
    """Test handle_portfolio decorator with MultiIndex columns."""

    class MockSelf:
        def __init__(self):
            self._tickers = ["AAPL", "MSFT", "Portfolio"]
            self._rounding = 4
            self._quarterly = False
            self._portfolio_weights = {
                "yearly": pd.DataFrame(
                    {"AAPL": [0.6], "MSFT": [0.4]},
                    index=pd.PeriodIndex(["2020"], freq="Y"),
                )
            }

    @helpers.handle_portfolio
    def test_function(self):
        columns = pd.MultiIndex.from_tuples([("AAPL", "Value"), ("MSFT", "Value")])
        return pd.DataFrame(
            {("AAPL", "Value"): [10], ("MSFT", "Value"): [15]},
            index=pd.PeriodIndex(["2020"], freq="Y"),
            columns=columns,
        )

    mock_self = MockSelf()
    result = test_function(mock_self)

    # Should not add Portfolio column for MultiIndex columns
    assert len(result.columns) == 2
