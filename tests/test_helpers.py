# ruff: noqa

"""Helpers Tests"""

from unittest.mock import patch

import pandas as pd

from financetoolkit import helpers


def test_enrich_historical_data_basic():
    """Test basic enrichment of historical data."""
    data = pd.DataFrame(
        {"Adj Close": [100, 110, 121, 133]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = helpers.enrich_historical_data(data)

    # Check new columns
    assert "Return" in result.columns
    assert "Cumulative Return" in result.columns

    # Volatility is intentionally no longer calculated here, it lives in the Risk
    # module instead (risk_model.get_volatility)
    assert "Volatility" not in result.columns

    # Check return calculation
    assert pd.isna(result["Return"].iloc[0])
    assert abs(result["Return"].iloc[1] - 0.1) < 0.01


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
