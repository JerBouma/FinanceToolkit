"""Portfolio Overview Model Tests"""

# ruff: noqa: PLR2004


import pandas as pd
import pytest

from financetoolkit.portfolio.overview_model import (
    create_portfolio_overview,
    create_profit_and_loss_overview,
    create_transactions_overview,
)


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    tickers = ["AAPL", "MSFT", "AMZN"]

    # Create MultiIndex for (Date, Ticker)
    index = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])

    return pd.DataFrame(
        {
            "Volume": [100, 50, 200, 0, 25, 100, 50, 0, 150],
            "Price": [150.0, 250.0, 100.0, 0.0, 260.0, 95.0, 155.0, 0.0, 98.0],
            "Costs": [1.0, 2.0, 3.0, 0.0, 1.5, 2.5, 1.0, 0.0, 2.0],
            "Name": ["Apple Inc"] * 3 + ["Microsoft Corp"] * 3 + ["Amazon.com Inc"] * 3,
        },
        index=index,
    )


@pytest.fixture
def sample_latest_returns():
    """Sample latest returns data"""
    return pd.Series([160.0, 255.0, 110.0], index=["AAPL", "MSFT", "AMZN"])


@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark data"""
    return pd.Series([500.0, 510.0, 520.0], index=["AAPL", "MSFT", "AMZN"])


@pytest.fixture
def sample_volatilities():
    """Sample volatilities data"""
    return pd.Series(
        [0.25, 0.30, 0.35, 0.20], index=["AAPL", "MSFT", "AMZN", "Benchmark"]
    )


@pytest.fixture
def sample_betas():
    """Sample betas data"""
    return pd.Series([1.2, 0.8, 1.5], index=["AAPL", "MSFT", "AMZN"])


def test_create_transactions_overview():
    """Test create_transactions_overview function"""
    portfolio_volume = pd.Series([100, 50, 200], index=["AAPL", "MSFT", "AMZN"])
    portfolio_price = pd.Series([150.0, 250.0, 100.0], index=["AAPL", "MSFT", "AMZN"])
    portfolio_costs = pd.Series([1.0, 2.0, 3.0], index=["AAPL", "MSFT", "AMZN"])
    latest_returns = pd.Series([160.0, 255.0, 110.0], index=["AAPL", "MSFT", "AMZN"])

    result = create_transactions_overview(
        portfolio_volume=portfolio_volume,
        portfolio_price=portfolio_price,
        portfolio_costs=portfolio_costs,
        latest_returns=latest_returns,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "Invested Amount" in result.columns
    assert "Current Value" in result.columns
    assert "% Return" in result.columns
    assert "Return" in result.columns

    # Check calculations
    expected_invested = portfolio_volume * portfolio_price + abs(portfolio_costs)
    expected_current = portfolio_volume * latest_returns
    expected_return_pct = (expected_current / expected_invested) - 1

    pd.testing.assert_series_equal(
        result["Invested Amount"], expected_invested, check_names=False
    )
    pd.testing.assert_series_equal(
        result["Current Value"], expected_current, check_names=False
    )
    pd.testing.assert_series_equal(
        result["% Return"], expected_return_pct, check_names=False
    )


def test_create_profit_and_loss_overview():
    """Test create_profit_and_loss_overview function"""
    transactions_overview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Ticker": ["AAPL", "AAPL", "AAPL"],
            "Volume": [100, -50, 25],  # Buy, sell, buy
            "Price": [150.0, 160.0, 155.0],
            "Costs": [1.0, 2.0, 1.5],
        }
    )

    result = create_profit_and_loss_overview(
        transactions_overview=transactions_overview,
        ticker_column="Ticker",
        volume_column="Volume",
        price_column="Price",
        method="FIFO",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "PnL" in result.columns
    assert "Cumulative PnL" in result.columns


def test_create_profit_and_loss_overview_lifo():
    """Test create_profit_and_loss_overview with LIFO method"""
    transactions_overview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Ticker": ["AAPL", "AAPL", "AAPL"],
            "Volume": [100, -50, 25],
            "Price": [150.0, 160.0, 155.0],
            "Costs": [1.0, 2.0, 1.5],
        }
    )

    result = create_profit_and_loss_overview(
        transactions_overview=transactions_overview,
        ticker_column="Ticker",
        volume_column="Volume",
        price_column="Price",
        method="LIFO",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "PnL" in result.columns
    assert "Cumulative PnL" in result.columns


def test_create_profit_and_loss_overview_average():
    """Test create_profit_and_loss_overview with AVERAGE method"""
    transactions_overview = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Ticker": ["AAPL", "AAPL", "AAPL"],
            "Volume": [100, -50, 25],
            "Price": [150.0, 160.0, 155.0],
            "Costs": [1.0, 2.0, 1.5],
        }
    )

    result = create_profit_and_loss_overview(
        transactions_overview=transactions_overview,
        ticker_column="Ticker",
        volume_column="Volume",
        price_column="Price",
        method="AVERAGE",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "PnL" in result.columns
    assert "Cumulative PnL" in result.columns


def test_create_portfolio_overview_with_missing_data():
    """Test create_portfolio_overview with missing data"""
    portfolio_name = pd.Series(["Apple Inc", "Microsoft Corp"], index=["AAPL", "MSFT"])
    portfolio_volume = pd.Series([100, 50], index=["AAPL", "MSFT"])
    portfolio_price = pd.Series([150.0, 250.0], index=["AAPL", "MSFT"])
    portfolio_costs = pd.Series([1.0, 2.0], index=["AAPL", "MSFT"])

    # Missing latest returns for MSFT
    latest_returns = pd.Series([160.0], index=["AAPL"])

    benchmark_prices = pd.Series([500.0, 510.0], index=["AAPL", "MSFT"])
    benchmark_latest_prices = pd.Series([520.0, 530.0], index=["AAPL", "MSFT"])
    volatilities = pd.Series([0.25, 0.30, 0.20], index=["AAPL", "MSFT", "Benchmark"])
    betas = pd.Series([1.2, 0.8], index=["AAPL", "MSFT"])

    with pytest.raises(KeyError):
        create_portfolio_overview(
            portfolio_name=portfolio_name,
            portfolio_volume=portfolio_volume,
            portfolio_price=portfolio_price,
            portfolio_costs=portfolio_costs,
            latest_returns=latest_returns,
            benchmark_prices=benchmark_prices,
            benchmark_latest_prices=benchmark_latest_prices,
            volatilities=volatilities,
            betas=betas,
            include_portfolio=True,
        )


def test_create_transactions_overview_with_negative_costs():
    """Test create_transactions_overview with negative costs"""
    portfolio_volume = pd.Series([100, 50], index=["AAPL", "MSFT"])
    portfolio_price = pd.Series([150.0, 250.0], index=["AAPL", "MSFT"])
    portfolio_costs = pd.Series([-1.0, -2.0], index=["AAPL", "MSFT"])  # Negative costs
    latest_returns = pd.Series([160.0, 255.0], index=["AAPL", "MSFT"])

    result = create_transactions_overview(
        portfolio_volume=portfolio_volume,
        portfolio_price=portfolio_price,
        portfolio_costs=portfolio_costs,
        latest_returns=latest_returns,
    )

    # Should handle negative costs by taking absolute value
    assert result.loc["AAPL", "Invested Amount"] == 100 * 150.0 + 1.0  # abs(-1.0)
    assert result.loc["MSFT", "Invested Amount"] == 50 * 250.0 + 2.0  # abs(-2.0)

    assert result.loc["MSFT", "Invested Amount"] == 50 * 250.0 + 2.0  # abs(-2.0)
